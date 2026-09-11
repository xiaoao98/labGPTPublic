"""LabGPT chat loop, on retrieval.

WHAT CHANGED, and why the shape of this file is so much smaller than it was.

It used to classify a query three ways and then paste whole corpora into the prompt:

    3 classifier calls, each budgeted 32768 tokens of generation for a one-word answer
      safety   -> the entire safety corpus
      protocol -> an LLM picks an experiment name from a list, SQL fetches that one row
      papers   -> the entire abstract index, and an LLM returns five ids from it
      members  -> the entire directory, on every query, unconditionally
    -> one very large prompt -> generate

On the real corpus a paper question assembled tens of thousands of tokens before
generation started, and none of it could be measured: nothing in the pipeline produced a
relevance score, so there was no way to say whether a retrieved document was any good and
no way to refuse when none of them were.

Now:

    1 classifier call, only to decide whether to search the web, which is a real
      external call worth gating
    hybrid retrieval over one index -> top k documents
    abstention gate on the retrieval score, before a token is spent
    generate with numbered sources and required citations
    citation validation, and withhold the answer if none resolve

Two behaviours are deliberately preserved. The web search leg is untouched. And the
directory still contributes, because the original prompt asked the assistant to name
someone who can help; instead of pasting all of it, the top matching people are retrieved
alongside, which keeps the behaviour at a fraction of the tokens.

The old per-domain scripts still work on their own: protocolDemo.py, safetyDemo.py,
memberInfoDemo.py, paperDemo.py. They are unchanged and still stuff their own corpus.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os
from rich import print as rprint
import labgpt_config as cfg
import labgpt_metrics as metrics

from labrag.answerer import (
    DEFAULT_ABSTAIN_COSINE, TransformersChatClient, should_abstain,
    validate_citations, count_uncited_sentences,
)
from labrag.embeddings import Embedder
from labrag.prompts import ABSTENTION_TEMPLATE, build_messages
from labrag.retriever import Retriever
from labrag.store import VectorStore

model_name = cfg.MODEL_NAME

# Generation caps. Every one of these was 32768 or 65536, for answers measured in words.
MAX_NEW_TOKENS_YESNO = 8      # "yes" / "no"
MAX_NEW_TOKENS_ANSWER = 1024  # the user-facing answer

# How many documents go into the prompt, and how many of them may be people. The
# directory is retrieved separately so that a protocol question cannot crowd out the
# "who can help" behaviour the original prompt asked for, and a people question cannot
# fill every slot with colleagues.
TOP_K = 6
TOP_K_MEMBERS = 2

# A retrieved person is only appended above this cosine. Filtering to the directory means
# the search always returns somebody, so without a floor a question about freezing medium
# gets two arbitrary postdocs attached and the model, told to name who can help, names
# them.
#
# Measured on the top members retrieved, before and after bios were chunked by sentence:
#
#                                                      whole bio   chunked
#   "who runs the flow cytometry platforms"  Shreyasee     0.566     0.662  right
#   "who should I ask about ordering reagents" Michelle    0.516     0.543  right
#   "  (the same query)"                       Aman        0.557     0.564  wrong
#   "what freezing medium is used for BJ cells" anyone     0.538     0.602  wrong
#
# Chunking lifted every member score, signal and noise alike, so the floor had to rise
# with it; at 0.55 it now admits everybody. At 0.62 it keeps Shreyasee, drops the
# freezing-medium case, and on the reagents question admits nobody rather than the wrong
# person, which is the better of the two outcomes available.
#
# It is still not a relevance test. On that reagents question the correct person scores
# below an incorrect one on cosine, so the floor cannot separate them; only the fused
# ranking gets Michelle to the top, and that is because BM25 matches "reagents" in her
# bio. This rejects the obviously unrelated and does no more.
MEMBER_FLOOR = DEFAULT_ABSTAIN_COSINE

INDEX_DIR = os.environ.get("LABGPT_INDEX_DIR", str(cfg.REPO_ROOT / ".index"))

ifSearchContent = (
    "Search is NEEDED (true) for: 1. Real-time information: Weather, stock prices, "
    "traffic, sports scores. 2. Recent events: Anything that happened after your "
    "knowledge cutoff. 3. Specific people or entities outside this laboratory. "
    "4. Niche or local information: Store hours, specific product recommendations, "
    "local regulations.\n"
    "Search is NOT needed for anything about this lab's protocols, reagents, safety "
    "guidance, members or publications, which are answered from local documents.\n"
    "Does this query need a web search (answer with one word, yes or no): "
)


def run_chat() -> None:
    try:
        store = VectorStore.load(INDEX_DIR)
    except Exception as exc:
        rprint(f"[red]{exc}[/red]")
        rprint("Build the index first: [bold]python -m labrag.cli index[/bold]")
        return

    rprint(f"[dim]index: {len(store.documents)} documents, {len(store.chunks)} chunks[/dim]")
    embedder = Embedder(store.manifest.get("embedding_model"))
    retriever = Retriever(store, embedder=embedder, mode="hybrid", final_k=TOP_K)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    rprint(f"[green]Welcome to chat with {cfg.ASSISTANT_NAME}![/green]")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal "
                  "encoding to utf-8.")
            continue
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip() in ("exit", "quit"):
            break
        if not query.strip():
            continue

        answer_query(query, retriever, tokenizer, model)


def answer_query(query, retriever, tokenizer, model) -> None:
    rprint(f"[green]{cfg.ASSISTANT_NAME}: [/green]")

    # --- retrieve -------------------------------------------------------------
    result = retriever.retrieve(query, k=TOP_K)
    # Filtered to the directory, so this always returns somebody: it hands back the two
    # least-bad people whether or not either is relevant. Asked about freezing medium it
    # will cheerfully produce two postdocs who have nothing to do with cryopreservation,
    # and the model, told to name who can help, will name them. Hence the floor below.
    members = retriever.retrieve(query, k=TOP_K_MEMBERS, doc_types=["member"],
                                 include_linked=False)

    abstain, reason = should_abstain(result, DEFAULT_ABSTAIN_COSINE)
    rprint(f"[dim]best match {result.best_cosine:.3f}, "
           f"{len(result.documents)} documents[/dim]")

    if abstain:
        # Refusing here costs nothing and is a correct answer for a safety corpus. The
        # model is never called, so it cannot improvise around the gap.
        rprint(f"[yellow]{ABSTENTION_TEMPLATE.format(reason=f'Reason: {reason}.')}[/yellow]")
        return

    # --- optional web search --------------------------------------------------
    sources = list(result.documents)
    web_summary = None
    if _needs_web_search(query, tokenizer, model):
        rprint("[blue]Searching the web ...[/blue]")
        web_summary = _summarise_web(query, tokenizer, model)

    # The directory is appended rather than competing for the main slots, so that the
    # "tell them who can help" behaviour survives without paying for the whole of it.
    # Only people who actually match are appended: a weak match is worse than none,
    # because the prompt asks the model to name someone and it will name whoever is here.
    #
    # The floor is on the cosine, not on Retrieved.score, which is the fused RRF value.
    # RRF encodes rank rather than similarity, so the top result of a search that matched
    # nothing still scores highly and a floor on it would never reject anybody.
    relevant = {
        hit.chunk.doc_id for hit in members.chunk_hits if hit.confidence >= MEMBER_FLOOR
    }
    seen = {item.document.doc_id for item in sources}
    for item in members.documents:
        if item.document.doc_id in relevant and item.document.doc_id not in seen:
            sources.append(item)

    for number, item in enumerate(sources, start=1):
        rprint(f"[dim]  [S{number}] {item.document.doc_type:<14} "
               f"{item.document.title[:62]}[/dim]")

    # --- generate -------------------------------------------------------------
    messages = build_messages(query, sources)
    if web_summary:
        messages[-1]["content"] += (
            f"\n\nAdditional context from a web search, which is NOT one of the "
            f"numbered sources and must not be cited as one:\n{web_summary}"
        )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    client = TransformersChatClient(
        tokenizer, model, max_new_tokens=MAX_NEW_TOKENS_ANSWER, streamer=streamer
    )

    with metrics.timer() as elapsed:
        text, usage = client.complete(messages)
    metrics.record(
        "answer",
        prompt_tokens=usage.get("prompt_tokens", 0),
        generated_tokens=usage.get("completion_tokens", 0),
        elapsed_s=elapsed[0],
        sources=len(sources),
        best_cosine=round(result.best_cosine, 4),
    )

    # --- validate -------------------------------------------------------------
    valid, invalid = validate_citations(text, len(sources))
    if not valid:
        # An answer with no usable provenance is indistinguishable from a fabrication
        # here, so it is withheld rather than served.
        rprint("\n[yellow]That draft carried no citation back to an indexed document, "
               "so it is being withheld.[/yellow]")
        rprint(f"[yellow]{ABSTENTION_TEMPLATE.format(reason='Reason: the generated answer cited no valid source.')}[/yellow]")
        return

    print()
    rprint("[dim]cited:[/dim]")
    for number in valid:
        document = sources[number - 1].document
        rprint(f"[dim]  [S{number}] {document.title[:64]} ({document.source})[/dim]")
    if invalid:
        rprint(f"[red]  invalid citations emitted: {invalid}[/red]")
    uncited = count_uncited_sentences(text)
    if uncited:
        rprint(f"[yellow]  {uncited} factual sentence(s) carried no citation[/yellow]")


# --- web search ---------------------------------------------------------------


def _needs_web_search(query, tokenizer, model) -> bool:
    """The one classifier call left. It gates a real external call, so it earns its cost.

    The other two are gone. Retrieval decides what the query is about by scoring the
    corpus, which is both cheaper and measurable, where the classifiers were neither.
    """
    answer = _short_answer(
        ifSearchContent + query, tokenizer, model,
        max_new_tokens=MAX_NEW_TOKENS_YESNO, label="classify:web",
    )
    return "yes" in answer.lower()


def _summarise_web(query, tokenizer, model):
    # Imported here rather than at module scope so the assistant starts without the
    # search extra installed. Web search is optional; selenium and a browser driver are
    # a heavy thing to require of a deployment that only wants to answer from the corpus.
    try:
        from search.duckducksearch import get_useful_link
        from search.getDynamicPage import fetch_all_webpage_content
    except ImportError as exc:
        rprint(f"[yellow]web search unavailable ({exc}); install with "
               f"uv sync --extra search[/yellow]")
        return None

    try:
        links, snippets = get_useful_link(query)
        pages = fetch_all_webpage_content(links)
    except Exception as exc:
        rprint(f"[red]web search failed: {exc}[/red]")
        return None
    if not pages:
        return None

    prompt = (
        "Summarise the following web results to answer this question. Be precise and "
        f"keep any useful link.\n\nQuestion: {query}\n"
    )
    for index, item in enumerate(pages):
        snippet = snippets[index] if index < len(snippets) else ""
        prompt += (f"\n--- result {index + 1} ---\nURL: {item['url']}\n"
                   f"Snippet: {snippet}\nContent: {item['content']}\n")

    return _short_answer(prompt, tokenizer, model, max_new_tokens=512,
                         label="web_summary", thinking=False)


def _short_answer(prompt, tokenizer, model, max_new_tokens, label, thinking=False) -> str:
    """One deterministic generation, instrumented. Sampling is off so routing is
    reproducible; otherwise the same question can take different paths on different runs.
    """
    client = TransformersChatClient(
        tokenizer, model, max_new_tokens=max_new_tokens, thinking=thinking
    )
    try:
        with metrics.timer() as elapsed:
            text, usage = client.complete([{"role": "user", "content": prompt}])
    except Exception as exc:
        print(f"\nError during model generation: {exc}")
        return ""
    metrics.record(
        label,
        prompt_tokens=usage.get("prompt_tokens", 0),
        generated_tokens=usage.get("completion_tokens", 0),
        elapsed_s=elapsed[0],
        cap=max_new_tokens,
    )
    return text


if __name__ == "__main__":
    run_chat()
