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

    1 classifier call on the local path, only to decide whether to search the web, which
      is a real external call worth gating; none at all on the endpoint path
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

WHERE GENERATION HAPPENS. Retrieval is always local. The answer comes from a hosted
endpoint by default, the institutional Azure OpenAI gateway in practice, or from a local
transformers model when LABGPT_BACKEND=local asks for one. See BACKEND below.

The web search leg is part of the local path only. On the endpoint path it is skipped
entirely, along with the classifier call that gates it, so an api run answers from the
retrieved corpus and nothing else.
"""

import os
from rich import print as rprint
import labgpt_config as cfg
import labgpt_metrics as metrics

from labrag.answerer import (
    DEFAULT_ABSTAIN_COSINE, TransformersChatClient, chat_client_from_env,
    should_abstain, validate_citations, count_uncited_sentences,
)
from labrag.embeddings import Embedder, resolve_model
from labrag.rerank import DEFAULT_RERANK_MODEL, Reranker, rerank_is_cheap
from labrag.prompts import (
    ABSTENTION_TEMPLATE, build_messages, build_messages_without_sources,
)
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

# Which model answers: "api" calls a hosted endpoint, "local" loads cfg.MODEL_NAME with
# transformers. The endpoint is the default, and the local model is opt-in.
#
# The default is the endpoint because it is what this lab actually answers on: the local
# 32B model needs a GPU that is not always at hand, where the approved Azure deployment is
# reachable from any machine on the network. Nothing about that choice is automatic, which
# is the point of naming it here rather than sniffing the environment: an unset variable
# should not silently move generation from one to the other.
#
# The local path is still the only configuration in which nothing leaves the machine, and
# it is one variable away. The api path needs an endpoint approved for this data, which the
# institutional Azure OpenAI gateway is and a personal OpenAI key is not;
# chat_client_from_env prefers Azure whenever its variables are set, so on a machine
# configured for both the approved endpoint wins without anyone remembering a flag.
BACKEND = os.environ.get("LABGPT_BACKEND", "api")

# What happens when retrieval scores below the abstention threshold. "answer" puts the
# question to the model with no sources attached and labels the result as general
# knowledge; "refuse" stops at the gate, which is what this pipeline did before.
#
# The refusal exists because an unsupported answer about a centrifuge speed or a spill is
# worse than no answer: it reads as verified. Answering anyway gives that back some of the
# ground it was standing on, so the fallback is built to keep the boundary visible rather
# than to blur it. No sources are supplied, so no [S#] can be emitted and citation
# validation is skipped; the prompt is a different one that opens by saying the lab's
# documentation does not cover this, refuses to state lab specifics it does not have, and
# sends anything urgent to a person. The transcript marks the whole answer as unsourced.
#
# The second gate is untouched. An answer that was given sources and cited none of them is
# still withheld: that is a model ignoring its evidence, not a gap in the corpus.
ABSTAIN_MODE = os.environ.get("LABGPT_ABSTAIN_MODE", "answer")

# Whether a cross-encoder re-sorts the fused shortlist before the top k is taken. "auto"
# defers to the hardware, which is what the CLI does and for the same measured reason: the
# pass buys roughly three end-to-end questions in a hundred, and costs 13.1 seconds a query
# on a CPU against 6.2 for generation. Free on a GPU, and the dominant cost of a question
# without one.
#
# demo.py ran without a reranker at all while the CLI reranked by default, so the same
# question could be answered from a differently ordered shortlist depending on which one
# asked it. That was an oversight, not a decision.
#
# It loads a second model, BAAI/bge-reranker-base, on first use. On a network that blocks
# huggingface.co that download fails the way the embedding model does, so "off" is the
# escape hatch and says so when it fires.
RERANK_MODE = os.environ.get("LABGPT_RERANK", "auto")
if RERANK_MODE not in ("auto", "on", "off"):
    raise SystemExit(f"LABGPT_RERANK must be auto, on or off, not {RERANK_MODE!r}")

# How many fused candidates the reranker scores. The CLI's default, kept in step.
RERANK_CANDIDATES = 40


def build_reranker():
    """The reranker RERANK_MODE asks for, or none."""
    wanted = RERANK_MODE == "on" or (RERANK_MODE == "auto" and rerank_is_cheap())
    if not wanted:
        return None
    reranker = Reranker(os.environ.get("LABGPT_RERANK_MODEL") or DEFAULT_RERANK_MODEL)
    # Scored once here to force the load. The Reranker loads on first use, which would
    # otherwise be inside the first question, where a hub that cannot be reached takes the
    # chat loop down with it instead of costing one degraded startup.
    reranker.score("warm up", ["warm up"])
    return reranker
if ABSTAIN_MODE not in ("answer", "refuse"):
    raise SystemExit(f"LABGPT_ABSTAIN_MODE must be answer or refuse, not {ABSTAIN_MODE!r}")


class LocalBackend:
    """cfg.MODEL_NAME, loaded once and reused for every call in the session."""

    streams = True
    web_search = True
    answer_tokens = MAX_NEW_TOKENS_ANSWER

    def __init__(self):
        # Imported here, not at module scope, so the api path runs on a machine with no
        # transformers and no torch installed at all.
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        self._TextStreamer = TextStreamer
        rprint(f"[dim]loading {model_name} ...[/dim]")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def describe(self) -> str:
        return f"local {model_name}"

    def client(self, max_tokens: int, stream: bool = False, thinking: bool = False):
        streamer = None
        if stream:
            streamer = self._TextStreamer(self.tokenizer, skip_prompt=True,
                                          skip_special_tokens=True)
        return TransformersChatClient(self.tokenizer, self.model,
                                      max_new_tokens=max_tokens, thinking=thinking,
                                      streamer=streamer)


class ApiBackend:
    """A hosted OpenAI-compatible endpoint, Azure OpenAI included.

    A fresh client per call, because the token budget differs per call and the clients are
    cheap to construct; each one keeps only its configuration.

    The budget has a floor here that the local path does not need. A reasoning model
    spends its reasoning tokens out of the same allowance and spends them first, so the
    8-token budget that is ample for a local model writing "yes" returns an empty string
    from gpt-5 with nothing left to answer with.
    """

    streams = False
    # No web leg on this path. The web branch exists to reach what the corpus does not
    # cover, and it costs a classifier call on every single question plus, when it fires,
    # a browser driver and several page fetches. On the endpoint path that trade is not
    # taken: the endpoint is asked about the retrieved corpus and nothing else, which also
    # means a question routed to the web cannot quietly send its text out through a second
    # channel that was never part of the approved one.
    web_search = False
    MIN_TOKENS = 64

    # The answer budget is larger here than on the local path, and this is the reason.
    # A reasoning deployment spends its reasoning tokens out of the same allowance and
    # spends them first, so the budget is not an answer length: at the endpoint's own
    # default effort, gpt-5 spent all 1024 tokens of the local path's budget on reasoning
    # and returned an empty answer.
    #
    # 2500 is the CLI's measured figure rather than a guess: over 101 real questions at
    # low effort, reasoning took a median of 192 tokens, 448 at the 90th percentile and
    # 704 at the worst successful call. Kept in step with cli.DEFAULT_MAX_TOKENS, and
    # overridden by the same LABGPT_MAX_TOKENS.
    #
    # The effort default is the other half. Answers on this workload are quoted from the
    # supplied sources rather than derived, so reasoning buys little: "minimal", "low" and
    # "medium" were measured returning the same answer with zero reasoning tokens spent,
    # where the endpoint default spends over a thousand. Low unless LABGPT_REASONING_EFFORT
    # says otherwise.
    DEFAULT_EFFORT = "low"

    def __init__(self):
        self.reasoning_effort = (os.environ.get("LABGPT_REASONING_EFFORT")
                                 or self.DEFAULT_EFFORT)
        self.answer_tokens = int(os.environ.get("LABGPT_MAX_TOKENS") or 2500)
        if not _endpoint_configured():
            # Reported here rather than as a 401 from api.openai.com on the first question,
            # which is what an unconfigured OpenAI-shaped client would produce.
            raise SystemExit(
                "no chat endpoint is configured, and the api backend is the default.\n"
                "  For the institutional Azure gateway, set AZURE_OPENAI_GATEWAY, "
                "AZURE_OPENAI_TEAM_ID,\n"
                "  AZURE_OPENAI_MODEL_ID, AZURE_OPENAI_API_VERSION and "
                "APIM_OPENAI_SUBSCRIPTION_KEY;\n"
                "  check them with python -m labrag.cli selftest.\n"
                "  For another endpoint, set LABGPT_LLM_BASE_URL and LABGPT_LLM_API_KEY.\n"
                f"  To answer on the local model instead: "
                f"LABGPT_BACKEND=local python demo.py"
            )
        # Constructed once here, purely so that a misconfigured endpoint fails now rather
        # than after the index has loaded and the first question has been typed.
        probe = chat_client_from_env(max_tokens=MAX_NEW_TOKENS_ANSWER)
        self._kind = type(probe).__name__
        self._model = probe.model

    def describe(self) -> str:
        return (f"{self._kind} {self._model}, {self.answer_tokens} tokens, "
                f"reasoning_effort={self.reasoning_effort}")

    def client(self, max_tokens: int, stream: bool = False, thinking: bool = False):
        # stream and thinking are accepted to match LocalBackend and ignored: an endpoint
        # is asked for a complete response, and a hosted model's reasoning is controlled
        # by reasoning_effort rather than by a chat-template flag.
        return chat_client_from_env(max_tokens=max(max_tokens, self.MIN_TOKENS),
                                    reasoning_effort=self.reasoning_effort)


def _endpoint_configured() -> bool:
    """Whether anything at all names an endpoint to call."""
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT")
                or (os.environ.get("AZURE_OPENAI_GATEWAY")
                    and os.environ.get("AZURE_OPENAI_TEAM_ID"))
                or os.environ.get("LABGPT_LLM_BASE_URL")
                or os.environ.get("LABGPT_LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY"))


def build_backend():
    """The backend BACKEND asks for. The endpoint unless the local model was asked for."""
    if BACKEND == "local":
        return LocalBackend()
    if BACKEND != "api":
        raise SystemExit(f"LABGPT_BACKEND must be api or local, not {BACKEND!r}")
    return ApiBackend()

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
    embedder = Embedder(resolve_model(store.manifest.get("embedding_model")))
    try:
        reranker = build_reranker()
    except Exception as exc:
        rprint(f"[yellow]reranker unavailable ({exc}); continuing without it. "
               f"Set LABGPT_RERANK=off to stop trying.[/yellow]")
        reranker = None
    retriever = Retriever(store, embedder=embedder, mode="hybrid", final_k=TOP_K,
                          reranker=reranker, rerank_candidates=RERANK_CANDIDATES)
    rprint(f"[dim]rerank: {reranker.model_name if reranker else 'off'}[/dim]")

    try:
        backend = build_backend()
    except Exception as exc:
        rprint(f"[red]{exc}[/red]")
        return
    rprint(f"[dim]model: {backend.describe()}[/dim]")

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

        answer_query(query, retriever, backend)


def answer_query(query, retriever, backend) -> None:
    rprint(f"[green]{cfg.ASSISTANT_NAME}: [/green]")

    # --- retrieve -------------------------------------------------------------
    result = retriever.retrieve(query, k=TOP_K)
    # Filtered to the directory, so this always returns somebody: it hands back the two
    # least-bad people whether or not either is relevant. Asked about freezing medium it
    # will cheerfully produce two postdocs who have nothing to do with cryopreservation,
    # and the model, told to name who can help, will name them. Hence the floor below.
    members = retriever.retrieve(query, k=TOP_K_MEMBERS, doc_types=["member"],
                                 include_linked=False)
    # The member floor below reads hit.confidence, which is a cosine and is left untouched
    # by reranking; the reranker's logit is not on that scale and is not a substitute.

    abstain, reason = should_abstain(result, DEFAULT_ABSTAIN_COSINE)
    rprint(f"[dim]best match {result.best_cosine:.3f}, "
           f"{len(result.documents)} documents[/dim]")

    if abstain:
        if ABSTAIN_MODE == "refuse":
            # The model is never called, so it cannot improvise around the gap.
            rprint(f"[yellow]{ABSTENTION_TEMPLATE.format(reason=f'Reason: {reason}.')}[/yellow]")
            return
        _answer_without_sources(query, backend, reason)
        return

    # --- optional web search --------------------------------------------------
    sources = list(result.documents)
    web_summary = None
    if backend.web_search and _needs_web_search(query, backend):
        rprint("[blue]Searching the web ...[/blue]")
        web_summary = _summarise_web(query, backend)

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

    # The local backend streams as it generates, so the answer is already on screen by the
    # time complete() returns. An endpoint returns it whole, and it is printed below.
    client = backend.client(backend.answer_tokens, stream=True)

    try:
        with metrics.timer() as elapsed:
            text, usage = client.complete(messages)
    except Exception as exc:
        rprint(f"[red]generation failed: {exc}[/red]")
        return
    if not backend.streams:
        print(text)
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


def _answer_without_sources(query, backend, reason: str) -> None:
    """Answer with nothing retrieved, labelled as general knowledge.

    Reached only when the retrieval gate failed, and only when LABGPT_ABSTAIN_MODE leaves
    it enabled. Nothing here is checked against the corpus, because there is no corpus
    content in the prompt; the banner and the prompt's own first rule are what keep the
    answer from being read as the lab's documented practice.
    """
    rprint(f"[yellow]Nothing in the indexed lab documentation matched this "
           f"({reason}).[/yellow]")
    rprint("[yellow]Answering from general knowledge instead. This is NOT this lab's "
           "documented practice, and it carries no citations.[/yellow]\n")

    client = backend.client(backend.answer_tokens, stream=True)
    try:
        with metrics.timer() as elapsed:
            text, usage = client.complete(build_messages_without_sources(query))
    except Exception as exc:
        rprint(f"[red]generation failed: {exc}[/red]")
        return
    if not backend.streams:
        print(text)
    metrics.record(
        "answer:unsourced",
        prompt_tokens=usage.get("prompt_tokens", 0),
        generated_tokens=usage.get("completion_tokens", 0),
        elapsed_s=elapsed[0],
        sources=0,
        best_cosine=0.0,
    )

    # Citations are not validated, because none were possible. A model that emitted one
    # anyway invented it, and saying so is the whole point of checking.
    _, invented = validate_citations(text, 0)
    if invented:
        rprint(f"\n[red]the answer cited {invented}, but no sources were supplied; "
               f"those citations are invented[/red]")
    rprint("\n[yellow]Unsourced answer. Confirm anything lab-specific with the protocol "
           "owner or the safety officer.[/yellow]")


# --- web search ---------------------------------------------------------------


def _needs_web_search(query, backend) -> bool:
    """The one classifier call left. It gates a real external call, so it earns its cost.

    The other two are gone. Retrieval decides what the query is about by scoring the
    corpus, which is both cheaper and measurable, where the classifiers were neither.

    Only reached on a backend whose web_search is set, which today means the local one.
    """
    answer = _short_answer(
        ifSearchContent + query, backend,
        max_new_tokens=MAX_NEW_TOKENS_YESNO, label="classify:web",
    )
    return "yes" in answer.lower()


def _summarise_web(query, backend):
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

    return _short_answer(prompt, backend, max_new_tokens=512,
                         label="web_summary", thinking=False)


def _short_answer(prompt, backend, max_new_tokens, label, thinking=False) -> str:
    """One generation, instrumented.

    Deterministic on the local backend, where sampling is off, so routing is reproducible.
    Not on a hosted one: a reasoning model refuses any temperature but its default, so the
    same question can take different paths on different runs and a comparison across runs
    has to treat the routing as noisy.
    """
    client = backend.client(max_new_tokens, thinking=thinking)
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
