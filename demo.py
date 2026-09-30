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

WHERE THE LOGIC LIVES. This file is the terminal front end and nothing else. Retrieval,
the gates, the backends and the web leg moved to labrag/chat.py when an HTTP service
needed the same behaviour, so the chat loop and the service cannot drift apart on when to
abstain or what counts as a valid citation. Everything below renders what
answer_structured returned.
"""

import os
from rich import print as rprint
import labgpt_config as cfg

from labrag.chat import (
    INDEX_DIR, RERANK_CANDIDATES, TOP_K, answer_structured, build_backend,
    build_reranker,
)
from labrag.embeddings import Embedder, resolve_model
from labrag.retriever import Retriever
from labrag.store import VectorStore


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

        rprint(f"[green]{cfg.ASSISTANT_NAME}: [/green]")
        try:
            answer = answer_structured(query, retriever, backend)
        except Exception as exc:
            rprint(f"[red]generation failed: {exc}[/red]")
            continue
        _render(answer, streamed=backend.streams)


def _render(answer: dict, streamed: bool) -> None:
    """Print what answer_structured decided.

    The local backend streams as it generates, so its text is already on screen by the
    time the dict comes back; the endpoint returns it whole and it is printed here.
    """
    mode = answer["mode"]
    rprint(f"[dim]best match {answer['best_cosine']:.3f}, "
           f"{len(answer['sources'])} documents[/dim]")

    if mode == "refused":
        rprint(f"[yellow]{answer['answer']}[/yellow]")
        return

    if mode == "unsourced":
        rprint(f"[yellow]Nothing in the indexed lab documentation matched this "
               f"({answer['reason']}).[/yellow]")
        rprint("[yellow]Answering from general knowledge instead. This is NOT this "
               "lab's documented practice, and it carries no citations.[/yellow]\n")
        if not streamed:
            print(answer["answer"])
        if answer["invalid_citations"]:
            rprint(f"\n[red]the answer cited {answer['invalid_citations']}, but no "
                   f"sources were supplied; those citations are invented[/red]")
        rprint("\n[yellow]Unsourced answer. Confirm anything lab-specific with the "
               "protocol owner or the safety officer.[/yellow]")
        return

    for item in answer["sources"]:
        rprint(f"[dim]  [S{item['n']}] {item['doc_type']:<14} "
               f"{item['title'][:62]}[/dim]")

    if mode == "withheld":
        rprint("\n[yellow]That draft carried no citation back to an indexed document, "
               "so it is being withheld.[/yellow]")
        rprint(f"[yellow]{answer['answer']}[/yellow]")
        return

    if not streamed:
        print(answer["answer"])
    print()
    rprint("[dim]cited:[/dim]")
    for number in answer["cited"]:
        item = answer["sources"][number - 1]
        rprint(f"[dim]  [S{number}] {item['title'][:64]} ({item['source']})[/dim]")
    if answer["invalid_citations"]:
        rprint(f"[red]  invalid citations emitted: {answer['invalid_citations']}[/red]")
    if answer["uncited_sentences"]:
        rprint(f"[yellow]  {answer['uncited_sentences']} factual sentence(s) carried "
               f"no citation[/yellow]")


if __name__ == "__main__":
    run_chat()
