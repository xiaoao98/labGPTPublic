"""Command line for the retrieval layer.

    python -m labrag.cli chunks     ingest only, write chunks.jsonl, print stats
                                    (no model needed, no numpy needed)
    python -m labrag.cli index      ingest, embed, write the full index
    python -m labrag.cli info       describe an existing index
    python -m labrag.cli search Q   retrieve against the index, no LLM involved

`chunks` exists so the chunking can be inspected and iterated on without paying for
embedding, which is the slow part. Get the chunk boundaries right first, then index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import labgpt_config as cfg  # noqa: E402

from .answerer import (  # noqa: E402
    Answerer, ChatClient, DEFAULT_ABSTAIN_COSINE, LLMError, should_abstain,
)
from .chunker import write_chunks  # noqa: E402
from .documents import write_documents  # noqa: E402
from .embeddings import DEFAULT_MODEL, Embedder  # noqa: E402
from .evaluate import (  # noqa: E402
    SERVED_K,
    abstention_separation, aggregate, by_category, check_labels, evaluate, load_questions,
)
from .ingest import ingest_all  # noqa: E402
from .prompts import ABSTENTION_TEMPLATE, build_messages  # noqa: E402
from .rerank import DEFAULT_RERANK_MODEL, Reranker  # noqa: E402
from .retriever import Retriever  # noqa: E402
from .store import VectorStore  # noqa: E402

DEFAULT_INDEX_DIR = Path(cfg.REPO_ROOT) / ".index"


def _report(chunks) -> None:
    sizes = sorted(c.tokens for c in chunks)
    total = sum(sizes)
    count = len(sizes)
    print(f"\n{count} chunks, {total:,} tokens estimated")
    if count:
        print(
            f"  size: min {sizes[0]}  median {sizes[count // 2]}  "
            f"p90 {sizes[int(count * 0.9)]}  max {sizes[-1]}"
        )
    oversized = [c for c in chunks if c.tokens > 500]
    if oversized:
        print(f"  {len(oversized)} chunks over 500 tokens:")
        for chunk in oversized[:10]:
            print(f"    {chunk.tokens:>5}  {chunk.chunk_id}")


def cmd_chunks(args) -> int:
    print(f"corpus: {cfg.DATA_DIR}")
    documents, chunks = ingest_all()
    _report(chunks)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_chunks(out, chunks)
        write_documents(out.parent / "documents.jsonl", documents)
        print(f"\nwrote {out} and {out.parent / 'documents.jsonl'}")
    return 0


def cmd_index(args) -> int:
    print(f"corpus: {cfg.DATA_DIR}")
    documents, chunks = ingest_all()
    _report(chunks)

    embedder = Embedder(args.model)
    print(f"\nembedding {len(chunks)} chunks ...")
    vectors = embedder.encode_documents([c.embed_text for c in chunks])

    store = VectorStore.build(args.index_dir, chunks, vectors, args.model, documents)
    print(f"\nwrote index to {args.index_dir}")
    print(store.describe())
    return 0


def cmd_info(args) -> int:
    store = VectorStore.load(args.index_dir)
    print(store.describe())
    return 0


def cmd_search(args) -> int:
    store = VectorStore.load(args.index_dir)
    embedder = None
    if args.mode in ("dense", "hybrid"):
        embedder = Embedder(store.manifest.get("embedding_model", DEFAULT_MODEL))

    retriever = Retriever(
        store,
        embedder=embedder,
        mode=args.mode,
        final_k=args.k,
        rrf_k=args.rrf_k,
        weight_dense=args.weight_dense,
        weight_lexical=args.weight_lexical,
    )
    result = retriever.retrieve(
        args.query, doc_types=args.doc_type or None, include_linked=not args.no_linked
    )

    print(f"\nquery: {args.query!r}   mode: {result.mode}")

    if args.explain:
        print(f"\nchunk hits before expansion (best cosine {result.best_cosine:.3f}):")
        header = f"  {'rrf':>7}{'dense':>8}{'cos':>7}{'bm25':>7}{'score':>8}  found_by  chunk"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for hit in result.chunk_hits:
            dense_rank = hit.dense_rank if hit.dense_rank is not None else "-"
            cosine = f"{hit.dense_score:.3f}" if hit.dense_score is not None else "-"
            lex_rank = hit.lexical_rank if hit.lexical_rank is not None else "-"
            lex_score = f"{hit.lexical_score:.2f}" if hit.lexical_score is not None else "-"
            print(
                f"  {hit.fused_score:>7.4f}{str(dense_rank):>8}{cosine:>7}"
                f"{str(lex_rank):>7}{lex_score:>8}  {hit.found_by:<9} {hit.chunk.chunk_id}"
            )

    print(f"\ndocuments returned ({len(result.documents)}):")
    for position, retrieved in enumerate(result.documents, start=1):
        doc = retrieved.document
        tag = f"   [linked from {retrieved.via_link_from}]" if retrieved.is_linked else ""
        print(f"\n{position}. {doc.title}  ({doc.doc_type})   score {retrieved.score:.4f}{tag}")
        print(f"   source: {doc.source}   {len(doc.text)} chars")
        if retrieved.matched_chunk_ids:
            print(f"   matched: {', '.join(retrieved.matched_chunk_ids)}")
        body = doc.text if args.full else doc.text[:280].replace("\n", " ")
        print(f"   {body}{'' if args.full or len(doc.text) <= 280 else ' ...'}")
    return 0


def cmd_ask(args) -> int:
    store = VectorStore.load(args.index_dir)
    embedder = Embedder(store.manifest.get("embedding_model", DEFAULT_MODEL))
    retriever = Retriever(store, embedder=embedder, mode="hybrid", final_k=args.k)

    result = retriever.retrieve(args.question, k=args.k)
    abstain, reason = should_abstain(result, args.threshold)

    print(f"\nquestion: {args.question!r}")
    print(f"best cosine {result.best_cosine:.3f}  threshold {args.threshold:.2f}  "
          f"-> {'ABSTAIN' if abstain else 'answer'}")

    if abstain:
        print()
        print(ABSTENTION_TEMPLATE.format(reason=f"Reason: {reason}."))
        return 0

    print(f"\n{len(result.documents)} sources retrieved:")
    for number, item in enumerate(result.documents, start=1):
        tag = "  [linked]" if item.is_linked else ""
        print(f"  [S{number}] {item.document.doc_type:<14} {item.document.title[:64]}{tag}")

    if args.dry_run:
        messages = build_messages(args.question, result.documents)
        prompt_chars = sum(len(m["content"]) for m in messages)
        print(f"\n--- assembled prompt, {prompt_chars:,} chars, not sent ---")
        print(messages[-1]["content"][: args.show])
        if len(messages[-1]["content"]) > args.show:
            print(f"\n... {len(messages[-1]['content']) - args.show:,} more chars")
        return 0

    answerer = Answerer(
        retriever,
        client=_client(args),
        k=args.k,
        abstain_cosine=args.threshold,
    )
    answer = answerer.answer(args.question)
    print(f"\n{answer.text}\n")
    if answer.citations:
        print("citations:")
        for citation in answer.citations:
            print(f"  {citation}")
    if answer.invalid_citations:
        print(f"INVALID citations emitted: {answer.invalid_citations}")
    if answer.uncited_sentences:
        print(f"uncited factual sentences: {answer.uncited_sentences}")
    return 0


def _client(args):
    """The chat client these arguments ask for."""
    return ChatClient(model=args.model, base_url=getattr(args, "base_url", None),
                      max_tokens=args.max_tokens,
                      reasoning_effort=args.reasoning_effort)


def _oracle_documents(retriever, store, question, k):
    """The k slots the model is given, with the labelled documents put into them.

    Gold first, then the highest ranked documents retrieval actually found, until the
    slots are full. Holding the slot count at k is the point: production gives the model
    k documents, so an oracle that handed it twenty would be measuring context length as
    well as retrieval, and the two could not be told apart afterwards.

    Four questions carry more gold than fits. Their gold is redundant rather than
    enumerative, nineteen documents naming the same Cre drivers and seven naming the same
    implantation site, so six of them is a complete answer and the cut changes nothing the
    model could say.
    """
    from .documents import Retrieved

    picked, seen = [], set()
    for doc_id in question.relevant[:k]:
        document = store.documents.get(doc_id)
        if document is None:
            continue
        picked.append(Retrieved(document=document, score=1.0))
        seen.add(doc_id)
    if len(picked) < k:
        for item in retriever.retrieve(question.question, k=k).documents:
            if len(picked) >= k:
                break
            if item.document.doc_id not in seen:
                picked.append(item)
                seen.add(item.document.doc_id)
    return picked


def cmd_selftest(args) -> int:
    """One trivial prompt, to prove the endpoint works before a batch is committed to it.

    Worth its own command because the failures are all configuration and they are much
    cheaper to read here than on question 87 of a run that has been going for an hour.
    """
    client = _client(args)
    print(f"endpoint    {client.url}  model={client.model}")
    print(f"key         {'set, ' + str(len(client._key)) + ' chars' if client._key else 'MISSING'}")
    print(f"shape       {'reasoning (max_completion_tokens, no temperature)' if client._reasoning else 'classic (max_tokens, temperature)'}")
    print("\nsending one prompt ...")
    started = time.perf_counter()
    try:
        text, usage = client.complete(
            [{"role": "user", "content": "Reply with exactly: OK"}])
    except LLMError as exc:
        print(f"\nFAILED: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    details = usage.get("completion_tokens_details", {})
    print(f"reply       {text!r}")
    print(f"elapsed     {elapsed:.1f}s")
    print(f"tokens      prompt {usage.get('prompt_tokens')}  "
          f"completion {usage.get('completion_tokens')}  "
          f"reasoning {details.get('reasoning_tokens', 0)}")
    if client._reasoning != type(client)._looks_like_reasoning_model(client.model):
        print("\nnote: the request shape was corrected on the first call; the deployment "
              "name did not predict it. Nothing to do, the client remembers.")
    print("\nOK")
    return 0


def cmd_answer_all(args) -> int:
    """Answer every question in a set, recording what each one cost.

    Writes JSONL, one record per line, flushed as it goes, so an interrupted run keeps
    everything it had finished. Re-run with --resume pointed at the same file to continue.
    """
    store = VectorStore.load(args.index_dir)
    questions = load_questions(args.questions)
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in questions if q.id in wanted]
    done: set[str] = set()
    if args.resume and args.out.exists():
        # A record carrying an error is not a finished question. Resuming used to skip it
        # anyway, which meant a run that lost seventeen questions to an exhausted token
        # budget could never be completed by resuming it, only by starting over. The kept
        # records are rewritten so a retry replaces its failure rather than appending a
        # second record under the same id.
        kept = []
        failed = 0
        for line in args.out.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("error"):
                failed += 1
            else:
                kept.append(record)
                done.add(record.get("id"))
        with args.out.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        questions = [q for q in questions if q.id not in done]
        print(f"resuming: {len(done)} answered, {failed} to retry, "
              f"{len(questions)} to do")
    if args.oracle:
        dropped = [q for q in questions if q.should_abstain]
        questions = [q for q in questions if not q.should_abstain]
        if dropped:
            print(f"oracle: skipping {len(dropped)} unanswerable questions, "
                  f"which have no gold documents to supply")
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("nothing to do")
        return 0

    embedder = Embedder(store.manifest.get("embedding_model", DEFAULT_MODEL))
    retriever = Retriever(store, embedder=embedder, mode="hybrid", final_k=args.k)
    answerer = Answerer(retriever, client=_client(args), k=args.k,
                        abstain_cosine=args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    handle = args.out.open("a" if args.resume else "w", encoding="utf-8")
    started = time.perf_counter()
    errors = 0
    try:
        for index, question in enumerate(questions, start=1):
            elapsed = time.perf_counter() - started
            eta = (elapsed / max(index - 1, 1)) * (len(questions) - index + 1) / 60
            print(f"[{index}/{len(questions)}] {question.id}  "
                  f"{question.question[:52]}   (eta {eta:.0f} min)")
            record = {"id": question.id, "category": question.category,
                      "should_abstain": question.should_abstain,
                      "question": question.question,
                      "system": "oracle-context" if args.oracle else "new-rag"}
            one = time.perf_counter()
            try:
                oracle = (_oracle_documents(retriever, store, question, args.k)
                          if args.oracle else None)
                answer = answerer.answer(question.question, documents=oracle)
                record.update(answer.to_dict())
                # Flattened out of usage so the cost of a run can be read without
                # digging: reasoning tokens are billed as completion tokens but are
                # not part of the answer, so the two are worth seeing side by side.
                usage = answer.usage or {}
                details = usage.get("completion_tokens_details", {})
                record.update({
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": details.get("reasoning_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                })
            except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
                errors += 1
                record.update({"answer": "", "error": repr(exc)})
                print(f"    error: {exc}")
            # Wall clock for the whole question. latency_ms from to_dict covers only the
            # model call, so the difference is retrieval and gating.
            record["elapsed_s"] = round(time.perf_counter() - one, 2)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    finally:
        handle.close()

    total = time.perf_counter() - started
    print(f"\nwrote {args.out}")
    print(f"  questions   {len(questions)}")
    print(f"  errors      {errors}")
    print(f"  wall clock  {total / 60:.1f} min  ({total / max(len(questions), 1):.1f}s each)")
    return 0


def _fmt(metrics: dict) -> str:
    return (
        f"{metrics['n']:>4}{metrics['success@1']:>9.3f}{metrics['success@6']:>9.3f}"
        f"{metrics['recall@6']:>9.3f}{metrics['recall@10']:>10.3f}"
        f"{metrics['mrr']:>8.3f}{metrics['ndcg@10']:>9.3f}"
    )


# S@6 and R@6 rather than @5, because six documents are what reach the model and five is
# a number no part of the system uses. The gap is not cosmetic: on the same run success@5
# reads 0.926 where success@6 reads 0.951.
HEADER = f"{'':<22}{'n':>4}{'S@1':>9}{'S@6':>9}{'R@6':>9}{'R@10':>10}{'MRR':>8}{'nDCG10':>9}"


def _build(store, mode, args, weight_lexical=1.0):
    embedder = None
    if mode in ("dense", "hybrid"):
        embedder = Embedder(store.manifest.get("embedding_model", DEFAULT_MODEL))
    return Retriever(store, embedder=embedder, mode=mode, final_k=10,
                     weight_lexical=weight_lexical,
                     top_k_dense=getattr(args, "top_k_dense", 20),
                     top_k_lexical=getattr(args, "top_k_lexical", 20),
                     reranker=_reranker(args),
                     rerank_candidates=getattr(args, "rerank_candidates", 40))


def _reranker(args):
    """One reranker for the whole run, so the model is loaded once rather than per query."""
    if not getattr(args, "rerank", False):
        return None
    if not hasattr(args, "_reranker_cache"):
        args._reranker_cache = Reranker(getattr(args, "rerank_model", None)
                                        or DEFAULT_RERANK_MODEL)
    return args._reranker_cache


def cmd_eval(args) -> int:
    store = VectorStore.load(args.index_dir)
    questions = load_questions(args.questions)
    errors, _ = check_labels(questions, store.documents)
    if errors:
        print(f"{len(errors)} label errors; fix these before trusting any number:")
        for error in errors:
            print("  ", error)
        return 1
    print(f"{len(questions)} questions, {len(store.documents)} documents, "
          f"{len(store.chunks)} chunks\n")

    results = evaluate(_build(store, args.mode, args), questions)
    print(HEADER)
    print("-" * len(HEADER))
    print(f"{'OVERALL':<22}{_fmt(aggregate(results))}")
    print()
    for name, metrics in by_category(results).items():
        print(f"{name:<22}{_fmt(metrics)}")

    separation = abstention_separation(results)
    if separation:
        print("\nabstention signal (best cosine per question):")
        print(f"  answerable    median {separation['answerable_median']:.3f}  "
              f"min {separation['answerable_min']:.3f}")
        print(f"  unanswerable  median {separation['unanswerable_median']:.3f}  "
              f"max {separation['unanswerable_max']:.3f}")
        print(f"  distributions overlap: {separation['overlap']}")
        print(f"  best single threshold {separation['best_threshold']:.3f} "
              f"-> {separation['best_accuracy']:.1%} correct")

    if args.failures:
        print("\nquestions where nothing relevant was retrieved at all:")
        for result in results:
            if result.question.answerable and result.recall_at_10 == 0.0:
                print(f"  {result.question.id} [{result.question.category}] "
                      f"{result.question.question}")
                print(f"      wanted {result.question.relevant}")
                print(f"      got    {result.retrieved[:3]}")
    return 0


def cmd_sweep(args) -> int:
    store = VectorStore.load(args.index_dir)
    questions = load_questions(args.questions)
    errors, _ = check_labels(questions, store.documents)
    if errors:
        print(f"{len(errors)} label errors; fix these first")
        for error in errors:
            print("  ", error)
        return 1

    configs = [
        ("dense", "dense", 1.0),
        ("bm25", "bm25", 1.0),
        ("hybrid", "hybrid", 1.0),
        ("hybrid w_lex=0.5", "hybrid", 0.5),
        ("hybrid w_lex=0.25", "hybrid", 0.25),
    ]
    print(f"{len(questions)} questions, {len(store.documents)} documents\n")
    print(HEADER)
    print("-" * len(HEADER))

    per_config = {}
    for label, mode, weight in configs:
        results = evaluate(_build(store, mode, args, weight_lexical=weight), questions)
        per_config[label] = results
        print(f"{label:<22}{_fmt(aggregate(results))}")

    print(f"\nsuccess@{SERVED_K} by category "
          f"(any relevant document in the {SERVED_K} served to the model)")
    categories = sorted({r.question.category for r in next(iter(per_config.values()))
                         if r.question.answerable})
    head = f"{'':<22}" + "".join(f"{c[:13]:>15}" for c in categories)
    print(head)
    print("-" * len(head))
    for label, results in per_config.items():
        cells = by_category(results)
        row = "".join(
            f"{cells[c]['success@6']:>15.3f}" if c in cells else f"{'-':>15}"
            for c in categories
        )
        print(f"{label:<22}{row}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="labrag", description=__doc__)
    parser.add_argument(
        "--index-dir", default=DEFAULT_INDEX_DIR, type=Path,
        help=f"where the index lives (default: {DEFAULT_INDEX_DIR})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_chunks = sub.add_parser("chunks", help="ingest and inspect chunks, no embedding")
    p_chunks.add_argument("--out", help="also write chunks.jsonl here")
    p_chunks.set_defaults(func=cmd_chunks)

    p_index = sub.add_parser("index", help="ingest, embed, and write the index")
    p_index.add_argument("--model", default=DEFAULT_MODEL)
    p_index.set_defaults(func=cmd_index)

    p_info = sub.add_parser("info", help="describe an existing index")
    p_info.set_defaults(func=cmd_info)

    p_search = sub.add_parser("search", help="retrieve against the index, no LLM involved")
    p_search.add_argument("query")
    p_search.add_argument("--mode", default="hybrid", choices=("dense", "bm25", "hybrid"))
    p_search.add_argument("-k", type=int, default=5, help="documents to return")
    p_search.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    p_search.add_argument(
        "--doc-type", action="append",
        help="restrict to a doc_type; repeatable. Off by default, ranking decides.",
    )
    p_search.add_argument(
        "--weight-dense", type=float, default=1.0,
        help="RRF weight for the dense leg (default 1.0, unweighted)",
    )
    p_search.add_argument(
        "--weight-lexical", type=float, default=1.0,
        help="RRF weight for the BM25 leg (default 1.0, unweighted)",
    )
    p_search.add_argument("--explain", action="store_true", help="show per-leg ranks")
    p_search.add_argument("--full", action="store_true", help="print whole documents")
    p_search.add_argument("--no-linked", action="store_true", help="skip linked documents")
    p_search.add_argument("--rerank", action="store_true",
                        help="cross-encoder reranking pass over the fused candidates")
    p_search.add_argument("--rerank-model", default=None,
                        help=f"reranker model (default: {DEFAULT_RERANK_MODEL})")
    p_search.add_argument("--rerank-candidates", type=int, default=40,
                        help="how many fused candidates the reranker scores")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="score retrieval against the labeled question set")
    p_eval.add_argument("--questions", default=Path(cfg.REPO_ROOT) / "eval" / "questions.yaml")
    p_eval.add_argument("--mode", default="hybrid", choices=("dense", "bm25", "hybrid"))
    p_eval.add_argument("--failures", action="store_true", help="list total misses")
    p_eval.add_argument("--top-k-dense", type=int, default=20,
                        help="candidates the dense leg returns before fusion")
    p_eval.add_argument("--top-k-lexical", type=int, default=20,
                        help="candidates the lexical leg returns before fusion")
    p_eval.add_argument("--rerank", action="store_true",
                        help="cross-encoder reranking pass over the fused candidates")
    p_eval.add_argument("--rerank-model", default=None,
                        help=f"reranker model (default: {DEFAULT_RERANK_MODEL})")
    p_eval.add_argument("--rerank-candidates", type=int, default=40,
                        help="how many fused candidates the reranker scores")
    p_eval.set_defaults(func=cmd_eval)

    p_sweep = sub.add_parser("sweep", help="compare retrieval configurations")
    p_sweep.add_argument("--questions", default=Path(cfg.REPO_ROOT) / "eval" / "questions.yaml")
    p_sweep.add_argument("--top-k-dense", type=int, default=20,
                        help="candidates the dense leg returns before fusion")
    p_sweep.add_argument("--top-k-lexical", type=int, default=20,
                        help="candidates the lexical leg returns before fusion")
    p_sweep.add_argument("--rerank", action="store_true",
                        help="cross-encoder reranking pass over the fused candidates")
    p_sweep.add_argument("--rerank-model", default=None,
                        help=f"reranker model (default: {DEFAULT_RERANK_MODEL})")
    p_sweep.add_argument("--rerank-candidates", type=int, default=40,
                        help="how many fused candidates the reranker scores")
    p_sweep.set_defaults(func=cmd_sweep)

    p_ask = sub.add_parser("ask", help="retrieve, gate, and answer")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=6, help="sources to retrieve")
    p_ask.add_argument("--threshold", type=float, default=DEFAULT_ABSTAIN_COSINE)
    p_ask.add_argument("--dry-run", action="store_true",
                       help="show the assembled prompt instead of calling a model")
    p_ask.add_argument("--show", type=int, default=2500, help="dry-run print limit")
    p_ask.add_argument("--model", default=os.environ.get("LABGPT_LLM_MODEL") or "gpt-4o-mini",
                       help="chat model; also LABGPT_LLM_MODEL")
    p_ask.add_argument("--base-url", default=None,
                       help="OpenAI-compatible endpoint; also LABGPT_LLM_BASE_URL")
    # Reasoning models spend this budget on thinking before writing anything, so it is a
    # ceiling on reasoning plus answer, not on answer length.
    #
    # It was 800, chosen against a one-word prompt where low effort spent no reasoning at
    # all. That generalised badly. Over 101 real questions with retrieved context, low
    # effort spent a median of 192 reasoning tokens, 448 at the 90th percentile and 704 at
    # the worst successful call, and seventeen questions consumed all 800 without writing
    # a word. 2500 leaves room for the worst of those plus a full cited answer.
    p_ask.add_argument("--max-tokens", type=int, default=2500)
    p_ask.add_argument("--reasoning-effort", default=None,
                       choices=("minimal", "low", "medium", "high"),
                       help="reasoning models only; omitted means the endpoint default")
    p_ask.add_argument("--rerank", action="store_true",
                        help="cross-encoder reranking pass over the fused candidates")
    p_ask.add_argument("--rerank-model", default=None,
                        help=f"reranker model (default: {DEFAULT_RERANK_MODEL})")
    p_ask.add_argument("--rerank-candidates", type=int, default=40,
                        help="how many fused candidates the reranker scores")
    p_ask.set_defaults(func=cmd_ask)

    p_self = sub.add_parser("selftest", help="send one prompt, to check the endpoint")
    p_self.add_argument("--model", default=os.environ.get("LABGPT_LLM_MODEL") or "gpt-4o-mini")
    p_self.add_argument("--base-url", default=None)
    p_self.add_argument("--max-tokens", type=int, default=512)
    p_self.add_argument("--reasoning-effort", default=None,
                        choices=("minimal", "low", "medium", "high"))
    p_self.set_defaults(func=cmd_selftest)

    p_all = sub.add_parser("answer-all", help="answer a whole question set, to JSONL")
    p_all.add_argument("--questions", type=Path, required=True)
    p_all.add_argument("--out", type=Path, required=True)
    p_all.add_argument("-k", type=int, default=6)
    p_all.add_argument("--threshold", type=float, default=DEFAULT_ABSTAIN_COSINE)
    p_all.add_argument("--limit", type=int, help="first N questions only")
    p_all.add_argument("--only", help="comma-separated question ids")
    p_all.add_argument("--resume", action="store_true",
                       help="append, skipping ids already in --out")
    p_all.add_argument("--oracle", action="store_true",
                       help="fill the k slots with the labelled documents, bypassing the "
                            "confidence gate, to measure the generator with retrieval held "
                            "perfect")
    p_all.add_argument("--model", default=os.environ.get("LABGPT_LLM_MODEL") or "gpt-4o-mini")
    p_all.add_argument("--base-url", default=None)
    p_all.add_argument("--max-tokens", type=int, default=2500)
    p_all.add_argument("--reasoning-effort", default=None,
                       choices=("minimal", "low", "medium", "high"))
    p_all.add_argument("--rerank", action="store_true",
                        help="cross-encoder reranking pass over the fused candidates")
    p_all.add_argument("--rerank-model", default=None,
                        help=f"reranker model (default: {DEFAULT_RERANK_MODEL})")
    p_all.add_argument("--rerank-candidates", type=int, default=40,
                        help="how many fused candidates the reranker scores")
    p_all.set_defaults(func=cmd_answer_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
