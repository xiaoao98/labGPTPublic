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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import labgpt_config as cfg  # noqa: E402

from .chunker import write_chunks  # noqa: E402
from .documents import write_documents  # noqa: E402
from .embeddings import DEFAULT_MODEL, Embedder  # noqa: E402
from .ingest import ingest_all  # noqa: E402
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
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
