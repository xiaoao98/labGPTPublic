"""Command line for the retrieval layer.

    python -m labrag.cli chunks     ingest only, write chunks.jsonl, print stats
                                    (no model needed, no numpy needed)
    python -m labrag.cli index      ingest, embed, write the full index
    python -m labrag.cli info       describe an existing index

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
