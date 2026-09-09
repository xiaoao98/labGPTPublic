"""The index on disk, and dense search over it.

Four files in the index directory:

    chunks.jsonl     one chunk per line, in the same order as the vectors
    documents.jsonl  the parent documents, keyed by doc_id
    vectors.npy      a float32 matrix, one row per chunk
    manifest.json    what built it: model, dimensions, counts, timestamp

Chunks are what get searched; documents are what get returned. See documents.py for why
those are deliberately different granularities.

The manifest exists so that a stale index announces itself. Change the embedding model or
edit the corpus, and the numbers in there stop matching, which is a clear error instead of
silently wrong retrieval.

Search is a brute-force matrix multiply, deliberately. This corpus is 82 chunks; a real
one is a few thousand. At 3000 x 384 float32 that is under 5 MB of RAM and roughly a
millisecond per query. A vector database would add a service to run, a client library, and
network latency, and would return the same rows. Do the arithmetic before adding
infrastructure: FAISS or pgvector start to pay off somewhere around a hundred thousand
chunks, which is more than a thousand times this corpus.

The interface below is the whole surface area, so swapping in pgvector later means writing
one class, not rewriting retrieval.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .chunker import Chunk, read_chunks, write_chunks
from .documents import Document, read_documents, write_documents

CHUNKS_FILE = "chunks.jsonl"
DOCUMENTS_FILE = "documents.jsonl"
VECTORS_FILE = "vectors.npy"
MANIFEST_FILE = "manifest.json"


class IndexError_(RuntimeError):
    pass


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors, manifest: dict, documents=None):
        if len(chunks) != vectors.shape[0]:
            raise IndexError_(
                f"index is inconsistent: {len(chunks)} chunks but {vectors.shape[0]} "
                f"vectors. Re-run: python -m labrag.cli index"
            )
        self.chunks = chunks
        self.vectors = vectors
        self.manifest = manifest
        self.documents: dict[str, Document] = documents or {}

        missing = {c.doc_id for c in chunks} - set(self.documents)
        if self.documents and missing:
            raise IndexError_(
                f"{len(missing)} chunks have no parent document, so expansion would "
                f"silently drop them. Re-run: python -m labrag.cli index"
            )

    # -- persistence --

    @staticmethod
    def build(index_dir, chunks: list[Chunk], vectors, model_name: str,
              documents=None) -> "VectorStore":
        import numpy as np

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        write_chunks(index_dir / CHUNKS_FILE, chunks)
        np.save(index_dir / VECTORS_FILE, vectors.astype("float32"))

        documents = documents or []
        if not isinstance(documents, dict):
            documents = {d.doc_id: d for d in documents}
        write_documents(index_dir / DOCUMENTS_FILE, documents.values())

        counts: dict[str, int] = {}
        doc_counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.doc_type] = counts.get(chunk.doc_type, 0) + 1
        for doc in documents.values():
            doc_counts[doc.doc_type] = doc_counts.get(doc.doc_type, 0) + 1

        manifest = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embedding_model": model_name,
            "dim": int(vectors.shape[1]),
            "chunks": len(chunks),
            "documents": len(documents),
            "by_doc_type": counts,
            "documents_by_doc_type": doc_counts,
            "links": sum(len(d.linked_doc_ids) for d in documents.values()) // 2,
        }
        (index_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return VectorStore(chunks, vectors.astype("float32"), manifest, documents)

    @staticmethod
    def load(index_dir) -> "VectorStore":
        import numpy as np

        index_dir = Path(index_dir)
        if not (index_dir / MANIFEST_FILE).exists():
            raise IndexError_(
                f"no index at {index_dir}.\n  python -m labrag.cli index"
            )
        manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        chunks = read_chunks(index_dir / CHUNKS_FILE)
        vectors = np.load(index_dir / VECTORS_FILE)
        documents_path = index_dir / DOCUMENTS_FILE
        documents = read_documents(documents_path) if documents_path.exists() else {}
        return VectorStore(chunks, vectors, manifest, documents)

    # -- search --

    def search_dense(self, query_vector, k: int = 10, doc_types=None):
        """Return [(chunk_index, cosine_similarity)], highest first.

        Both sides are unit-normalised, so this single matrix multiply *is* the cosine
        similarity of the query against every chunk at once.
        """
        import numpy as np

        if self.vectors.size == 0:
            return []

        scores = self.vectors @ np.asarray(query_vector, dtype="float32").reshape(-1)

        if doc_types:
            allowed = set(doc_types)
            mask = np.array([c.doc_type in allowed for c in self.chunks])
            # -inf rather than 0: a genuine cosine can be negative, and zeroing would
            # rank an excluded chunk above a weakly-but-honestly-matching one.
            scores = np.where(mask, scores, -np.inf)

        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > -np.inf]

    def __len__(self) -> int:
        return len(self.chunks)

    def describe(self) -> str:
        lines = [
            f"index built {self.manifest.get('built_at')}",
            f"  model     {self.manifest.get('embedding_model')} ({self.manifest.get('dim')} dims)",
            f"  chunks    {len(self.chunks)}  (searched)",
            f"  documents {self.manifest.get('documents', len(self.documents))}  (served)",
            f"  links     {self.manifest.get('links', 0)} protocol/reagent pairs",
        ]
        chunk_counts = self.manifest.get("by_doc_type", {})
        doc_counts = self.manifest.get("documents_by_doc_type", {})
        if chunk_counts:
            lines.append(f"    {'doc_type':<14}{'docs':>6}{'chunks':>8}")
            for doc_type in sorted(chunk_counts):
                lines.append(
                    f"    {doc_type:<14}{doc_counts.get(doc_type, 0):>6}"
                    f"{chunk_counts[doc_type]:>8}"
                )
        return "\n".join(lines)
