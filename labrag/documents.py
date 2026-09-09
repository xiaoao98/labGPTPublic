"""Documents, and expanding retrieved chunks back up to them.

Retrieval and delivery want different granularities, and conflating them is a common way
to build a RAG system that is either imprecise or unsafe.

  Retrieval wants small units. A query like "which protocols use RIPA buffer" or "how long
  in the water bath" only matches if the body text is indexed. Index titles alone and
  those queries can never hit anything.

  Delivery wants whole units. A protocol is executed end to end, so handing back steps 1
  to 9 without step 10 is worse than handing back nothing: the reader follows what they
  were given and silently skips the rest.

So chunks are what get scored, and documents are what get returned. This is the pattern
usually called small-to-big, or parent-document retrieval.

Expansion applies only to the doc types where a document is genuinely larger than its
chunks. Member bios, safety Q&A and paper abstracts are already whole documents, so their
chunk and document texts are identical and expanding them is a no-op.

Linking is separate from expansion. A protocol and a reagent list for the same experiment
are related but not the same document, and in the real corpus their key spaces do not even
match: reagent keys are generated per cell line ("<Cell line> Cryorecovery") while protocol
keys are SOP document filenames. So a link is attached when the names happen to match and
is simply absent when they do not.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

# Doc types where the document is bigger than the chunk and must be served whole.
EXPANDING_TYPES = frozenset({"protocol", "reagent"})

# Doc types that are linked to each other when they describe the same experiment.
LINKED_TYPES = (("protocol", "reagent"),)


@dataclass
class Document:
    doc_id: str
    doc_type: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    linked_doc_ids: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.title} ({self.source})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass
class Retrieved:
    """One document on its way into a prompt, with the chunks that found it."""

    document: Document
    score: float
    matched_chunk_ids: list[str] = field(default_factory=list)
    via_link_from: str | None = None  # set when pulled in as a linked document

    @property
    def is_linked(self) -> bool:
        return self.via_link_from is not None


def normalize_key(text: str) -> str:
    """Fold an experiment name for cross-corpus matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def build_links(documents: Iterable[Document]) -> int:
    """Attach linked_doc_ids between paired doc types. Returns the number of links made.

    Matching is on the normalised experiment name. Where the two key spaces do not
    overlap, which is the expected case on the real corpus, nothing is linked and
    retrieval simply treats the two corpora independently.
    """
    documents = list(documents)
    by_type: dict[str, dict[str, Document]] = {}
    for doc in documents:
        title_key = normalize_key(doc.metadata.get("experiment") or doc.title)
        by_type.setdefault(doc.doc_type, {})[title_key] = doc

    made = 0
    for left_type, right_type in LINKED_TYPES:
        left = by_type.get(left_type, {})
        right = by_type.get(right_type, {})
        for key, left_doc in left.items():
            right_doc = right.get(key)
            if right_doc is None:
                continue
            if right_doc.doc_id not in left_doc.linked_doc_ids:
                left_doc.linked_doc_ids.append(right_doc.doc_id)
                made += 1
            if left_doc.doc_id not in right_doc.linked_doc_ids:
                right_doc.linked_doc_ids.append(left_doc.doc_id)
    return made


def expand_hits(
    hits,
    documents: dict[str, Document],
    expand_types: frozenset[str] = EXPANDING_TYPES,
    include_linked: bool = True,
    limit: int | None = None,
) -> list[Retrieved]:
    """Turn scored chunks into whole documents, deduplicated, best score first.

    `hits` is an iterable of (chunk, score). Several chunks of one protocol collapse into
    a single Retrieved carrying the best of their scores, so a long document does not
    crowd out everything else by occupying several slots.
    """
    order: list[str] = []
    picked: dict[str, Retrieved] = {}

    for chunk, score in hits:
        doc = documents.get(chunk.doc_id)
        if doc is None:
            # No parent on record: fall back to the chunk itself rather than dropping it.
            doc = Document(
                doc_id=chunk.doc_id or chunk.chunk_id,
                doc_type=chunk.doc_type,
                title=chunk.doc_title,
                text=chunk.text,
                source=chunk.source,
                metadata=dict(chunk.metadata),
            )
        if chunk.doc_type not in expand_types:
            # Chunk already is the document; keep the chunk text verbatim.
            doc = Document(
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                title=doc.title,
                text=chunk.text,
                source=chunk.source,
                metadata=doc.metadata,
                linked_doc_ids=list(doc.linked_doc_ids),
            )

        existing = picked.get(doc.doc_id)
        if existing is None:
            picked[doc.doc_id] = Retrieved(
                document=doc, score=score, matched_chunk_ids=[chunk.chunk_id]
            )
            order.append(doc.doc_id)
        else:
            existing.score = max(existing.score, score)
            if chunk.chunk_id not in existing.matched_chunk_ids:
                existing.matched_chunk_ids.append(chunk.chunk_id)

    results = sorted(
        (picked[doc_id] for doc_id in order), key=lambda r: -r.score
    )
    if limit is not None:
        results = results[:limit]

    if include_linked:
        results = _attach_linked(results, documents)
    return results


def _attach_linked(results: list[Retrieved], documents: dict[str, Document]) -> list[Retrieved]:
    """Append linked documents directly after the result that pulled them in.

    A linked document inherits a slightly lower score than its parent so that ordering
    stays stable, and is marked so a caller can tell it was not itself a match.
    """
    present = {r.document.doc_id for r in results}
    out: list[Retrieved] = []
    for result in results:
        out.append(result)
        for linked_id in result.document.linked_doc_ids:
            if linked_id in present:
                continue
            linked = documents.get(linked_id)
            if linked is None:
                continue
            present.add(linked_id)
            out.append(
                Retrieved(
                    document=linked,
                    score=result.score - 1e-6,
                    matched_chunk_ids=[],
                    via_link_from=result.document.doc_id,
                )
            )
    return out


def write_documents(path, documents: Iterable[Document]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for doc in documents:
            handle.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def read_documents(path) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                doc = Document.from_dict(json.loads(line))
                documents[doc.doc_id] = doc
    return documents
