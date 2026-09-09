"""Turn the corpus files into chunks (what gets searched) and documents (what gets returned).

Six sources with five different shapes go in; two uniform lists come out. Everything
downstream sees only those and never has to know which file a piece of text came from.

Every chunk carries the doc_id of its parent document, which is what makes small-to-big
expansion possible at retrieval time: score the chunk, serve the document. See
documents.py for why those are deliberately different granularities.

Chunk granularity is decided per corpus, and mostly the data decides it:

  members.tsv        one chunk per person   - already the right unit; "who is the safety
                                              officer" should match one row, not a slice
                                              through the middle of a table
  safety.json        one chunk per Q&A      - already atomic
  papers.json        one chunk per abstract - already atomic
  paper_content      one chunk per section  - method and result are separate answers
  protocols.csv      step-aware split       - searched in pieces, served whole
  reagents.csv       step-aware split       - searched in pieces, served whole

Splitting the small corpora further would be a mistake. A member bio cut in half retrieves
as two weak fragments instead of one strong match.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import labgpt_config as cfg  # noqa: E402

from .chunker import Chunk, chunk_procedure  # noqa: E402
from .documents import Document, build_links  # noqa: E402

# Roles carry retrieval signal ("who is the lab manager"), so they stay in the text
# rather than being pushed into metadata where nothing would index them.
MEMBER_TEMPLATE = "{name} - {role}\n{bio}"


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit] or "x"


def _single(doc: Document, section: str, ordinal: int) -> Chunk:
    """A chunk that is its own document: one bio, one Q&A, one abstract."""
    return Chunk(
        chunk_id=f"{doc.doc_id}::0",
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        doc_title=doc.title,
        section=section,
        text=doc.text,
        source=doc.source,
        ordinal=ordinal,
        metadata=dict(doc.metadata),
    )


# --- per-corpus loaders -------------------------------------------------------


def ingest_members(path=None):
    path = Path(path or cfg.MEMBERS_PATH)
    cfg.require(path)
    docs, chunks = [], []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            print(f"  skipping members.tsv line {index + 1}: expected 3 columns, got {len(parts)}")
            continue
        name, role, bio = (p.strip() for p in parts)
        doc = Document(
            doc_id=f"member-{_slug(name)}",
            doc_type="member",
            title=name,
            text=MEMBER_TEMPLATE.format(name=name, role=role, bio=bio),
            source=f"{path.name}#{index + 1}",
            metadata={"name": name, "role": role},
        )
        docs.append(doc)
        chunks.append(_single(doc, section=role, ordinal=index))
    return docs, chunks


def ingest_safety(path=None):
    path = Path(path or cfg.SAFETY_PATH)
    cfg.require(path)
    entries = json.loads(path.read_text(encoding="utf-8"))
    docs, chunks = [], []
    for index, entry in enumerate(entries):
        question = (entry.get("instruction") or "").strip()
        answer = (entry.get("output") or "").strip()
        if not (question and answer):
            continue
        # The question stays in the text because it is phrased the way users ask, which
        # gives the dense retriever a much closer target than the answer alone.
        doc = Document(
            doc_id=f"safety-{index:03d}",
            doc_type="safety",
            title="Laboratory safety guidance",
            text=f"Q: {question}\nA: {answer}",
            source=f"{path.name}#{index}",
            metadata={"question": question},
        )
        docs.append(doc)
        chunks.append(_single(doc, section=question, ordinal=index))
    return docs, chunks


def _ingest_procedure_csv(path, doc_type: str, target_tokens: int):
    path = Path(path)
    cfg.require(path)
    docs, chunks = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if len(row) < 2 or not row[0].strip():
                continue
            experiment, content = row[0].strip(), row[1].strip()
            doc_id = f"{doc_type}-{_slug(experiment)}"
            source = f"{path.name}#{index + 1}"
            docs.append(
                Document(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    title=experiment,
                    text=content,
                    source=source,
                    metadata={"experiment": experiment},
                )
            )
            chunks.extend(
                chunk_procedure(
                    doc_type=doc_type,
                    doc_title=experiment,
                    text=content,
                    source=source,
                    doc_id=doc_id,
                    target_tokens=target_tokens,
                    metadata={"experiment": experiment},
                )
            )
    return docs, chunks


def ingest_protocols(path=None):
    return _ingest_procedure_csv(path or cfg.PROTOCOLS_CSV, "protocol", target_tokens=220)


def ingest_reagents(path=None):
    # Reagent lists are denser and shorter than protocols, and an item plus its
    # sub-items is a natural answer, so they pack a little tighter.
    return _ingest_procedure_csv(path or cfg.REAGENTS_CSV, "reagent", target_tokens=160)


def ingest_papers(path=None):
    path = Path(path or cfg.PAPERS_PATH)
    cfg.require(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    docs, chunks = [], []
    for order, (paper_id, entry) in enumerate(data.items()):
        title = (entry.get("title") or "").strip()
        abstract = (entry.get("abstract") or "").strip()
        if not (title and abstract):
            continue
        doc = Document(
            doc_id=f"paper-{paper_id}-abstract",
            doc_type="paper",
            title=title,
            text=abstract,
            source=f"{path.name}#{paper_id}",
            metadata={"paper_id": paper_id},
        )
        docs.append(doc)
        chunks.append(_single(doc, section="abstract", ordinal=order))
    return docs, chunks


def ingest_paper_sections(path=None):
    path = Path(path or cfg.PAPER_CONTENT_PATH)
    cfg.require(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    docs, chunks = [], []
    order = 0
    for paper_id, entry in data.items():
        title = (entry.get("title") or "").strip()
        for section in ("method", "result"):
            body = (entry.get(section) or "").strip()
            if not body or body == "N/A":
                continue
            doc = Document(
                doc_id=f"paper-{paper_id}-{section}",
                doc_type="paper_section",
                title=title,
                text=body,
                source=f"{path.name}#{paper_id}.{section}",
                metadata={"paper_id": paper_id, "section": section},
            )
            docs.append(doc)
            chunks.append(_single(doc, section=section, ordinal=order))
            order += 1
    return docs, chunks


# --- entry point --------------------------------------------------------------

LOADERS = (
    ("member", ingest_members),
    ("safety", ingest_safety),
    ("protocol", ingest_protocols),
    ("reagent", ingest_reagents),
    ("paper", ingest_papers),
    ("paper_section", ingest_paper_sections),
)


def ingest_all(verbose: bool = True):
    """Return (documents, chunks). Documents are what get served, chunks what get searched."""
    all_docs: list[Document] = []
    all_chunks: list[Chunk] = []

    for name, loader in LOADERS:
        docs, chunks = loader()
        all_docs.extend(docs)
        all_chunks.extend(chunks)
        if verbose:
            tokens = sum(c.tokens for c in chunks)
            per_doc = f"{len(chunks) / len(docs):.1f}" if docs else "-"
            print(
                f"  {name:<14} {len(docs):>4} docs  {len(chunks):>4} chunks "
                f"({per_doc} per doc)  {tokens:>7,} tokens"
            )

    _assert_unique((c.chunk_id for c in all_chunks), "chunk_id")
    _assert_unique((d.doc_id for d in all_docs), "doc_id")

    known = {d.doc_id for d in all_docs}
    orphans = sorted({c.chunk_id for c in all_chunks if c.doc_id not in known})
    if orphans:
        raise ValueError(f"chunks with no parent document, expansion would fail: {orphans}")

    links = build_links(all_docs)
    if verbose:
        print(f"\n  linked {links} protocol/reagent pairs by experiment name")

    return all_docs, all_chunks


def _assert_unique(values, label: str) -> None:
    seen, duplicates = set(), set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {label}, which would corrupt citations: {sorted(duplicates)}")
