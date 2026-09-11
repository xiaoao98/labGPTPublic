"""Turn the corpus files into chunks (what gets searched) and documents (what gets returned).

Six sources with five different shapes go in; two uniform lists come out. Everything
downstream sees only those and never has to know which file a piece of text came from.

Every chunk carries the doc_id of its parent document, which is what makes small-to-big
expansion possible at retrieval time: score the chunk, serve the document. See
documents.py for why those are deliberately different granularities.

Chunk granularity is decided per corpus, and mostly the data decides it:

  members.tsv        sentence-level split   - searched in pieces, served whole. See
                                              ingest_members for why this reverses an
                                              earlier decision to keep bios intact.
  safety.json        one chunk per Q&A      - already atomic
  papers.json        one chunk per abstract - already atomic
  paper_content      one chunk per section  - method and result are separate answers
  protocols.csv      step-aware split       - searched in pieces, served whole
  reagents.csv       step-aware split       - searched in pieces, served whole

Safety entries, abstracts and paper sections stay whole because each is already a single
answer. Member bios do not: see ingest_members.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import labgpt_config as cfg  # noqa: E402

from .chunker import Chunk, chunk_procedure, estimate_tokens  # noqa: E402
from .documents import Document, build_links  # noqa: E402

# Roles carry retrieval signal ("who is the lab manager"), so they stay in the text
# rather than being pushed into metadata where nothing would index them.
MEMBER_TEMPLATE = "{name} - {role}\n{bio}"

# An abstract longer than this is not an abstract, it is a proceedings volume.
MAX_ABSTRACT_TOKENS = 900

# Roughly two sentences of a biography. Small enough that one relevant sentence is not
# drowned by the rest, large enough that a fragment still reads as a statement.
MEMBER_CHUNK_TOKENS = 60


def _slug(text: str, limit: int = 40) -> str:
    """A readable, collision-free id fragment.

    The hash suffix is not decoration. Truncating to a readable length collides in real
    corpora: two protocol articles both have a stage headed "Quantification and
    statistical analysis", and once the trailing article id is cut off by the limit their
    slugs are identical. Colliding ids silently merge two documents, so the hash of the
    full text is appended to keep them distinct whatever the truncation does.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].strip("-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{digest}" if slug else digest


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
        # Bios are chunked, which reverses an earlier decision to keep each one whole.
        # A bio is around 180 tokens of education, previous posts, current project and
        # hobbies, and a question like "who runs the flow cytometry platforms" is answered
        # by one sentence of it. Embedding the whole thing dilutes that sentence into the
        # rest, and `people` was consequently the worst category for dense retrieval in
        # every sweep. Each chunk still carries the name and role in its breadcrumb, so a
        # fragment is never anonymous, and member is an expanding type so the whole bio is
        # still what gets served.
        chunks.extend(
            chunk_procedure(
                doc_type="member",
                doc_title=name,
                text=bio,
                source=f"{path.name}#{index + 1}",
                doc_id=doc.doc_id,
                target_tokens=MEMBER_CHUNK_TOKENS,
                overlap_tokens=0,
                hard_max_tokens=MEMBER_CHUNK_TOKENS + 20,
                metadata={"name": name, "role": role},
                section_prefix=role,
            )
        )
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
        # An abstract is an atomic unit and splitting one produces two weak fragments
        # instead of one strong match, so it stays whole. The guard is for corpus
        # accidents: a conference proceedings volume arrives as a single "abstract" of
        # a hundred thousand characters, and one of those must not become one chunk.
        if estimate_tokens(abstract) > MAX_ABSTRACT_TOKENS:
            print(
                f"  {paper_id}: abstract is {estimate_tokens(abstract):,} tokens, "
                f"splitting; it is probably a proceedings volume"
            )
            chunks.extend(
                chunk_procedure(
                    doc_type="paper",
                    doc_title=title,
                    text=abstract,
                    source=f"{path.name}#{paper_id}",
                    doc_id=doc.doc_id,
                    target_tokens=300,
                    metadata={"paper_id": paper_id},
                    section_prefix="abstract",
                )
            )
        else:
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
            doc_id = f"paper-{paper_id}-{section}"
            source = f"{path.name}#{paper_id}.{section}"
            docs.append(
                Document(
                    doc_id=doc_id,
                    doc_type="paper_section",
                    title=title,
                    text=body,
                    source=source,
                    metadata={"paper_id": paper_id, "section": section},
                )
            )
            # Methods and results sections are prose and can run to several thousand
            # tokens on a real article, so unlike an abstract they have to be split.
            # chunk_procedure falls back to paragraph packing when it finds no numbered
            # steps, which is the right behaviour for prose.
            chunks.extend(
                chunk_procedure(
                    doc_type="paper_section",
                    doc_title=title,
                    text=body,
                    source=source,
                    doc_id=doc_id,
                    target_tokens=260,
                    metadata={"paper_id": paper_id, "section": section},
                    section_prefix=section,
                )
            )
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
