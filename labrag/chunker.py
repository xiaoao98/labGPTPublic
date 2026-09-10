"""Structure-aware chunking.

A chunk is the unit of retrieval: what gets scored against a query, and what gets pasted
into the prompt. Its size is a trade-off. Too large and irrelevant text rides along into
the context window; too small and a fragment loses the context that made it meaningful.

The naive approach is a fixed window: cut every N tokens. On this corpus that is actively
dangerous, because roughly half of it is numbered laboratory procedure where the warning
belongs to the step. Cut "Thaw the vial in a 37 C water bath for 90 to 120 seconds" away
from "Do not exceed 3 minutes" and retrieval can serve half a safety instruction.

So chunking here works on structure:

  1. Split a procedure into atoms. An atom is one numbered step together with any
     sub-items and continuation lines beneath it. Atoms are never split.
  2. Pack atoms into chunks up to a target size.
  3. Carry a little overlap between consecutive chunks, as whole sentences.

Every chunk also carries a breadcrumb. "90 to 120 seconds" is meaningless alone;
"Cell line cryorecovery > steps 1-4" is retrievable. The breadcrumb is prepended to the
text that gets embedded and indexed, which measurably helps both retrieval legs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any

# "1." "2)" "Step 3." at the start of a line: a top-level step.
STEP_RE = re.compile(r"^\s{0,3}(?:step\s+)?(\d{1,3})[.)]\s", re.IGNORECASE)
# "i." "ii." "a)" "-": a sub-item, which belongs to the step above it.
SUBSTEP_RE = re.compile(r"^\s*(?:[ivxlc]{1,5}[.)]|[a-z][.)]|[-*+])\s", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def estimate_tokens(text: str) -> int:
    """Word count scaled by 1.3.

    Deliberately dependency-free. This only decides chunk boundaries, where being within
    ten percent is fine. It is not a tokenizer and must not be used for billing or for
    context-window accounting.
    """
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str            # parent document, for small-to-big expansion at retrieval time
    doc_type: str          # safety | protocol | reagent | member | paper | paper_section
    doc_title: str
    section: str
    text: str
    source: str
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def breadcrumb(self) -> str:
        return f"{self.doc_title} > {self.section}" if self.section else self.doc_title

    @property
    def embed_text(self) -> str:
        """What actually gets embedded and BM25-indexed, breadcrumb included."""
        return f"{self.breadcrumb}\n\n{self.text}"

    @property
    def citation(self) -> str:
        """Human-readable provenance, shown beside an answer."""
        return f"{self.breadcrumb} ({self.source})"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.embed_text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


# --- step splitting -----------------------------------------------------------


def split_atoms(text: str) -> list[str]:
    """Split procedure text into indivisible units.

    A numbered step owns every following line until the next numbered step, so sub-items
    ("i. 4 mg/mL collagenase IV") stay attached to their parent ("1. Digestion buffer"),
    and a trailing warning stays attached to the step it qualifies.

    Text with no numbered steps is split on blank lines instead.
    """
    lines = text.splitlines()
    if not any(STEP_RE.match(line) for line in lines):
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text)]
        return [b for b in blocks if b]

    atoms: list[str] = []
    current: list[str] = []
    for line in lines:
        if STEP_RE.match(line) and current:
            atoms.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        atoms.append("\n".join(current).strip())
    return [a for a in atoms if a]


def step_range(atoms: list[str]) -> str:
    """Label a group of atoms as 'step 3' or 'steps 3-6', for the breadcrumb."""
    numbers = []
    for atom in atoms:
        match = STEP_RE.match(atom)
        if match:
            numbers.append(int(match.group(1)))
    if not numbers:
        return ""
    if len(numbers) == 1:
        return f"step {numbers[0]}"
    return f"steps {min(numbers)}-{max(numbers)}"


# --- packing ------------------------------------------------------------------


def pack_atoms(
    atoms: list[str],
    target_tokens: int = 220,
    overlap_tokens: int = 40,
    hard_max_tokens: int = 700,
) -> list[list[str]]:
    """Group atoms into chunks near the target size, never splitting an atom.

    An atom larger than hard_max is split on sentence boundaries as a last resort; that
    only happens for prose, since a single procedure step is never that long.
    """
    expanded: list[str] = []
    for atom in atoms:
        if estimate_tokens(atom) > hard_max_tokens:
            expanded.extend(_split_sentences(atom, target_tokens))
        else:
            expanded.append(atom)

    groups: list[list[str]] = []
    current: list[str] = []
    size = 0

    for atom in expanded:
        atom_tokens = estimate_tokens(atom)
        if current and size + atom_tokens > target_tokens:
            groups.append(current)
            tail = _tail_sentences("\n".join(current), overlap_tokens)
            current = [tail] if tail else []
            size = estimate_tokens(tail) if tail else 0
        current.append(atom)
        size += atom_tokens

    if current and any(part.strip() for part in current):
        groups.append(current)
    return groups


def _split_sentences(text: str, target_tokens: int) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in SENTENCE_RE.split(text):
        tokens = estimate_tokens(sentence)
        if current and size + tokens > target_tokens:
            out.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += tokens
    if current:
        out.append(" ".join(current))
    return out


def _tail_sentences(text: str, overlap_tokens: int) -> str:
    """Whole trailing sentences worth roughly overlap_tokens, for chunk overlap."""
    if overlap_tokens <= 0:
        return ""
    sentences = SENTENCE_RE.split(text.replace("\n", " "))
    picked: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        tokens = estimate_tokens(sentence)
        if picked and size + tokens > overlap_tokens:
            break
        picked.insert(0, sentence)
        size += tokens
        if size >= overlap_tokens:
            break
    return " ".join(picked).strip()


def chunk_procedure(
    doc_type: str,
    doc_title: str,
    text: str,
    source: str,
    doc_id: str,
    target_tokens: int = 220,
    overlap_tokens: int = 40,
    metadata: dict[str, Any] | None = None,
    section_prefix: str = "",
) -> list[Chunk]:
    """Chunk one procedure, or any prose that has no numbered steps.

    `section_prefix` names the part of the document a chunk came from, for text where
    step numbering is not the useful label. A methods section chunks as prose, so
    step_range gives nothing, and "method" is what belongs in the breadcrumb instead.
    """
    atoms = split_atoms(text)
    groups = pack_atoms(atoms, target_tokens, overlap_tokens)
    chunks: list[Chunk] = []
    for ordinal, group in enumerate(groups):
        steps = step_range(group)
        label = " ".join(part for part in (section_prefix, steps) if part)
        if section_prefix and len(groups) > 1 and not steps:
            label = f"{section_prefix} part {ordinal + 1}/{len(groups)}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{ordinal}",
                doc_id=doc_id,
                doc_type=doc_type,
                doc_title=doc_title,
                section=label,
                text="\n".join(group).strip(),
                source=source,
                ordinal=ordinal,
                metadata=dict(metadata or {}),
            )
        )
    return chunks


def write_chunks(path, chunks: list[Chunk]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_chunks(path) -> list[Chunk]:
    chunks = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
