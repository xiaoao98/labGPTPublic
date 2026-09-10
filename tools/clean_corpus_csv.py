"""Clean the CSVs exported from experiments.db: drop junk, fix keys, split merged protocols.

Kept separate from export_experiments_db.py so that export stays a faithful dump and every
edit to the corpus is a declared, auditable rule rather than something buried in the
extraction. Re-run export then clean and the result is reproducible.

WHAT IS DROPPED, and why each one:

  reagents
    Two rows whose formulation is the string "nan". One is nothing but that; the other is
    "nan; PBS; Cell dissociation buffer (...)", which is worse, because the boilerplate
    suffix makes it look populated while the actual medium is missing. An assistant would
    read "nan" out as if it were a reagent.

  protocols
    Three genotyping documents that duplicate another entry. Confirmed by comparing primer
    sequences rather than titles: template reuse across this corpus pushes unrelated
    protocols to 0.8 title similarity, so similarity alone would have deleted real
    content. Podoplanin Cre vs Podoplanin TK, Col1a1 TK vs FAP-TK, and Rab27btm1a vs
    Sdcbptm1a all look like duplicates and are not; their primers differ, so they are
    different alleles or genes and all six are kept.

WHAT IS NOT DROPPED, because it looks droppable and is not:

  The 61 genotyping documents in "1) ORIGINAL PAPER / 2) SOURCE" form have no procedural
  steps and read like metadata cards, but they carry primer sequences, PCR master mixes,
  thermocycler programs and expected band sizes. That is the densest factual content in
  the corpus and exactly what the assistant should be able to answer.

  Two very short reagent rows, "DMEM, 20% FBS" and "RPMI, 15% FBS", are terse but complete
  formulations, not truncation.

SPLITTING MERGED PROTOCOLS

Seventeen documents each hold several real protocols merged together, up to 9,562
characters covering four unrelated procedures. Retrieving one drags all of them into the
prompt, and no chunk boundary can fix that because the document is the unit that gets
served. They are split on their own headings.

The heading rule is ordered, because the markers are not equally trustworthy:

  1. "Part N: title"    reliable, always a real section heading
  2. "Method N: title"  only when the document has no Part headings. Inside a document
                        that has them, "Method 1:" is used for inline alternatives within
                        a step ("Method 1: Dilute DiI dye in DMSO at a 1:7 ratio."), and
                        splitting there would cut a step in half
  3. "A. Title"         alphabetic sections, for documents with neither of the above
  4. otherwise          left whole; the remaining ones are short enough to serve as-is

Each piece keeps its parent title as a prefix, so provenance survives and a query naming
the parent still matches.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# --- declared deletions -------------------------------------------------------

DROP_REAGENTS = {
    "NK-92  Cryorecovery": "content is the string 'nan'",
    "U87 MG  Mcherry Luc Cryopreservation": "medium missing, content begins 'nan;'",
}

DROP_PROTOCOLS = {
    "ROSA rtTA genotyping protocol  2": "byte-identical to 'ROSA rtTA genotyping protocol'",
    "2-R26-LSL-mCherry-CD9 genotyping protocol":
        "same 4 primers as 'Rosa26-LSL-mCherry-CD9 genotyping protocol'",
    "cMycFF Touchdown Genotyping Protocol":
        "same 4 primers as 'Myc Flox Genotyping Protocol'",
}

RENAME = {
    "XCR1 genotyping.docx": "XCR1 genotyping",   # filename extension left in the key
    "Exasome isolation": "Exosome isolation",     # misspelling; breaks lexical retrieval
}

# --- splitting ----------------------------------------------------------------

CONSOLIDATED_RE = re.compile(
    r"(?i)^(consolidated|master protocol)|this document (consolidates|combines|outlines)"
)
PART_RE = re.compile(r"(?m)^[ \t]*(Part\s+\d+)\s*[:.–-]\s*(.+?)[ \t]*$", re.I)
METHOD_RE = re.compile(r"(?m)^[ \t]*(Method\s+\d+)\s*[:.–-]\s*(.+?)[ \t]*$", re.I)
ALPHA_RE = re.compile(r"(?m)^[ \t]*([A-H])[.)]\s+([A-Z][^\n]{3,70})[ \t]*$")

# A heading names a thing; a sentence describes an action and ends in a full stop.
SENTENCE_TAIL_RE = re.compile(r"[.!?]$")


def _looks_like_heading(title: str) -> bool:
    return len(title) <= 80 and not SENTENCE_TAIL_RE.search(title.strip())


def split_consolidated(title: str, content: str) -> list[tuple[str, str]]:
    """Return [(name, text)]; a single element means it was left whole."""
    for regex, kind in ((PART_RE, "part"), (METHOD_RE, "method"), (ALPHA_RE, "alpha")):
        matches = [m for m in regex.finditer(content) if _looks_like_heading(m.group(2))]
        if len(matches) < 2:
            continue

        pieces: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            body = content[match.start():end].strip()
            if len(body) < 120:
                # Too short to be a protocol on its own; almost certainly a stray match.
                continue
            label = match.group(2).strip().rstrip(":")
            pieces.append((f"{title} - {label}", body))

        if len(pieces) >= 2:
            # The preamble before the first heading carries the overview; keep it so the
            # document is still findable by its own name.
            preamble = content[: matches[0].start()].strip()
            if len(preamble) >= 120:
                pieces.insert(0, (f"{title} - overview", preamble))
            return pieces
    return [(title, content)]


# --- driver -------------------------------------------------------------------


def load(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [(row[0], row[1]) for row in csv.reader(handle) if len(row) >= 2]


def write(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def normalise_key(key: str) -> str:
    return re.sub(r"\s+", " ", RENAME.get(key, key)).strip()


def clean_reagents(rows):
    kept, dropped = [], []
    for key, content in rows:
        if key in DROP_REAGENTS:
            dropped.append((key, DROP_REAGENTS[key]))
            continue
        kept.append((normalise_key(key), content))
    return kept, dropped


def clean_protocols(rows, split: bool):
    kept, dropped, split_report = [], [], []
    for key, content in rows:
        if key in DROP_PROTOCOLS:
            dropped.append((key, DROP_PROTOCOLS[key]))
            continue
        name = normalise_key(key)
        if split and CONSOLIDATED_RE.search(content.strip()[:200]):
            pieces = split_consolidated(name, content)
            if len(pieces) > 1:
                split_report.append((name, len(pieces)))
            kept.extend(pieces)
        else:
            kept.append((name, content))
    return kept, dropped, split_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--suffix", default="T")
    parser.add_argument("--no-split", action="store_true", help="drop junk but do not split")
    args = parser.parse_args(argv)

    protocols_path = args.dir / f"protocols{args.suffix}.csv"
    reagents_path = args.dir / f"reagents{args.suffix}.csv"
    for path in (protocols_path, reagents_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run tools/export_experiments_db.py first")

    reagents, dropped_r = clean_reagents(load(reagents_path))
    protocols, dropped_p, split_report = clean_protocols(
        load(protocols_path), split=not args.no_split
    )

    print("dropped from reagents:")
    for key, reason in dropped_r:
        print(f"    {key:<44} {reason}")
    print("dropped from protocols:")
    for key, reason in dropped_p:
        print(f"    {key:<44} {reason}")

    if split_report:
        print(f"\nsplit {len(split_report)} merged documents into "
              f"{sum(n for _, n in split_report)} protocols:")
        for name, count in split_report:
            print(f"    {name[:52]:<54} -> {count}")

    duplicates = {k for k, _ in protocols if [x for x, _ in protocols].count(k) > 1}
    if duplicates:
        print(f"\nWARNING duplicate protocol keys after cleaning: {sorted(duplicates)}")

    write(reagents_path, reagents)
    write(protocols_path, protocols)
    print(f"\nreagents  {len(reagents)} rows -> {reagents_path}")
    print(f"protocols {len(protocols)} rows -> {protocols_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
