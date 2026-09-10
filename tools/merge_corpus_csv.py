"""Append the lab-database CSVs onto the PMC-derived ones, into one evaluation corpus.

The evaluation corpus has two sources with different characters, and mixing them is the
point: PMC supplies breadth and public provenance, the lab database supplies the actual
protocols the assistant exists to serve. Retrieval has to choose between them.

They stay distinguishable without any extra bookkeeping, because PMC-derived keys all
carry their article id ("Preparation of electroporation (PMC8683654)") and lab keys never
do ("Cryorecovery"). That is enough to slice results by source later.

ORDER MATTERS, and this is the trap. The pipeline is:

    fetch_eval_corpus.py     writes protocols.csv and reagents.csv   (PMC, overwrites)
    export_experiments_db.py writes protocolsT.csv and reagentsT.csv (lab, overwrites)
    clean_corpus_csv.py      rewrites the T files in place
    merge_corpus_csv.py      appends T onto the main files

Re-running the fetch discards the merge, because it overwrites rather than appends. So a
rebuild means running all four again in that order. This script refuses to run twice
against the same data rather than silently doubling the corpus.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [(row[0], row[1]) for row in csv.reader(handle) if len(row) >= 2]


def merge(base_path: Path, extra_path: Path, dry_run: bool) -> dict:
    base, extra = load(base_path), load(extra_path)
    base_keys = {key for key, _ in base}

    already = sum(1 for key, _ in extra if key in base_keys)
    if already == len(extra) and extra:
        raise SystemExit(
            f"{extra_path.name} appears to have been merged into {base_path.name} "
            f"already ({already}/{len(extra)} keys present). Re-run the fetch and export "
            f"steps to rebuild from scratch rather than merging twice."
        )

    added, collided = [], []
    for key, content in extra:
        if key in base_keys:
            collided.append(key)
            continue
        added.append((key, content))
        base_keys.add(key)

    if not dry_run:
        with base_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(added)

    return {
        "base": base_path.name,
        "before": len(base),
        "added": len(added),
        "collided": collided,
        "after": len(base) + len(added),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--suffix", default="T")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    reports = []
    for stem in ("protocols", "reagents"):
        base = args.dir / f"{stem}.csv"
        extra = args.dir / f"{stem}{args.suffix}.csv"
        for path in (base, extra):
            if not path.exists():
                raise SystemExit(f"missing {path}")
        reports.append(merge(base, extra, args.dry_run))

    print(f"{'file':<18}{'before':>8}{'added':>8}{'after':>8}{'collisions':>12}")
    print("-" * 54)
    for report in reports:
        print(f"{report['base']:<18}{report['before']:>8}{report['added']:>8}"
              f"{report['after']:>8}{len(report['collided']):>12}")
        for key in report["collided"][:5]:
            print(f"    skipped duplicate key: {key[:60]}")
    if args.dry_run:
        print("\ndry run, nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
