"""Export the protocols and reagents tables of an experiments.db into corpus CSVs.

The assistant's SQLite database is the original source of protocol and reagent text.
This dumps it into the two-column CSV shape the ingest layer reads, so a real corpus can
be indexed and searched without going through the demo pipeline.

The two tables have unrelated key spaces, which is worth knowing before expecting them to
join. Protocol keys are SOP names ("Tumor digestion for cryopreservation"), because they
came from document filenames. Reagent keys are a cell line plus an operation ("Panc1
Cryorecovery"), because they were generated per cell line from a cell bank spreadsheet.
So there is no one-to-one correspondence, and labrag's protocol/reagent linking will
match nothing here. That is expected and is why linking degrades to independent retrieval
rather than failing.

Output is two columns, no header, matching data/sample: experiment, content.

Usage:
    uv run python tools/export_experiments_db.py data/eval/experiments.db --out data/eval
    uv run python tools/export_experiments_db.py db.sqlite --prefix "" --out somewhere
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

# Table -> the column holding the body text. Older builds of this database named the
# reagent column "reagents" rather than "content", so both are accepted.
CONTENT_COLUMNS = ("content", "reagents", "protocol", "text")

STEP_RE = re.compile(r"(?m)^\s*\d{1,3}[.)]\s")


def content_column(connection: sqlite3.Connection, table: str) -> str:
    columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
    for candidate in CONTENT_COLUMNS:
        if candidate in columns:
            return candidate
    raise SystemExit(
        f"table {table!r} has no recognisable content column; saw {columns}"
    )


def export(connection: sqlite3.Connection, table: str, destination: Path) -> dict:
    column = content_column(connection, table)
    has_id = "id" in [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
    select = f'SELECT experiment, "{column}"{", id" if has_id else ""} FROM "{table}"'
    rows = connection.execute(f"{select} ORDER BY experiment").fetchall()

    written, skipped, conflicts = [], 0, 0
    seen: dict[str, str] = {}
    for row in rows:
        experiment = (row[0] or "").strip()
        content = (row[1] or "").strip()
        row_id = row[2] if has_id and len(row) > 2 else None
        if not experiment or not content:
            skipped += 1
            continue

        if experiment in seen:
            if seen[experiment] == content:
                # Same key, same text: a true duplicate, safe to drop.
                skipped += 1
                continue
            # Same key, different text. This is a real conflict in the source data:
            # "BJ Cryopreservation" exists twice with DMEM and with EMEM. Dropping one
            # would make the assistant confidently serve a single medium while the lab
            # holds two disagreeing records. Both are kept, disambiguated by row id so
            # each is traceable, and retrieval can surface the disagreement.
            conflicts += 1
            suffix = f" (record {row_id})" if row_id is not None else f" (variant {conflicts})"
            print(f"    conflicting duplicate kept as {experiment[:50]!r}{suffix}")
            experiment = f"{experiment}{suffix}"

        seen[experiment] = content
        written.append((experiment, content))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(written)

    lengths = sorted(len(c) for _, c in written)
    numbered = sum(1 for _, c in written if STEP_RE.search(c))
    return {
        "table": table,
        "column": column,
        "rows": len(rows),
        "written": len(written),
        "skipped": skipped,
        "conflicts": conflicts,
        "median_chars": lengths[len(lengths) // 2] if lengths else 0,
        "max_chars": lengths[-1] if lengths else 0,
        "numbered": numbered,
        "path": destination,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument(
        "--suffix", default="T",
        help="appended to the output stem: protocols{suffix}.csv (default: T)",
    )
    args = parser.parse_args(argv)

    if not args.database.exists():
        raise SystemExit(f"no such database: {args.database}")

    connection = sqlite3.connect(args.database)
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}

    reports = []
    for table, stem in (("protocols", "protocols"), ("reagents", "reagents")):
        if table not in tables:
            print(f"  {table}: not present in this database, skipping")
            continue
        destination = args.out / f"{stem}{args.suffix}.csv"
        reports.append(export(connection, table, destination))

    print(f"{'table':<12}{'rows':>7}{'written':>9}{'skipped':>9}{'conflict':>9}"
          f"{'median':>8}{'max':>8}{'numbered':>10}  file")
    print("-" * 88)
    for report in reports:
        print(
            f"{report['table']:<12}{report['rows']:>7}{report['written']:>9}"
            f"{report['skipped']:>9}{report['conflicts']:>9}{report['median_chars']:>8}{report['max_chars']:>8}"
            f"{report['numbered']:>10}  {report['path']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
