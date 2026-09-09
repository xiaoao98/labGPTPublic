"""Build experiments.db from the corpus CSV files.

The database is generated rather than committed. Binary artifacts in git are noise, they
do not diff, and in this repository's case they were also carrying institutional data.

Usage:
    python buildDB/build_sample_db.py

Builds against data/sample by default. Set LABGPT_DATA_DIR to build against a private
corpus with the same file layout.
"""

import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import labgpt_config as cfg  # noqa: E402


def read_two_column_csv(path):
    """Read `name,content` rows. Content may span lines when quoted."""
    cfg.require(path)
    rows = []
    with open(path, mode="r", newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 2:
                if any(cell.strip() for cell in row):
                    print(f"  skipping malformed row {line_number} in {path.name}: {row}")
                continue
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def build(db_path=None):
    db_path = Path(db_path or cfg.DATABASE_PATH)
    protocols = read_two_column_csv(cfg.PROTOCOLS_CSV)
    reagents = read_two_column_csv(cfg.REAGENTS_CSV)

    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        # Column names match what protocolDemo.py queries: experiment, content.
        for table in ("protocols", "reagents"):
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            cursor.execute(
                f"CREATE TABLE {table} ("
                "  experiment TEXT PRIMARY KEY,"
                "  content    TEXT NOT NULL"
                ")"
            )
        cursor.executemany(
            "INSERT INTO protocols (experiment, content) VALUES (?, ?)", protocols
        )
        cursor.executemany(
            "INSERT INTO reagents (experiment, content) VALUES (?, ?)", reagents
        )
        connection.commit()
    finally:
        connection.close()

    print(f"Wrote {db_path}")
    print(f"  protocols: {len(protocols)}")
    print(f"  reagents:  {len(reagents)}")
    return db_path


if __name__ == "__main__":
    build()
