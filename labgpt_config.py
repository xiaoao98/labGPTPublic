"""Central configuration.

Six corpus paths and the root they hang off. Paths used to be hardcoded as module-level
constants in five separate files, two of which pointed at files that were not where the
constant said they were; everything resolves through here now.

The model name and the SQLite database path also lived here, and left with the code that
read them. Retrieval reads the corpus files directly, and the generation model is named
where the client is built.

The corpus location is an environment variable so that a private corpus can live outside
the repository. Default is the synthetic sample set in data/sample, which is what ships.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Point this at a private corpus directory to run against real data.
# The directory must contain the same filenames as data/sample.
DATA_DIR = Path(os.environ.get("LABGPT_DATA_DIR", REPO_ROOT / "data" / "sample"))

MEMBERS_PATH = DATA_DIR / "members.tsv"
SAFETY_PATH = DATA_DIR / "safety.json"
PROTOCOLS_CSV = DATA_DIR / "protocols.csv"
REAGENTS_CSV = DATA_DIR / "reagents.csv"
PAPERS_PATH = DATA_DIR / "papers.json"
PAPER_CONTENT_PATH = DATA_DIR / "paper_content.json"


def require(path):
    """Fail loudly and early when a corpus file is missing.

    The previous code caught FileNotFoundError, printed, and returned None, which turned a
    missing file into an AttributeError three frames away. A missing corpus file is not
    recoverable, so it raises here with the path and the fix.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Expected corpus files in {DATA_DIR}.\n"
            f"Set LABGPT_DATA_DIR to point at your corpus, or use the synthetic sample "
            f"set in data/sample."
        )
    return path
