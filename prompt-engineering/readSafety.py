"""Convert a question/answer source file into instruction-tuning JSON.

This file previously held 49 question and answer pairs as hardcoded variables, then used
`eval(f"question{i}")` in a loop to collect them. That embedded an institution's safety
corpus directly in source, and it meant adding an entry required editing code.

Now it reads a TSV and writes the same output format, so the data lives in the corpus
directory where it belongs.

Input format, tab-separated, one pair per line, blank lines and # comments ignored:

    question<TAB>answer

Usage:
    python prompt-engineering/readSafety.py input.tsv output.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import labgpt_config as cfg  # noqa: E402


def read_qa_pairs(path):
    """Parse a TSV of question/answer pairs."""
    pairs = []
    with open(cfg.require(path), encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"  skipping line {line_number}, no tab separator: {line[:60]}")
                continue
            question, answer = parts[0].strip(), parts[1].strip()
            if question and answer:
                pairs.append((question, answer))
    return pairs


def to_instruction_format(pairs):
    """The {instruction, input, output} shape the rest of the pipeline expects."""
    return [
        {"instruction": question, "input": "", "output": answer}
        for question, answer in pairs
    ]


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 1

    source, destination = Path(argv[1]), Path(argv[2])
    pairs = read_qa_pairs(source)
    if not pairs:
        print(f"No usable pairs found in {source}.")
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(to_instruction_format(pairs), handle, indent=4, ensure_ascii=False)

    print(f"Wrote {len(pairs)} pairs to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
