"""Per-generation-call instrumentation.

Records prompt tokens, generated tokens, and wall time for every call to the model, as
one JSON line per call. The point is to have a real measured baseline before the retrieval
work changes anything, so that "prompt size dropped from X to Y" is a number that came out
of a run rather than an estimate.

Token counts come from the tensors themselves, not from an approximation, so they are
exact for the tokenizer in use.

Disabled unless LABGPT_METRICS is set:

    LABGPT_METRICS=metrics.jsonl python demo.py

Read the results with:

    python labgpt_metrics.py metrics.jsonl
"""

import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

METRICS_PATH = os.environ.get("LABGPT_METRICS")


def enabled():
    return bool(METRICS_PATH)


def record(label, prompt_tokens, generated_tokens, elapsed_s, **extra):
    """Append one call record. Silently does nothing when metrics are off."""
    if not METRICS_PATH:
        return
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated_tokens),
        "elapsed_s": round(float(elapsed_s), 3),
        "tokens_per_s": (
            round(generated_tokens / elapsed_s, 1) if elapsed_s > 0 else None
        ),
        **extra,
    }
    with open(METRICS_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def timer():
    """Yields a one-element list that receives the elapsed seconds on exit."""
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start


def summarize(path):
    """Print a per-label breakdown. This is the before/after table."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print(f"{path} is empty.")
        return

    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)

    header = f"{'label':<22}{'calls':>7}{'prompt tok':>13}{'gen tok':>10}{'sec':>9}"
    print(header)
    print("-" * len(header))

    total_prompt = total_gen = total_time = 0
    for label in sorted(groups):
        group = groups[label]
        prompt = sum(r["prompt_tokens"] for r in group)
        gen = sum(r["generated_tokens"] for r in group)
        secs = sum(r["elapsed_s"] for r in group)
        total_prompt += prompt
        total_gen += gen
        total_time += secs
        print(
            f"{label:<22}{len(group):>7}{prompt:>13,}{gen:>10,}{secs:>9.1f}"
            f"   (mean prompt {prompt // len(group):,})"
        )

    print("-" * len(header))
    print(f"{'TOTAL':<22}{len(rows):>7}{total_prompt:>13,}{total_gen:>10,}{total_time:>9.1f}")

    queries = len({r.get("query_id") for r in rows if r.get("query_id") is not None})
    if queries:
        print(
            f"\n{queries} queries: "
            f"{total_prompt // queries:,} prompt tokens and "
            f"{total_time / queries:.1f}s per query on average"
        )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else METRICS_PATH
    if not target or not Path(target).exists():
        print(__doc__)
        sys.exit(1)
    summarize(target)
