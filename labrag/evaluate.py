"""Evaluation: label checking, retrieval metrics, and configuration sweeps.

Labels are doc_ids, so the question set survives re-chunking. Change the chunk size and
the labels stay valid, which is what makes it possible to sweep chunking as well as
retrieval settings.

`check_labels` exists because a wrong label is worse than a missing one. A doc_id that
does not resolve silently makes a question unanswerable and drags every configuration's
score down equally, so the mistake hides: the comparison still looks sensible and the
absolute numbers are quietly wrong. It is run before every evaluation.

Metrics, all computed over documents rather than chunks:

    recall@k   fraction of the relevant documents that appear in the top k
    MRR        1 / rank of the first relevant document, averaged
    nDCG@k     rewards putting relevant documents high, not merely inside the window

For abstention, the question set carries deliberately unanswerable questions and the
measure is how well the confidence signal separates them from answerable ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Question:
    id: str
    question: str
    category: str
    relevant: list[str] = field(default_factory=list)
    should_abstain: bool = False
    note: str = ""

    @property
    def answerable(self) -> bool:
        return not self.should_abstain


def load_questions(path) -> list[Question]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("questions", raw) if isinstance(raw, dict) else raw
    questions = []
    for entry in entries:
        questions.append(
            Question(
                id=entry["id"],
                question=entry["question"],
                category=entry.get("category", "uncategorised"),
                relevant=list(entry.get("relevant") or []),
                should_abstain=bool(entry.get("should_abstain", False)),
                note=entry.get("note", ""),
            )
        )
    return questions


# --- label checking -----------------------------------------------------------


def check_labels(questions: list[Question], documents: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors mean the set cannot be trusted."""
    errors: list[str] = []
    warnings: list[str] = []
    known = set(documents)

    seen_ids: set[str] = set()
    for question in questions:
        if question.id in seen_ids:
            errors.append(f"{question.id}: duplicate question id")
        seen_ids.add(question.id)

        if question.should_abstain and question.relevant:
            errors.append(f"{question.id}: marked should_abstain but lists relevant documents")
        if not question.should_abstain and not question.relevant:
            errors.append(f"{question.id}: answerable but has no relevant documents")

        for doc_id in question.relevant:
            if doc_id not in known:
                suggestion = _closest(doc_id, known)
                hint = f"  did you mean {suggestion}?" if suggestion else ""
                errors.append(f"{question.id}: unknown doc_id {doc_id!r}{hint}")
    return errors, warnings


def _closest(doc_id: str, known: set[str]) -> str | None:
    """Find the real id a mistyped one was probably reaching for.

    Ids carry a hash suffix that cannot be guessed by hand, so a label written from
    memory will have the right stem and the wrong suffix. Matching on the stem recovers
    it, and an unambiguous single match can be applied automatically.
    """
    stem = doc_id.rsplit("-", 1)[0]
    matches = [k for k in known if k.rsplit("-", 1)[0] == stem]
    if len(matches) == 1:
        return matches[0]
    prefix_matches = sorted(k for k in known if k.startswith(stem[: max(len(stem) - 8, 8)]))
    return prefix_matches[0] if len(prefix_matches) == 1 else None


# --- metrics ------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(relevant)


def success_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """1.0 if any relevant document is in the top k.

    This is the honest metric for a question with many valid answers. recall@k is
    structurally capped at k / len(relevant), so a question with twenty correct answers
    can never score above 0.25 at k=5 no matter how good retrieval is, and averaging that
    together with single-answer questions produces a number that means nothing. Report
    both: recall for how much was found, success for whether anything was.
    """
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    wanted = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Binary-relevance nDCG. Ideal ordering puts every relevant document first."""
    if not relevant:
        return 0.0
    wanted = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in wanted
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(wanted), k) + 1))
    return dcg / ideal if ideal else 0.0


# --- running ------------------------------------------------------------------


@dataclass
class QuestionResult:
    question: Question
    retrieved: list[str]
    best_cosine: float
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_6: float = 0.0
    recall_at_10: float = 0.0
    success_at_1: float = 0.0
    success_at_5: float = 0.0
    success_at_6: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0


#: Documents actually handed to the model. Metrics are reported here rather than at a
#: round number, because this is the only k whose value changes an answer: a document
#: ranked seventh is not in the prompt and might as well not have been retrieved.
#: Keep in step with the -k default in the cli.
SERVED_K = 6


def evaluate(retriever, questions: list[Question], k: int = 10) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for question in questions:
        outcome = retriever.retrieve(question.question, k=k, include_linked=False)
        retrieved = [r.document.doc_id for r in outcome.documents]
        result = QuestionResult(
            question=question,
            retrieved=retrieved,
            best_cosine=outcome.best_cosine,
        )
        if question.answerable:
            result.recall_at_1 = recall_at_k(retrieved, question.relevant, 1)
            result.recall_at_3 = recall_at_k(retrieved, question.relevant, 3)
            result.recall_at_5 = recall_at_k(retrieved, question.relevant, 5)
            result.recall_at_6 = recall_at_k(retrieved, question.relevant, SERVED_K)
            result.recall_at_10 = recall_at_k(retrieved, question.relevant, 10)
            result.success_at_1 = success_at_k(retrieved, question.relevant, 1)
            result.success_at_5 = success_at_k(retrieved, question.relevant, 5)
            result.success_at_6 = success_at_k(retrieved, question.relevant, SERVED_K)
            result.mrr = reciprocal_rank(retrieved, question.relevant)
            result.ndcg_at_10 = ndcg_at_k(retrieved, question.relevant, 10)
        results.append(result)
    return results


def aggregate(results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if r.question.answerable]
    if not answerable:
        return {}
    n = len(answerable)
    return {
        "n": n,
        "recall@1": sum(r.recall_at_1 for r in answerable) / n,
        "recall@3": sum(r.recall_at_3 for r in answerable) / n,
        "recall@5": sum(r.recall_at_5 for r in answerable) / n,
        "recall@6": sum(r.recall_at_6 for r in answerable) / n,
        "recall@10": sum(r.recall_at_10 for r in answerable) / n,
        "success@1": sum(r.success_at_1 for r in answerable) / n,
        "success@5": sum(r.success_at_5 for r in answerable) / n,
        "success@6": sum(r.success_at_6 for r in answerable) / n,
        "mrr": sum(r.mrr for r in answerable) / n,
        "ndcg@10": sum(r.ndcg_at_10 for r in answerable) / n,
    }


def by_category(results: list[QuestionResult]) -> dict[str, dict]:
    groups: dict[str, list[QuestionResult]] = {}
    for result in results:
        groups.setdefault(result.question.category, []).append(result)
    return {name: aggregate(group) for name, group in sorted(groups.items()) if aggregate(group)}


def abstention_separation(results: list[QuestionResult]) -> dict:
    """How well the confidence signal separates answerable from unanswerable questions.

    Reported as the two score distributions plus the threshold that best separates them,
    rather than as a single accuracy, because the useful question is not "how good is
    this threshold" but "is there a threshold at all".
    """
    answerable = [r.best_cosine for r in results if r.question.answerable]
    unanswerable = [r.best_cosine for r in results if not r.question.answerable]
    if not answerable or not unanswerable:
        return {}

    candidates = sorted({round(v, 3) for v in answerable + unanswerable})
    best = {"threshold": None, "accuracy": 0.0}
    for threshold in candidates:
        correct = sum(1 for v in answerable if v >= threshold)
        correct += sum(1 for v in unanswerable if v < threshold)
        accuracy = correct / (len(answerable) + len(unanswerable))
        if accuracy > best["accuracy"]:
            best = {"threshold": threshold, "accuracy": accuracy}

    return {
        "answerable_min": min(answerable),
        "answerable_median": sorted(answerable)[len(answerable) // 2],
        "unanswerable_max": max(unanswerable),
        "unanswerable_median": sorted(unanswerable)[len(unanswerable) // 2],
        "overlap": max(unanswerable) >= min(answerable),
        "best_threshold": best["threshold"],
        "best_accuracy": best["accuracy"],
    }
