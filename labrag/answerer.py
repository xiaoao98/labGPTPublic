"""Answer generation with two gates around the model call.

Before it, an abstention gate: if retrieval was weak, refuse without spending a token.
After it, citation validation: every [S#] the model emitted is checked against the sources
that were actually supplied, and an answer with no valid citation is withheld.

The second gate matters more than it looks. A model citing [S7] when six sources were
given has invented its provenance, and an answer with invented provenance is worse than no
answer because it reads as verified. Any claim this assistant makes about a protocol step
or a safety procedure has to be traceable to a document a person can open.

ABSTENTION, and why it is not just a threshold.

The evaluation set carries deliberately unanswerable questions. Measured against them on
the 101-question set, the signal works but is not clean:

    threshold   answered ok   refused ok   wrong refusal   wrong answer   accuracy
        0.55            86            3               0             12      88.1%
        0.62            83            9               3              6      91.1%
        0.70            66           12              20              3      77.2%

    answerable    n=86   min 0.578   median 0.734   max 0.879
    unanswerable  n=15   min 0.526   median 0.614   max 0.744

The distributions overlap: the worst answerable question scores 0.578 and the best
unanswerable one 0.744, so no threshold separates them. Every value trades wrong refusals
against wrong answers, and the table is the trade-off rather than a search for a right
answer.

0.62 is where accuracy peaks on this set, and two caveats belong with that number. It was
chosen by looking at the same questions it is scored on, so its accuracy on unseen
questions will be lower. And the gap to 0.61 is two questions out of 101, which is noise,
not a tuned optimum.

The errors are not symmetric in cost. A wrong refusal is visible and recoverable: the
reader asks a person. A wrong answer about a spill or a centrifuge speed is neither. That
argues for a higher threshold, but the table shows what that buys: going from 0.62 to 0.70
prevents three more wrong answers and causes seventeen more wrong refusals, and an
assistant that refuses a quarter of real questions stops being consulted at all, which is
its own safety problem. 0.62 is a compromise, not a solution, and the citation gate
downstream is the second line of defence for what this one lets through.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .prompts import ABSTENTION_TEMPLATE, build_messages

CITATION_RE = re.compile(r"\[S(\d{1,2})\]")
# A sentence together with any citation markers trailing it.
#
# This was a plain split on "(?<=[.!?])\s+", which is wrong in a way that only showed up
# the first time a real model generated an answer. The prompt asks for the citation after
# the sentence it supports, so the text reads "... soap and water. [S2]". Splitting on the
# punctuation cuts between "water." and "[S2]", handing the marker to the *following*
# sentence. Every cited answer therefore reported its first sentence as uncited and ended
# with a bare "[S2]" fragment counted as a sentence of its own. The count came out roughly
# right by cancellation while the attribution was shifted by one throughout.
#
# The lookahead for whitespace or end of string is what keeps "0.22 um" and "100,000 x g"
# from splitting mid-number: a period only ends a sentence if something blank follows it.
SENTENCE_RE = re.compile(r"\s*(\S.*?[.!?]+(?:[ \t]*\[S\d{1,2}\])*)(?=\s|$)", re.S)

# Where accuracy peaks on the evaluation set, at 91.1%. Selected on the same questions
# it is measured on, so treat it as an operating point rather than a validated number.
DEFAULT_ABSTAIN_COSINE = 0.62


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    reason: str = ""
    retrieved: list = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: int = 0
    best_cosine: float = 0.0
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def citations(self) -> list[str]:
        out = []
        for number in self.cited:
            if 1 <= number <= len(self.retrieved):
                document = self.retrieved[number - 1].document
                out.append(f"[S{number}] {document.title} ({document.source})")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "reason": self.reason,
            "cited": self.cited,
            "invalid_citations": self.invalid_citations,
            "uncited_sentences": self.uncited_sentences,
            "best_cosine": round(self.best_cosine, 4),
            "citations": self.citations,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
        }


# --- gates --------------------------------------------------------------------


def should_abstain(result, threshold: float = DEFAULT_ABSTAIN_COSINE) -> tuple[bool, str]:
    """Decide, before spending a token, whether retrieval was good enough to answer.

    The test is on the cosine, never on the fused RRF score. RRF encodes rank, not
    similarity, so the top hit of a search that found nothing relevant still scores
    highly; thresholding it would abstain almost never.
    """
    if not result.documents:
        return True, "retrieval returned nothing"

    best = result.best_cosine
    if best < threshold:
        return True, (
            f"the closest match scored {best:.3f}, below the {threshold:.2f} "
            f"confidence threshold"
        )
    return False, ""


def validate_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Split emitted [S#] markers into those pointing at a real source and those not."""
    valid, invalid = [], []
    for raw in CITATION_RE.findall(text):
        number = int(raw)
        target = valid if 1 <= number <= n_sources else invalid
        if number not in target:
            target.append(number)
    return sorted(valid), sorted(invalid)


def split_sentences(text: str) -> list[str]:
    """Sentences, each keeping the citation markers that follow it.

    Text after the last sentence-ending punctuation is returned as a final sentence, so a
    model that stops mid-thought still has its trailing clause examined rather than
    silently dropped.
    """
    sentences: list[str] = []
    end = 0
    for match in SENTENCE_RE.finditer(text):
        sentences.append(match.group(1))
        end = match.end()
    remainder = text[end:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def count_uncited_sentences(text: str) -> int:
    """Factual-looking sentences with no citation. A quality signal, not a gate."""
    count = 0
    for sentence in split_sentences(text):
        stripped = sentence.strip()
        if len(stripped.split()) < 5 or stripped.endswith("?"):
            continue
        if not CITATION_RE.search(stripped):
            count += 1
    return count


# --- model --------------------------------------------------------------------


class LLMError(RuntimeError):
    pass


class ChatClient:
    """Any OpenAI-compatible /v1/chat/completions endpoint.

    One client covers the OpenAI API, Azure OpenAI, a self-hosted vLLM, Ollama and an
    internal LiteLLM proxy. For institutional data, point base_url at whichever of those
    has been approved and nothing leaves that boundary.

    TWO REQUEST SHAPES.

    The reasoning models are not drop-in compatible with the chat models that came before
    them, and the differences are rejections rather than warnings. Measured against gpt-5
    rather than taken from documentation:

        max_tokens        400  "is not supported with this model. Use
                                'max_completion_tokens' instead"
        temperature=0.0   400  "does not support 0.0 with this model. Only the default
                                (1) value is supported"

    So one payload cannot serve both, and the model name alone is not a reliable test: a
    proxy may rename models and the family keeps growing. The name check picks the likely
    shape, and a 400 naming one of these parameters corrects it and retries once. The
    client then remembers, so a batch run pays that cost at most twice.

    TEMPERATURE IS UNAVAILABLE, WHICH MATTERS FOR EVALUATION.

    This client used to send temperature=0.0 for reproducibility. A reasoning model
    refuses anything but 1, so identical inputs can yield different answers and a rerun
    will not reproduce exactly. `seed` is accepted and passed when given, but it is
    documented as best effort, not a guarantee. Any comparison between two systems has to
    treat the generated half as noisy and say so.

    THE EMPTY ANSWER.

    Reasoning tokens are charged against the same budget as the answer and are spent
    first. Asking gpt-5 to "Reply with exactly: OK" with a budget of 16 returns HTTP 200,
    finish_reason "length", and an empty string: all sixteen went to reasoning and none
    were left to write the reply with. Nothing about that response is an error, so a batch
    run would record a hundred blank answers and they would read as the model declining to
    answer. That case is raised here instead. A crash during an evaluation is cheap; a
    silently empty result set is not.

    reasoning_effort controls the spend. On this workload answers are quoted from supplied
    sources rather than derived, so the reasoning budget buys little: "minimal", "low" and
    "medium" all returned the same answer with zero reasoning tokens while "high" spent 64.
    It defaults to None, meaning the endpoint's own default.
    """

    # Parameters whose rejection says the shape is wrong rather than the value.
    _SHAPE_PARAMS = frozenset({"max_tokens", "max_completion_tokens", "temperature"})

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 800,
                 reasoning_effort: str | None = None, seed: int | None = None):
        base = (base_url or os.environ.get("LABGPT_LLM_BASE_URL")
                or "https://api.openai.com/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self._key = (os.environ.get("LABGPT_LLM_API_KEY")
                     or os.environ.get("OPENAI_API_KEY", ""))
        self._reasoning = self._looks_like_reasoning_model(model)

    @staticmethod
    def _looks_like_reasoning_model(model: str) -> bool:
        """First guess at the request shape. A wrong guess is corrected on the first 400."""
        name = model.lower().rsplit("/", 1)[-1]
        return name.startswith(("gpt-5", "o1", "o3", "o4"))

    def _payload(self, messages) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self._reasoning:
            body["max_completion_tokens"] = self.max_tokens
            # temperature is omitted rather than sent as 1, so that if the shape flips
            # back the caller's own value is what gets sent.
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        else:
            body["max_tokens"] = self.max_tokens
            body["temperature"] = self.temperature
        if self.seed is not None:
            body["seed"] = self.seed
        return body

    def _headers(self) -> dict:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}"}

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))

    @classmethod
    def _rejected_shape_param(cls, code: int, detail: str):
        """The parameter a 400 blamed, when it is one that distinguishes the two shapes."""
        if code != 400:
            return None
        try:
            param = json.loads(detail).get("error", {}).get("param")
        except (ValueError, AttributeError):
            param = None
        if param in cls._SHAPE_PARAMS:
            return param
        # A gateway in front of the model may rewrap the error and drop the param field,
        # leaving only prose. The quoted name is still in it, and the quotes are what keep
        # 'max_tokens' from matching inside 'max_completion_tokens'.
        for name in ("max_completion_tokens", "max_tokens", "temperature"):
            if f"'{name}'" in detail and ("not supported" in detail
                                          or "Unsupported" in detail):
                return name
        return None

    def complete(self, messages) -> tuple[str, dict]:
        try:
            body = self._post(self._payload(messages))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            param = self._rejected_shape_param(exc.code, detail)
            if param is None:
                raise LLMError(f"chat endpoint returned {exc.code}: {detail[:300]}") from exc
            # The guess was wrong. Flip it, remember, and try once more.
            self._reasoning = not self._reasoning
            try:
                body = self._post(self._payload(messages))
            except urllib.error.HTTPError as retry:
                retry_detail = retry.read().decode("utf-8", errors="replace")[:300]
                raise LLMError(
                    f"chat endpoint rejected {param!r}, and the other request shape also "
                    f"failed with {retry.code}: {retry_detail}"
                ) from retry
        except urllib.error.URLError as exc:
            raise LLMError(f"could not reach {self.url}: {exc.reason}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"chat endpoint returned no choices: {str(body)[:200]}")

        choice = choices[0]
        content = (choice.get("message", {}).get("content") or "").strip()
        if not content:
            spent = (body.get("usage", {})
                         .get("completion_tokens_details", {})
                         .get("reasoning_tokens"))
            if choice.get("finish_reason") == "length":
                raise LLMError(
                    f"the model returned an empty answer: the {self.max_tokens}-token "
                    f"budget ran out before any reply was written"
                    + (f", {spent} of it spent on reasoning" if spent else "")
                    + ". Raise max_tokens, or lower reasoning_effort."
                )
            raise LLMError("the model returned an empty answer with finish_reason "
                           f"{choice.get('finish_reason')!r}")
        return content, body.get("usage", {})


class Answerer:
    def __init__(self, retriever, client=None, k: int = 6,
                 abstain_cosine: float = DEFAULT_ABSTAIN_COSINE):
        self.retriever = retriever
        self.client = client
        self.k = k
        self.abstain_cosine = abstain_cosine

    def answer(self, question: str, documents=None) -> Answer:
        """Answer from retrieval, or from documents the caller supplies.

        Passing documents is the oracle path. It skips retrieval and skips the confidence
        gate, because the gate is a judgement about retrieval and retrieval is the thing
        being held fixed. The citation gate below still applies: it judges the answer, not
        the search, and an answer with no usable provenance is withheld either way.
        """
        started = time.perf_counter()
        if documents is None:
            result = self.retriever.retrieve(question, k=self.k)
            abstain, reason = should_abstain(result, self.abstain_cosine)
            if abstain:
                return Answer(
                    question=question,
                    text=ABSTENTION_TEMPLATE.format(reason=f"Reason: {reason}."),
                    abstained=True,
                    reason=reason,
                    retrieved=result.documents,
                    best_cosine=result.best_cosine,
                    latency_ms=_elapsed(started),
                )
            served, cosine = result.documents, result.best_cosine
        else:
            served, cosine = list(documents), 0.0

        if self.client is None:
            raise LLMError(
                "retrieval passed the confidence gate but no chat client is configured.\n"
                "  set LABGPT_LLM_BASE_URL and LABGPT_LLM_API_KEY, or pass --dry-run "
                "to see the assembled prompt without calling a model."
            )

        text, usage = self.client.complete(build_messages(question, served))
        valid, invalid = validate_citations(text, len(served))

        if not valid:
            # An answer with no usable provenance is indistinguishable from a
            # fabrication in this domain, so it is not served.
            return Answer(
                question=question,
                text=ABSTENTION_TEMPLATE.format(
                    reason="A draft answer was produced but carried no valid citation "
                           "back to an indexed document, so it was withheld."),
                abstained=True,
                reason="no valid citation in the generated answer",
                retrieved=served,
                invalid_citations=invalid,
                best_cosine=cosine,
                latency_ms=_elapsed(started),
                usage=usage,
            )

        return Answer(
            question=question,
            text=text,
            abstained=False,
            retrieved=served,
            cited=valid,
            invalid_citations=invalid,
            uncited_sentences=count_uncited_sentences(text),
            best_cosine=cosine,
            latency_ms=_elapsed(started),
            usage=usage,
        )


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
