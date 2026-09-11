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
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

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


def count_uncited_sentences(text: str) -> int:
    """Factual-looking sentences with no citation. A quality signal, not a gate."""
    count = 0
    for sentence in SENTENCE_SPLIT_RE.split(text):
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

    One client covers Azure OpenAI, a self-hosted vLLM, Ollama and an internal LiteLLM
    proxy. For institutional data, point base_url at whichever of those has been approved
    and nothing leaves that boundary.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 800):
        base = (base_url or os.environ.get("LABGPT_LLM_BASE_URL")
                or "https://api.openai.com/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._key = (os.environ.get("LABGPT_LLM_API_KEY")
                     or os.environ.get("OPENAI_API_KEY", ""))

    def complete(self, messages) -> tuple[str, dict]:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMError(f"chat endpoint returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"could not reach {self.url}: {exc.reason}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"chat endpoint returned no choices: {str(body)[:200]}")
        return choices[0]["message"]["content"].strip(), body.get("usage", {})


class TransformersChatClient:
    """A locally loaded transformers model, behind the same interface as ChatClient.

    demo.py runs an open-weights model on the machine itself, which is the whole reason
    the corpus can stay institutional. Answerer only needs `.complete(messages)`, so the
    local and hosted paths are interchangeable and nothing downstream knows which is in
    use.

    The tokenizer and model are passed in rather than loaded here, because loading a 32B
    model takes minutes and demo.py already holds one for the classifier call.
    """

    def __init__(self, tokenizer, model, max_new_tokens: int = 1024,
                 thinking: bool = True, streamer=None):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.streamer = streamer

    def complete(self, messages) -> tuple[str, dict]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=False, streamer=self.streamer,
        )
        new_tokens = generated[0][len(inputs.input_ids[0]):]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # A thinking model emits its reasoning before </think>; only what follows is the
        # answer, and citation validation must not see the reasoning.
        if "</think>" in answer:
            answer = answer.split("</think>")[-1]

        usage = {
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "completion_tokens": int(len(new_tokens)),
        }
        return answer.strip(), usage


# --- answering ----------------------------------------------------------------


class Answerer:
    def __init__(self, retriever, client=None, k: int = 6,
                 abstain_cosine: float = DEFAULT_ABSTAIN_COSINE):
        self.retriever = retriever
        self.client = client
        self.k = k
        self.abstain_cosine = abstain_cosine

    def answer(self, question: str) -> Answer:
        started = time.perf_counter()
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

        if self.client is None:
            raise LLMError(
                "retrieval passed the confidence gate but no chat client is configured.\n"
                "  set LABGPT_LLM_BASE_URL and LABGPT_LLM_API_KEY, or pass --dry-run "
                "to see the assembled prompt without calling a model."
            )

        text, usage = self.client.complete(build_messages(question, result.documents))
        valid, invalid = validate_citations(text, len(result.documents))

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
                retrieved=result.documents,
                invalid_citations=invalid,
                best_cosine=result.best_cosine,
                latency_ms=_elapsed(started),
                usage=usage,
            )

        return Answer(
            question=question,
            text=text,
            abstained=False,
            retrieved=result.documents,
            cited=valid,
            invalid_citations=invalid,
            uncited_sentences=count_uncited_sentences(text),
            best_cosine=result.best_cosine,
            latency_ms=_elapsed(started),
            usage=usage,
        )


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
