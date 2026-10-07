"""The answering pipeline, shared by the terminal chat loop and the HTTP service.

This was demo.py's, and demo.py was the only caller until a service needed the same
behaviour. Two front ends printing and serialising the same result is fine; two front ends
each deciding for themselves when to abstain, whether to attach the directory, and what
counts as a valid citation is how the terminal and the web quietly stop agreeing. So the
decisions live here, in answer_structured, which returns a dict and prints nothing.

demo.py renders that dict to a terminal. serve.py returns it as JSON.
"""

from __future__ import annotations

import os

import labgpt_config as cfg
import labgpt_metrics as metrics

from .answerer import (
    DEFAULT_ABSTAIN_COSINE, TransformersChatClient, chat_client_from_env,
    count_uncited_sentences, should_abstain, validate_citations,
)
from .prompts import (
    ABSTENTION_TEMPLATE, build_messages, build_messages_without_sources,
)
from .rerank import DEFAULT_RERANK_MODEL, Reranker, rerank_is_cheap

model_name = cfg.MODEL_NAME

# Generation caps. Every one of these was 32768 or 65536, for answers measured in words.
MAX_NEW_TOKENS_YESNO = 8      # "yes" / "no"
MAX_NEW_TOKENS_ANSWER = 1024  # the user-facing answer

# How many documents go into the prompt, and how many of them may be people. The
# directory is retrieved separately so that a protocol question cannot crowd out the
# "who can help" behaviour the original prompt asked for, and a people question cannot
# fill every slot with colleagues.
TOP_K = 6
TOP_K_MEMBERS = 2

# A retrieved person is only appended above this cosine. Filtering to the directory means
# the search always returns somebody, so without a floor a question about freezing medium
# gets two arbitrary postdocs attached and the model, told to name who can help, names
# them.
#
# Measured on the top members retrieved, before and after bios were chunked by sentence:
#
#                                                      whole bio   chunked
#   "who runs the flow cytometry platforms"  Shreyasee     0.566     0.662  right
#   "who should I ask about ordering reagents" Michelle    0.516     0.543  right
#   "  (the same query)"                       Aman        0.557     0.564  wrong
#   "what freezing medium is used for BJ cells" anyone     0.538     0.602  wrong
#
# Chunking lifted every member score, signal and noise alike, so the floor had to rise
# with it; at 0.55 it now admits everybody. At 0.62 it keeps Shreyasee, drops the
# freezing-medium case, and on the reagents question admits nobody rather than the wrong
# person, which is the better of the two outcomes available.
#
# It is still not a relevance test. On that reagents question the correct person scores
# below an incorrect one on cosine, so the floor cannot separate them; only the fused
# ranking gets Michelle to the top, and that is because BM25 matches "reagents" in her
# bio. This rejects the obviously unrelated and does no more.
MEMBER_FLOOR = DEFAULT_ABSTAIN_COSINE

INDEX_DIR = os.environ.get("LABGPT_INDEX_DIR", str(cfg.REPO_ROOT / ".index"))

# chat_client_from_env prefers Azure whenever its variables are set, so on a machine
# configured for both the approved endpoint wins without anyone remembering a flag.
BACKEND = os.environ.get("LABGPT_BACKEND", "api")

# What happens when retrieval scores below the abstention threshold. "answer" puts the
# question to the model with no sources attached and labels the result as general
# knowledge; "refuse" stops at the gate, which is what this pipeline did before.
#
# The refusal exists because an unsupported answer about a centrifuge speed or a spill is
# worse than no answer: it reads as verified. Answering anyway gives that back some of the
# ground it was standing on, so the fallback is built to keep the boundary visible rather
# than to blur it. No sources are supplied, so no [S#] can be emitted and citation
# validation is skipped; the prompt is a different one that opens by saying the lab's
# documentation does not cover this, refuses to state lab specifics it does not have, and
# sends anything urgent to a person. The transcript marks the whole answer as unsourced.
#
# The second gate is untouched. An answer that was given sources and cited none of them is
# still withheld: that is a model ignoring its evidence, not a gap in the corpus.
ABSTAIN_MODE = os.environ.get("LABGPT_ABSTAIN_MODE", "answer")

# Whether a cross-encoder re-sorts the fused shortlist before the top k is taken. Off
# unless asked for, matching the CLI: the pass buys roughly three end-to-end questions in
# a hundred at n=100, which is inside the interval, and costs 13.1 seconds a query on a
# CPU against 6.2 for generation. It also downloads a second model, BAAI/bge-reranker-base,
# which a network that blocks huggingface.co will not allow.
#
#   off   (default)  no reranking
#   on               always, whatever the hardware
#   auto             on where a GPU makes it nearly free, off otherwise
#
# demo.py ran without a reranker at all while the CLI reranked by default, so the same
# question could be answered from a differently ordered shortlist depending on which one
# asked it. Both are off by default now, and both say so at startup.
RERANK_MODE = os.environ.get("LABGPT_RERANK", "off")
if RERANK_MODE not in ("auto", "on", "off"):
    raise SystemExit(f"LABGPT_RERANK must be auto, on or off, not {RERANK_MODE!r}")

# How many fused candidates the reranker scores. The CLI's default, kept in step.
RERANK_CANDIDATES = 40


def build_reranker():
    """The reranker RERANK_MODE asks for, or none."""
    wanted = RERANK_MODE == "on" or (RERANK_MODE == "auto" and rerank_is_cheap())
    if not wanted:
        return None
    reranker = Reranker(os.environ.get("LABGPT_RERANK_MODEL") or DEFAULT_RERANK_MODEL)
    # Scored once here to force the load. The Reranker loads on first use, which would
    # otherwise be inside the first question, where a hub that cannot be reached takes the
    # chat loop down with it instead of costing one degraded startup.
    reranker.score("warm up", ["warm up"])
    return reranker
if ABSTAIN_MODE not in ("answer", "refuse"):
    raise SystemExit(f"LABGPT_ABSTAIN_MODE must be answer or refuse, not {ABSTAIN_MODE!r}")

class LocalBackend:
    """cfg.MODEL_NAME, loaded once and reused for every call in the session."""

    streams = True
    web_search = True
    answer_tokens = MAX_NEW_TOKENS_ANSWER

    def __init__(self):
        # Imported here, not at module scope, so the api path runs on a machine with no
        # transformers and no torch installed at all.
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        self._TextStreamer = TextStreamer
        rprint(f"[dim]loading {model_name} ...[/dim]")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def describe(self) -> str:
        return f"local {model_name}"

    def client(self, max_tokens: int, stream: bool = False, thinking: bool = False):
        streamer = None
        if stream:
            streamer = self._TextStreamer(self.tokenizer, skip_prompt=True,
                                          skip_special_tokens=True)
        return TransformersChatClient(self.tokenizer, self.model,
                                      max_new_tokens=max_tokens, thinking=thinking,
                                      streamer=streamer)


class ApiBackend:
    """A hosted OpenAI-compatible endpoint, Azure OpenAI included.

    A fresh client per call, because the token budget differs per call and the clients are
    cheap to construct; each one keeps only its configuration.

    The budget has a floor here that the local path does not need. A reasoning model
    spends its reasoning tokens out of the same allowance and spends them first, so the
    8-token budget that is ample for a local model writing "yes" returns an empty string
    from gpt-5 with nothing left to answer with.
    """

    streams = False
    # No web leg on this path. The web branch exists to reach what the corpus does not
    # cover, and it costs a classifier call on every single question plus, when it fires,
    # a browser driver and several page fetches. On the endpoint path that trade is not
    # taken: the endpoint is asked about the retrieved corpus and nothing else, which also
    # means a question routed to the web cannot quietly send its text out through a second
    # channel that was never part of the approved one.
    web_search = False
    MIN_TOKENS = 64

    # The answer budget is larger here than on the local path, and this is the reason.
    # A reasoning deployment spends its reasoning tokens out of the same allowance and
    # spends them first, so the budget is not an answer length: at the endpoint's own
    # default effort, gpt-5 spent all 1024 tokens of the local path's budget on reasoning
    # and returned an empty answer.
    #
    # 2500 is the CLI's measured figure rather than a guess: over 101 real questions at
    # low effort, reasoning took a median of 192 tokens, 448 at the 90th percentile and
    # 704 at the worst successful call. Kept in step with cli.DEFAULT_MAX_TOKENS, and
    # overridden by the same LABGPT_MAX_TOKENS.
    #
    # The effort default is the other half. Answers on this workload are quoted from the
    # supplied sources rather than derived, so reasoning buys little: "minimal", "low" and
    # "medium" were measured returning the same answer with zero reasoning tokens spent,
    # where the endpoint default spends over a thousand. Low unless LABGPT_REASONING_EFFORT
    # says otherwise.
    DEFAULT_EFFORT = "low"

    def __init__(self):
        self.reasoning_effort = (os.environ.get("LABGPT_REASONING_EFFORT")
                                 or self.DEFAULT_EFFORT)
        self.answer_tokens = int(os.environ.get("LABGPT_MAX_TOKENS") or 2500)
        if not _endpoint_configured():
            # Reported here rather than as a 401 from api.openai.com on the first question,
            # which is what an unconfigured OpenAI-shaped client would produce.
            raise SystemExit(
                "no chat endpoint is configured, and the api backend is the default.\n"
                "  For the institutional Azure gateway, set AZURE_OPENAI_GATEWAY, "
                "AZURE_OPENAI_TEAM_ID,\n"
                "  AZURE_OPENAI_MODEL_ID, AZURE_OPENAI_API_VERSION and "
                "APIM_OPENAI_SUBSCRIPTION_KEY;\n"
                "  check them with python -m labrag.cli selftest.\n"
                "  For another endpoint, set LABGPT_LLM_BASE_URL and LABGPT_LLM_API_KEY.\n"
                f"  To answer on the local model instead: "
                f"LABGPT_BACKEND=local python demo.py"
            )
        # Constructed once here, purely so that a misconfigured endpoint fails now rather
        # than after the index has loaded and the first question has been typed.
        probe = chat_client_from_env(max_tokens=MAX_NEW_TOKENS_ANSWER)
        self._kind = type(probe).__name__
        self._model = probe.model

    def describe(self) -> str:
        return (f"{self._kind} {self._model}, {self.answer_tokens} tokens, "
                f"reasoning_effort={self.reasoning_effort}")

    def client(self, max_tokens: int, stream: bool = False, thinking: bool = False):
        # stream and thinking are accepted to match LocalBackend and ignored: an endpoint
        # is asked for a complete response, and a hosted model's reasoning is controlled
        # by reasoning_effort rather than by a chat-template flag.
        return chat_client_from_env(max_tokens=max(max_tokens, self.MIN_TOKENS),
                                    reasoning_effort=self.reasoning_effort)


def _endpoint_configured() -> bool:
    """Whether anything at all names an endpoint to call."""
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT")
                or (os.environ.get("AZURE_OPENAI_GATEWAY")
                    and os.environ.get("AZURE_OPENAI_TEAM_ID"))
                or os.environ.get("LABGPT_LLM_BASE_URL")
                or os.environ.get("LABGPT_LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY"))


def build_backend():
    """The backend BACKEND asks for. The endpoint unless the local model was asked for."""
    if BACKEND == "local":
        return LocalBackend()
    if BACKEND != "api":
        raise SystemExit(f"LABGPT_BACKEND must be api or local, not {BACKEND!r}")
    return ApiBackend()

ifSearchContent = (
    "Search is NEEDED (true) for: 1. Real-time information: Weather, stock prices, "
    "traffic, sports scores. 2. Recent events: Anything that happened after your "
    "knowledge cutoff. 3. Specific people or entities outside this laboratory. "
    "4. Niche or local information: Store hours, specific product recommendations, "
    "local regulations.\n"
    "Search is NOT needed for anything about this lab's protocols, reagents, safety "
    "guidance, members or publications, which are answered from local documents.\n"
    "Does this query need a web search (answer with one word, yes or no): "
)


# --- web search ---------------------------------------------------------------


def _needs_web_search(query, backend) -> bool:
    """The one classifier call left. It gates a real external call, so it earns its cost.

    The other two are gone. Retrieval decides what the query is about by scoring the
    corpus, which is both cheaper and measurable, where the classifiers were neither.

    Only reached on a backend whose web_search is set, which today means the local one.
    """
    answer = _short_answer(
        ifSearchContent + query, backend,
        max_new_tokens=MAX_NEW_TOKENS_YESNO, label="classify:web",
    )
    return "yes" in answer.lower()


def _summarise_web(query, backend):
    # Imported here rather than at module scope so the assistant starts without the
    # search extra installed. Web search is optional; selenium and a browser driver are
    # a heavy thing to require of a deployment that only wants to answer from the corpus.
    try:
        from search.duckducksearch import get_useful_link
        from search.getDynamicPage import fetch_all_webpage_content
    except ImportError as exc:
        rprint(f"[yellow]web search unavailable ({exc}); install with "
               f"uv sync --extra search[/yellow]")
        return None

    try:
        links, snippets = get_useful_link(query)
        pages = fetch_all_webpage_content(links)
    except Exception as exc:
        rprint(f"[red]web search failed: {exc}[/red]")
        return None
    if not pages:
        return None

    prompt = (
        "Summarise the following web results to answer this question. Be precise and "
        f"keep any useful link.\n\nQuestion: {query}\n"
    )
    for index, item in enumerate(pages):
        snippet = snippets[index] if index < len(snippets) else ""
        prompt += (f"\n--- result {index + 1} ---\nURL: {item['url']}\n"
                   f"Snippet: {snippet}\nContent: {item['content']}\n")

    return _short_answer(prompt, backend, max_new_tokens=512,
                         label="web_summary", thinking=False)


def _short_answer(prompt, backend, max_new_tokens, label, thinking=False) -> str:
    """One generation, instrumented.

    Deterministic on the local backend, where sampling is off, so routing is reproducible.
    Not on a hosted one: a reasoning model refuses any temperature but its default, so the
    same question can take different paths on different runs and a comparison across runs
    has to treat the routing as noisy.
    """
    client = backend.client(max_new_tokens, thinking=thinking)
    try:
        with metrics.timer() as elapsed:
            text, usage = client.complete([{"role": "user", "content": prompt}])
    except Exception as exc:
        print(f"\nError during model generation: {exc}")
        return ""
    metrics.record(
        label,
        prompt_tokens=usage.get("prompt_tokens", 0),
        generated_tokens=usage.get("completion_tokens", 0),
        elapsed_s=elapsed[0],
        cap=max_new_tokens,
    )
    return text


# --- answering ----------------------------------------------------------------


def answer_structured(query: str, retriever, backend) -> dict:
    """Answer one question. Returns a dict and prints nothing.

    `mode` is what a caller should key its presentation off, because the four outcomes are
    not variations of one answer, they are different things to show a person:

        sourced     grounded in the index, carries citations that were checked
        unsourced   the index had nothing; general knowledge, explicitly not lab practice
        refused     the index had nothing and the fallback is off
        withheld    a draft existed and cited nothing valid, so it is not shown

    The distinction that matters most is sourced against unsourced. Rendering them alike
    is how a general answer about centrifuge speeds gets read as the lab's own protocol.
    """
    import time
    started = time.perf_counter()

    result = retriever.retrieve(query, k=TOP_K)
    # Filtered to the directory, so this always returns somebody: it hands back the two
    # least-bad people whether or not either is relevant. Hence the floor below.
    members = retriever.retrieve(query, k=TOP_K_MEMBERS, doc_types=["member"],
                                 include_linked=False)

    abstain, reason = should_abstain(result, DEFAULT_ABSTAIN_COSINE)
    if abstain:
        if ABSTAIN_MODE == "refuse":
            return {
                "mode": "refused",
                "question": query,
                "answer": ABSTENTION_TEMPLATE.format(reason=f"Reason: {reason}."),
                "reason": reason,
                "sources": [], "cited": [], "invalid_citations": [],
                "uncited_sentences": 0,
                "best_cosine": round(result.best_cosine, 4),
                "latency_ms": _elapsed(started), "usage": {},
            }
        return _answer_without_sources(query, backend, reason, result, started)

    sources = list(result.documents)
    web_summary = None
    if backend.web_search and _needs_web_search(query, backend):
        web_summary = _summarise_web(query, backend)

    # The directory is appended rather than competing for the main slots. Only people who
    # actually match: a weak match is worse than none, because the prompt asks the model to
    # name someone and it will name whoever is here. The floor is on the cosine, not on the
    # fused RRF score, which encodes rank and would never reject anybody.
    relevant = {
        hit.chunk.doc_id for hit in members.chunk_hits if hit.confidence >= MEMBER_FLOOR
    }
    seen = {item.document.doc_id for item in sources}
    for item in members.documents:
        if item.document.doc_id in relevant and item.document.doc_id not in seen:
            sources.append(item)

    messages = build_messages(query, sources)
    if web_summary:
        messages[-1]["content"] += (
            f"\n\nAdditional context from a web search, which is NOT one of the "
            f"numbered sources and must not be cited as one:\n{web_summary}"
        )

    client = backend.client(backend.answer_tokens, stream=backend.streams)
    with metrics.timer() as elapsed:
        text, usage = client.complete(messages)
    metrics.record(
        "answer",
        prompt_tokens=usage.get("prompt_tokens", 0),
        generated_tokens=usage.get("completion_tokens", 0),
        elapsed_s=elapsed[0],
        sources=len(sources),
        best_cosine=round(result.best_cosine, 4),
    )

    valid, invalid = validate_citations(text, len(sources))
    listed = _list_sources(sources)
    if not valid:
        # An answer with no usable provenance is indistinguishable from a fabrication in
        # this domain, so it is not served.
        return {
            "mode": "withheld",
            "question": query,
            "answer": ABSTENTION_TEMPLATE.format(
                reason="Reason: the generated answer cited no valid source."),
            "reason": "no valid citation in the generated answer",
            "draft": text,
            "sources": listed, "cited": [], "invalid_citations": invalid,
            "uncited_sentences": 0,
            "best_cosine": round(result.best_cosine, 4),
            "latency_ms": _elapsed(started), "usage": usage,
        }

    return {
        "mode": "sourced",
        "question": query,
        "answer": text,
        "reason": "",
        "sources": listed,
        "cited": valid,
        "invalid_citations": invalid,
        "uncited_sentences": count_uncited_sentences(text),
        "best_cosine": round(result.best_cosine, 4),
        "latency_ms": _elapsed(started),
        "usage": usage,
        "web_context": bool(web_summary),
    }


def _answer_without_sources(query, backend, reason, result, started) -> dict:
    """The gate failed and the fallback is on: answer from general knowledge, labelled.

    Nothing here is checked against the corpus, because no corpus content is in the
    prompt. The mode and the prompt's own first rule are what keep it from being read as
    the lab's documented practice, so a caller that renders this like a sourced answer has
    removed the only safeguard there is.
    """
    client = backend.client(backend.answer_tokens, stream=backend.streams)
    with metrics.timer() as elapsed:
        text, usage = client.complete(build_messages_without_sources(query))
    metrics.record(
        "answer:unsourced",
        prompt_tokens=usage.get("prompt_tokens", 0),
        generated_tokens=usage.get("completion_tokens", 0),
        elapsed_s=elapsed[0], sources=0, best_cosine=0.0,
    )
    # Not validation: nothing was supplied to cite. A marker here was invented.
    _, invented = validate_citations(text, 0)
    return {
        "mode": "unsourced",
        "question": query,
        "answer": text,
        "reason": reason,
        "sources": [], "cited": [], "invalid_citations": invented,
        "uncited_sentences": 0,
        "best_cosine": round(result.best_cosine, 4),
        "latency_ms": _elapsed(started), "usage": usage,
    }


def _list_sources(sources) -> list[dict]:
    """The numbered sources, as [S#] refers to them."""
    out = []
    for number, item in enumerate(sources, start=1):
        document = item.document
        out.append({
            "n": number,
            "doc_type": document.doc_type,
            "title": document.title,
            "source": document.source,
            "linked": item.is_linked,
        })
    return out


def _elapsed(started: float) -> int:
    import time
    return int((time.perf_counter() - started) * 1000)
