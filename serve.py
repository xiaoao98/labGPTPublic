"""HTTP front end: the same pipeline demo.py runs, behind one endpoint and one page.

Everything that decides an answer is in labrag/chat.py. This file adds what a shared
service needs and a terminal does not: a token on every request, a log of what was asked
and answered, and a health check that says what the process actually loaded.

ONE WORKER, ON PURPOSE. The index and the embedding model are loaded once at startup and
held in memory. Retrieval is milliseconds of numpy; the wait in a request is the endpoint
call, which is I/O, so a single process with a thread pool serves a lab comfortably. More
workers would multiply the loaded index by the worker count for no throughput.

THE LOG CONTAINS THE CORPUS. Every question and every answer is written to LABGPT_LOG_PATH,
which is what makes it useful for seeing what people ask and which questions get refused,
and which also means it quotes institutional documents back verbatim and records who
wanted to know what. It belongs on the server, readable by the account that runs this and
nobody else, and it does not belong in a backup that leaves the building.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from labrag.chat import (
    ABSTAIN_MODE, INDEX_DIR, RERANK_CANDIDATES, RERANK_MODE, TOP_K,
    answer_structured, build_backend, build_reranker,
)
from labrag.embeddings import Embedder, resolve_model
from labrag.retriever import Retriever
from labrag.store import VectorStore
from labrag import voice

WEB_DIR = Path(__file__).resolve().parent / "web"
LOG_PATH = os.environ.get("LABGPT_LOG_PATH")
TOKEN = os.environ.get("LABGPT_API_TOKEN", "")
MAX_QUESTION_CHARS = 2000

state: dict = {}
_log_lock = threading.Lock()


def _load() -> None:
    """Index, embedder, reranker and chat client, once.

    The prints flush because stdout is block-buffered when it is a pipe, which is what it
    is under a container runtime or systemd: without it these lines sit in the buffer and
    a started service looks like a silent one in the logs.
    """
    store = VectorStore.load(INDEX_DIR)
    print(f"index: {len(store.documents)} documents, {len(store.chunks)} chunks",
          flush=True)

    # Loaded here rather than on first use. sentence-transformers is lazy, which puts the
    # load inside whichever request arrives first: 14.3 s against 7.1 s for every one
    # after it, paid by one colleague who then reports that the service is slow. The port
    # is not accepting connections until this returns, so a watchdog pointed at /health
    # should allow for a startup that takes a few seconds.
    embedder = Embedder(resolve_model(store.manifest.get("embedding_model")))
    warm_started = time.time()
    embedder.warm()
    print(f"embedder: {embedder.model_name} warmed in "
          f"{time.time() - warm_started:.1f}s", flush=True)

    try:
        reranker = build_reranker()
    except Exception as exc:
        print(f"reranker unavailable ({exc}); continuing without it", flush=True)
        reranker = None
    print(f"rerank: {reranker.model_name if reranker else 'off'}", flush=True)

    state["store"] = store
    state["retriever"] = Retriever(store, embedder=embedder, mode="hybrid", final_k=TOP_K,
                                   reranker=reranker, rerank_candidates=RERANK_CANDIDATES)
    state["backend"] = build_backend()
    state["reranker"] = reranker.model_name if reranker else None
    state["started"] = time.time()
    print(f"model: {state['backend'].describe()}", flush=True)


app = FastAPI(title="LabGPT", docs_url=None, redoc_url=None)


@app.on_event("startup")
def startup() -> None:
    if not TOKEN:
        # Refused rather than defaulted. A service that answers from an institutional
        # corpus should not come up open to the network because a variable was unset.
        raise SystemExit(
            "LABGPT_API_TOKEN is not set, and this service will not start without one.\n"
            "  Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(24))\"\n"
            "  and pass it in the environment file, not on the command line."
        )
    _load()


def require_token(x_labgpt_token: str = Header(default="")) -> None:
    """Constant-time check, so the comparison cannot be timed character by character."""
    if not secrets.compare_digest(x_labgpt_token, TOKEN):
        raise HTTPException(status_code=401, detail="bad or missing token")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


@app.get("/health")
def health() -> dict:
    """Unauthenticated on purpose: it is what a restart policy and a human both poll.

    It reports what was loaded, never what is in the corpus.
    """
    store = state.get("store")
    return {
        "ok": store is not None,
        "documents": len(store.documents) if store else 0,
        "chunks": len(store.chunks) if store else 0,
        "model": state["backend"].describe() if state.get("backend") else None,
        "rerank": state.get("reranker") or "off",
        "abstain_mode": ABSTAIN_MODE,
        "voice": voice.configured(),
        "uptime_s": int(time.time() - state["started"]) if state.get("started") else 0,
    }


@app.post("/ask", dependencies=[Depends(require_token)])
def ask(body: Question) -> dict:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")
    try:
        answer = answer_structured(question, state["retriever"], state["backend"])
    except Exception as exc:
        # The endpoint failing is the common case here: a key rotated, a budget exhausted,
        # the gateway unreachable. The message is the operator's, not the corpus's.
        _log({"event": "error", "question": question, "error": str(exc)[:500]})
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc
    _log({"event": "answer", **answer})
    return answer


def _log(record: dict) -> None:
    """One JSON object per line, appended under a lock so concurrent writes stay whole."""
    if not LOG_PATH:
        return
    record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    line = json.dumps(record, ensure_ascii=False)
    try:
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        # A log that cannot be written must not take the answer down with it.
        print(f"could not write to {LOG_PATH}: {exc}", flush=True)


class Speech(BaseModel):
    """What to say. Either an answer, which is rewritten for listening, or plain text."""

    text: str = Field(default="", max_length=8000)
    mode: str = Field(default="sourced", max_length=32)


@app.post("/speak", dependencies=[Depends(require_token)])
def speak(body: Speech) -> Response:
    """Answer text to mp3, with the key kept on this side.

    The spoken form is not the text on screen: citation markers are dropped, and an answer
    that is not grounded in the corpus says so before anything else, because audio has no
    banner to carry that.
    """
    text = voice.spoken_text({"answer": body.text, "mode": body.mode})
    try:
        audio = voice.speak(text)
    except voice.VoiceError as exc:
        _log({"event": "voice_error", "direction": "tts", "error": str(exc)[:300]})
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/mpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/transcribe", dependencies=[Depends(require_token)])
def transcribe(audio: bytes = Body(default=b""),
               mime: str = Query(default="audio/webm", max_length=64)) -> dict:
    """A recording to text. The body is the raw audio the browser captured."""
    try:
        text = voice.transcribe(audio, mime)
    except voice.VoiceError as exc:
        _log({"event": "voice_error", "direction": "stt", "error": str(exc)[:300]})
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _log({"event": "transcribed", "question": text, "bytes": len(audio)})
    return {"text": text}


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
