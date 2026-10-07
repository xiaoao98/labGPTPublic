"""Speech in and speech out, through ElevenLabs.

The key stays on the server. A browser calling api.elevenlabs.io directly would carry the
key to every person who opens the page, so the service proxies both directions and the
page never sees it. That also leaves one host to allow outbound if this ever runs
somewhere with a restricted egress policy, which the lab server has.

WHAT GETS SPOKEN IS NOT WHAT IS ON SCREEN, and the difference is the point.

An answer carries [S1] markers. Read aloud they become "bracket ess one", which is noise,
so they are removed. Removing them also removes the thing that tells a reader this
sentence came from a document, which matters more in audio than on screen: a confident
synthetic voice reading general knowledge sounds exactly like one reading the lab's own
protocol, and there is no amber banner in audio to tell them apart. So an answer that is
not grounded in the corpus says so in its first sentence, before anything a listener could
act on.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

API_ROOT = "https://api.elevenlabs.io/v1"
# "George", from the public voice library, used in ElevenLabs' own examples. A default
# rather than a lookup: an API key can be scoped per endpoint, and a key allowed to
# synthesise speech is not necessarily allowed to list voices, which is a 401 on a call
# the caller never asked for. Override with ELEVENLABS_VOICE_ID; browse at
# elevenlabs.io/app/voice-library.
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2")

CITATION_RE = re.compile(r"\s*\[S\d{1,2}\]")

# What each outcome sounds like. The screen has a coloured banner and a layout to carry
# this; audio has only the words, and only in the order they are said.
SPOKEN_PREAMBLE = {
    "unsourced": "Before I answer: this is not from the lab's documentation. "
                 "The index had nothing on this, so what follows is general knowledge. "
                 "Check anything lab-specific with the protocol owner. ",
}

# Modes where the answer on screen is a refusal rather than an answer. Reading that text
# aloud says the same thing twice and ends on advice meant for whoever runs the service,
# about re-running the index. Spoken, these are one sentence and the screen keeps the rest.
SPOKEN_INSTEAD = {
    "refused": "I have nothing on this in the lab's indexed documentation. "
               "Try the protocol owner or the safety officer.",
    "withheld": "I had a draft answer, but it cited no indexed document, so I am not "
                "reading it out. The details are on screen.",
}


class VoiceError(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def _key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise VoiceError(
            "ELEVENLABS_API_KEY is not set, so speech is unavailable. Create a key at "
            "https://elevenlabs.io/app/settings/api-keys and put it in the environment "
            "file the service reads."
        )
    return key


def spoken_text(answer: dict) -> str:
    """The answer as it should be heard rather than read."""
    mode = answer.get("mode", "")
    if mode in SPOKEN_INSTEAD:
        return SPOKEN_INSTEAD[mode]
    text = CITATION_RE.sub("", answer.get("answer", "")).strip()
    return SPOKEN_PREAMBLE.get(mode, "") + text


def _request(path: str, data: bytes, content_type: str, accept: str = "application/json"):
    request = urllib.request.Request(
        f"{API_ROOT}{path}", data=data,
        headers={"xi-api-key": _key(), "Content-Type": content_type, "Accept": accept},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code == 401:
            # The key is probably valid and scoped too narrowly: ElevenLabs keys carry
            # per-endpoint permissions, so one that transcribes need not be allowed to
            # synthesise. Saying so beats "Unauthorized" against a key that demonstrably
            # works elsewhere.
            raise VoiceError(
                f"ElevenLabs refused this call as unauthorised: {detail}\n"
                f"  If other calls work, the key is valid but scoped: check its "
                f"permissions at https://elevenlabs.io/app/settings/api-keys"
            ) from exc
        raise VoiceError(f"ElevenLabs returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise VoiceError(f"could not reach ElevenLabs: {exc.reason}") from exc


def voice_id() -> str:
    """The voice to speak with: whatever was configured, otherwise a known public one."""
    return os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID


def speak(text: str) -> bytes:
    """Text to mp3."""
    if not text.strip():
        raise VoiceError("nothing to speak")
    body = json.dumps({"text": text, "model_id": TTS_MODEL}).encode("utf-8")
    return _request(f"/text-to-speech/{voice_id()}", body, "application/json", "audio/mpeg")


def transcribe(audio: bytes, mime: str = "audio/webm") -> str:
    """Recorded audio to text.

    The multipart body is assembled by hand rather than with a library: the service has no
    HTTP client dependency beyond the standard library, and one file field with one string
    field is not worth adding one.
    """
    if not audio:
        raise VoiceError("no audio received")
    boundary = "----labgpt" + os.urandom(8).hex()
    extension = {"audio/webm": "webm", "audio/mp4": "mp4", "audio/mpeg": "mp3",
                 "audio/wav": "wav", "audio/ogg": "ogg"}.get(mime.split(";")[0], "webm")
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\n"
        f"{STT_MODEL}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"speech.{extension}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
        audio,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    raw = _request("/speech-to-text", b"".join(parts),
                   f"multipart/form-data; boundary={boundary}")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise VoiceError(f"unreadable transcription response: {raw[:200]!r}") from exc
    text = payload.get("text")
    if text is None and payload.get("transcripts"):
        text = payload["transcripts"][0].get("text")
    if not text:
        raise VoiceError("the recording produced no text")
    return text.strip()
