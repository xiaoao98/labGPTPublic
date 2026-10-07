# LabGPT, with voice

A retrieval assistant over a biology lab's own documentation — protocols, reagent records,
safety guidance, the staff directory and the lab's publications — that can be asked a
question out loud and read its answer back.

The voice layer is the subject of this branch. It exists for a specific situation: someone
at a bench, gloved, hands wet or contaminated, who needs step 7 of a protocol and cannot
touch a laptop. Everything else here is the system that layer sits on, summarised far
enough down to make sense of it.

```
speech  ──▶  ElevenLabs Scribe  ──▶  question text
                                          │
                                   local retrieval        no network
                                   grounded answer        leaves the host
                                   citation check
                                          │
answer  ◀──  ElevenLabs TTS      ◀──  text written for listening
```

## The voice layer

Two endpoints on the service, `POST /transcribe` and `POST /speak`, and a microphone
button and a Read aloud button on the page. Three decisions inside them are worth more
than the wiring.

### The spoken answer is not the answer on screen

A grounded answer carries citation markers: *"Warm the vial in a 37 °C water bath. [S2]"*.
Read aloud, `[S2]` becomes "bracket ess two". They are stripped before synthesis, and the
citations stay on screen where they can be clicked.

That removal takes something with it. A citation is what tells a reader the sentence came
from a document, and in audio there is nothing left to carry that distinction.

### An ungrounded answer says so before anything else

The system answers from the lab's documents when retrieval finds them, and from general
knowledge when it does not — labelled, on screen, behind an amber banner. **Audio has no
banner.** A confident synthetic voice reading general advice about centrifuge speeds
sounds exactly like one reading the lab's own protocol.

So the spoken form of an ungrounded answer opens with what it is, before any content a
listener could act on:

> Before I answer: this is not from the lab's documentation. The index had nothing on
> this, so what follows is general knowledge. Check anything lab-specific with the
> protocol owner.

Two outcomes are spoken differently again. When retrieval finds nothing and the fallback
is off, or when a draft answer cited no real document and was withheld, the text on screen
is a refusal that ends in advice for whoever runs the service — *check the corpus
directory and re-run the index*. Spoken, that is one sentence and the rest stays on the
screen where it belongs.

| outcome | what is heard |
|---|---|
| grounded | the answer, citation markers removed |
| ungrounded | the disclosure above, then the answer |
| nothing found | "I have nothing on this in the lab's indexed documentation. Try the protocol owner or the safety officer." |
| answer withheld | "I had a draft answer, but it cited no indexed document, so I am not reading it out. The details are on screen." |

### The API key never reaches the browser

Both directions are proxied by the service. A page calling `api.elevenlabs.io` directly
would hand the key to everyone who opens it. Proxying also leaves exactly one host to
allow outbound when this runs somewhere with a restricted egress policy, which the
production host has.

`GET /health` reports whether speech is configured at all, and the page hides both buttons
when it is not, rather than offering a control that fails.

### What it calls

| | endpoint | model |
|---|---|---|
| speech to text | `POST /v1/speech-to-text` | `scribe_v2` |
| text to speech | `POST /v1/text-to-speech/{voice_id}` | `eleven_multilingual_v2` |

The browser records with `MediaRecorder` — webm/opus in Chrome and Firefox, mp4 in Safari
— and the recording's type travels with it, so the server never guesses the format.

The voice defaults to George from the public library rather than looking one up. An
ElevenLabs key carries per-endpoint permissions, and a key allowed to synthesise speech is
not necessarily allowed to list voices: during development the same key transcribed
successfully and returned 401 on `GET /v1/voices`. Removing that call removed a permission
requirement that nothing needed. `ELEVENLABS_VOICE_ID` overrides it.

## Running the demo locally

Needs Python 3.10+, an ElevenLabs API key, and an OpenAI-compatible endpoint for
generation (this deployment uses Azure OpenAI; `LABGPT_LLM_*` works for the OpenAI API or
any compatible proxy).

```bash
uv sync --extra serve
```

Build the index once. It reads the corpus named by `LABGPT_DATA_DIR`, defaulting to the
synthetic sample set that ships with the repository, and writes `.index/`:

```bash
uv run python -m labrag.cli index
```

Then set the environment and start the service:

```bash
export LABGPT_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
export ELEVENLABS_API_KEY=...        # elevenlabs.io/app/settings/api-keys
export LABGPT_LLM_API_KEY=...        # or the AZURE_OPENAI_* set
uv run --extra serve uvicorn serve:app --host 127.0.0.1 --port 8080
```

It prints what it loaded, then open `http://127.0.0.1:8080` and paste the token once:

```
index: 678 documents, 2745 chunks
embedder: BAAI/bge-small-en-v1.5 warmed in 5.1s
rerank: off
model: AzureChatClient gpt-5, 2500 tokens, reasoning_effort=low
```

Click the microphone, ask a question out loud, click it again to stop. The transcript
fills the box and submits itself; **Read aloud** appears under the answer.

The microphone needs a secure context, which `127.0.0.1` counts as. Serving the same page
from a LAN address over plain HTTP will leave the button present and `getUserMedia`
refusing.

## Where the voice layer falls short

Mostly on the vocabulary this corpus is made of. Counted over the 678 indexed documents:

| in the corpus | occurrences | what synthesis does with it |
|---|---|---|
| `100,000 x g`, `1,500 × g` | 110 | reads the `x` as a letter; "times gravity" is meant |
| `0.22 um`, `0.2 µm`, `0.2 μm` | 29 | three spellings of micron, none of them said as one |
| primer sequences, `AAAACCGCTGATCACGCTCTG` | 526 | attempts to pronounce it as a word |
| molarity, `0.056 M` | 558 | "em" rather than "molar" |
| `1,300 rpm` | 46 | usually fine, sometimes spelled out |
| catalog numbers, `# 23225` | 20 | read as a quantity |

None of this is handled yet. The text sent for synthesis is the answer with its citation
markers removed and, when ungrounded, a disclosure in front — nothing else is rewritten.

Two fixes, in the order I would do them:

**Normalise before synthesis.** Units, scientific notation and catalog numbers are
deterministic rewrites: `100,000 x g` → "one hundred thousand times gravity", `0.22 um` →
"zero point two two micron". This belongs in the service, not in a dictionary, because it
depends on context the dictionary cannot see — `M` after a number is molar, `M` elsewhere
is not.

**Pronunciation dictionaries for the rest.** ElevenLabs supports
[alias and phoneme rules](https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-rules),
which is the right tool for the terms a rewrite cannot derive: reagent and kit names,
enzymes, species, and the lab's own shorthand. Alias rules cover most of it; phoneme rules
with IPA cover the ones that need exact sounds.

Primer sequences need neither. A twenty-base sequence read aloud is useless however it is
pronounced, and the right behaviour is to say "the sequence is on screen" and stop.

Three more limits, unrelated to vocabulary:

- **Synthesis waits for the whole answer.** Generation takes about 7.5 seconds, then
  synthesis runs on the finished text. Streaming both would let speech start while the
  answer is still being written.
- **Transcription is not checked against the corpus.** A misheard reagent name becomes a
  question about something the lab does not have, and the retrieval gate then correctly
  refuses it. The vocabulary to bias transcription toward is sitting in the index.
- **No barge-in.** Playback cannot be interrupted by speaking, which is what the bench
  case actually wants.

## What the voice layer sits on

Retrieval is hybrid: a local embedding model and a hand-written BM25, fused by reciprocal
rank, chunks expanded to whole documents. It runs entirely on the machine holding the
corpus — only generation calls an endpoint, with the question and six retrieved documents.

Two gates decide whether an answer is served. Before generation, retrieval below a cosine
threshold means the model is asked without sources and the answer is labelled; after it,
every `[S#]` is checked against the documents actually supplied, and an answer citing none
is withheld.

Measured on a public corpus with a labelled 100-question set, answers graded by hand:
**90 correct**, 93 with an optional reranking pass. Across 176 answers over two runs: zero
invented citations, zero answers with no citation, and none of the 19 deliberately
unanswerable questions answered anyway.

The service runs single-worker and holds the index in memory. Forty simultaneous questions
were measured at 8.2 seconds each against 7.5 for one, because the wait is network I/O and
the waits overlap.

`git log` on this branch is the voice work; the branch it came from carries the rest.
