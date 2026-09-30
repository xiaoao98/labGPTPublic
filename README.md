# LabGPT

A laboratory knowledge assistant. It answers questions about lab protocols, reagents,
safety guidance, team responsibilities and the lab's own publications.

Indexing, retrieval and reranking run entirely on the machine holding the corpus, so
nothing leaves during search. Generation is the one step with a choice: an
OpenAI-compatible endpoint, Azure OpenAI included, which means the handful of documents
retrieved for a question do leave; or a local open-weights model, in which case nothing
leaves at all. Which endpoint that is decides what a private corpus is exposed to, and is
the deployment choice this repository cannot make for you.

Two ways in: `labrag.cli` for indexing, retrieval, evaluation and batch answering, and
`demo.py` for an interactive chat loop over the same index.

Retrieval is hybrid: a local embedding model and a hand-written BM25, fused by reciprocal
rank. Every sourced answer cites the documents it came from, and every citation is checked
against the sources that were actually supplied. When retrieval comes back weak the CLI
refuses rather than guesses; the chat loop can instead answer from general knowledge, with
the answer marked as not being the lab's documentation.

The repository ships with a **synthetic corpus** in `data/sample`. Point it at a real
corpus to run it for real.

**Measured end to end on 100 labelled questions, answers graded by hand: 90% correct, or
93% with the optional reranking pass.** Reranking is off by default and turned on with
`--rerank`, or `LABGPT_RERANK=on` for the chat loop. The numbers, and what they do and do
not establish, are in [Results](#results).

## How it works

```
query
  |
  +-- dense   top 20 by cosine over chunk embeddings
  +-- BM25    top 20 by exact term overlap
  |
  +-- reciprocal rank fusion            rank-based, not score-based
  |
  +-- cross-encoder rerank              optional, off by default: --rerank
  |
  +-- expand chunks to documents        small-to-big, protocols served whole
  |
  +-- abstention gate                   cosine below threshold -> refuse, no model call
  |
  +-- generate with numbered sources
  |
  +-- citation validation               an answer with no valid citation is withheld
```

`demo.py` runs the same pipeline with three additions, none of which the CLI has: the
team directory is retrieved separately and appended above a relevance floor, so "who can
help" survives without the directory competing for the main slots; a web search leg, gated
by one classifier call and available on the local backend only; and, below the abstention
threshold, an answer from general knowledge with no sources, labelled as such. That last
one is a behavioural difference worth knowing about before comparing the two:
[When retrieval finds nothing](#when-retrieval-finds-nothing).

Chunks are what get scored and documents are what get returned. A query like "how long in
the water bath" only matches if the body text is indexed, while a protocol handed back
without its last step is worse than no answer at all, so the two granularities are kept
apart.

The two retrieval legs fail in opposite directions, which is the whole argument for
running both. Dense handles paraphrase: "how cold do I spin it" finds "centrifuge at 4 C"
without sharing a word. BM25 handles exact tokens: `SMP-17104`, `100000 x g` and `0.22 um`
match exactly or not at all. Measured on the same corpus, twenty questions are answered by
one leg and missed by the other.

## Results

Public corpus, 556 documents. 100 questions, 81 answerable and 19 deliberately
unanswerable, labelled by hand and checked twice: that every document id resolves, and
that the document actually contains the answer its note claims. Generation by gpt-5 at low
reasoning effort, six documents per query. Every answer was then read and graded by a
person.

**End to end, 90 of 100 correct, or 93 with the reranker on.** Reranking is opt-in; the
90 is what the shipped default produces. Both were graded by
reading every answer.

| | answerable, 81 | unanswerable, 19 | total |
|---|---|---|---|
| default | 71 | 19 | **90 / 100** |
| with reranker | 74 | 19 | **93 / 100** |
| oracle context | 80 | not applicable | — |

The default run is the first row: reranking is off unless `--rerank` asks for it, and
`eval` prints which one produced a given table. The nineteen unanswerable questions are
refused either way, with nothing invented in either run.

### Where the failures come from

The same questions were run a second time with the labelled documents placed into the six
slots, holding everything else fixed. Only one variable changes: whether the correct
document was present.

| | correct | incomplete | wrong |
|---|---|---|---|
| production | 71 (87.7%) | 3 | 7 |
| oracle context | **80 (98.8%)** | 0 | 1 |

Given the right documents the generator gets 80 of 81 right, so **retrieval accounts for
9 of the 10 failures and generation for 1**. Prompt changes, a larger model or a bigger
reasoning budget have at most 1.2% to win; retrieval has 11.1%. The three incomplete
answers also become complete under oracle context, so partial answers are a retrieval
symptom too rather than a generation habit.

The two confidence intervals do not overlap, 0.805–0.948 against 0.964–1.000, which is
what makes the conclusion hold at this sample size.

### Retrieval

Reported at k=6 because six documents are what reach the model.

| | success@6 | nDCG@10 |
|---|---|---|
| dense only | 0.815 | 0.638 |
| BM25 only | 0.840 | 0.678 |
| **hybrid** | **0.938** | **0.702** |

Hybrid beats the better single leg by 9.8 points. Which leg is stronger flips between
corpora, BM25 on one and dense on another, which is the case for running both rather than
picking one on a hunch.

Weakest categories, and the two that account for most of the retrieval loss:

| category | success@6 | n |
|---|---|---|
| paraphrase | 0.333 | 3 |
| people | 0.778 | 9 |

Both for the same reason: a 60-token biography and a question sharing no vocabulary with
its document give neither leg much to match on, against full-text sections that give both
legs plenty.

### Reranking

A cross-encoder pass over the fused candidates. It exists because fusion reads ranks and
discards scores, so a document one leg is certain about loses to one both legs merely
like.

Off by default, on with `--rerank`. It was on-where-a-GPU-is-present for a while, and a
default that changes with the machine makes two runs incomparable without reading the
header that says which one happened, for a gain of three questions at n=100 that the
interval does not resolve. `--no-rerank` still says off explicitly.

| | success@1 | success@6 | nDCG@10 | seconds per query |
|---|---|---|---|---|
| no reranker | 0.469 | 0.938 | 0.702 | ~0.1 |
| ms-marco-MiniLM-L-6, 22M | 0.630 | 0.938 | 0.781 | 2.3 |
| **bge-reranker-base, 278M** | **0.802** | **0.963** | **0.854** | 13.1 |

The larger model is better on every measure, and the difference between the two is not a
matter of degree. MiniLM reorders without finding anything new, leaving success@6 at
0.938, and it pays for its gains elsewhere: `safety` drops from 1.000 to 0.875 and
`paraphrase`'s recall@10 from 1.000 to 0.333. bge moves success@6 to 0.963, holds `safety`
and `paper_topic` at 1.000, and takes `exact_identifier` and `paper_result` to 1.000 at
success@1. `people` goes from 0.333 to 0.889 at success@1.

So the first reading, that reranking helps ordering but costs safety, was a property of
the small model rather than of reranking.

`paraphrase` stays at 0.333 under every configuration tried. Three questions written to
share no vocabulary with their sources defeat the cross-encoder as thoroughly as they
defeat the bi-encoder, so what remains of the retrieval loss is not a ranking problem.

#### What it is worth end to end

Three questions in a hundred. 71 correct of 81 answerable without it, 74 with, against 80
for the oracle context, so it recovers a third of the nine failures the oracle run
attributes to retrieval. The intervals overlap, 0.841-0.959 against 0.880-0.980, so the
direction holds and the size does not at this sample size.

Retrieval moves much further than the answers do, success@1 from 0.469 to 0.802 against
three questions changing, and the reason was measured rather than guessed. The reranker
changes the six documents served on 76 of the 81 questions, but on 73 of those the
documents that came and went contain no answer either way: the gold document was already
there and the model could already use it. Only three questions gain a gold document they
did not have, and none lose one.

Three of the nineteen unanswerable questions move from the model refusing to the gate
refusing, because the reranked set lowers their best cosine below the threshold. Same
outcome, reached without spending a token.

#### Cost

13.1 seconds a query on this CPU against 4.6 for generation, so on this machine reranking
costs about three times what answering does. On a GPU it should be a rounding error, which
is the case worth turning it on for.

Candidate pool size was tested and left at 20 per leg. Raising both to 50 gives exactly
the same success@1 with success@6 1.2 points lower and twice the latency, which follows
from something already measured: gold documents reach 100% coverage by fused rank 30, so a
larger pool adds only candidates ranked below where any of them sit.

### Grounding

Counted over the two runs in the table above, 176 answers in total: 90 from the default
run and 86 from the reranked one, the rest of each being refusals.

| | result |
|---|---|
| invented citation numbers | 0 of 176 answers |
| answers with no citation at all | 0 of 176 |
| unanswerable questions answered anyway | 0 of 19 |
| answers graded "unsupported" by a human | 0 of 176 |

The first three rows are recomputed from the run files rather than transcribed, so they
move if the runs are regenerated. Both gates held across both runs.

Of the 19 unanswerable questions, 9 were stopped by the cosine gate before any model call
and 10 were refused by the model itself under the prompt's third rule, which is the split
the two-gate design predicts.

**These numbers are the CLI's behaviour.** `ask` and `answer-all` refuse below the gate,
which is what was measured. `demo.py` defaults to answering those questions from general
knowledge instead, so the "unanswerable questions answered anyway" row would not hold for
it, and is not supposed to: the answer is marked as unsourced rather than served as the
lab's documentation. Run `demo.py` with `LABGPT_ABSTAIN_MODE=refuse` to get the measured
behaviour.

### Cost

374,993 tokens over 100 questions, 3,750 per query, of which the retrieved evidence is
89.8%. 4.6 seconds per question, and no failed calls. Measured from the default run the
table above grades, `runs/prod.jsonl` below.

### What these numbers do not establish

- **n=81, so the interval is roughly ±7 points.** Category-level figures with three or
  nine questions in them are not measurements.
- **One annotator.** The same person wrote the labels and graded the answers, and the
  labels have been corrected four times during development.
- **The oracle context is built from those labels.** A question whose gold set is missing
  a document that answers it would score the model wrong for being right, which biases the
  98.8% downward rather than up.
- **The private corpus has never been through generation.** The comparison against the
  original prompt-stuffing assistant is not done, and that assistant is no longer in the
  tree, so running it now means checking out `d681ce6` alongside this.

Reproduce with:

```bash
uv run python -m labrag.cli --index-dir .index-pub eval --questions eval/questions_public.yaml
uv run python -m labrag.cli --index-dir .index-pub answer-all \
    --questions eval/questions_public.yaml --out runs/prod.jsonl
uv run python -m labrag.cli --index-dir .index-pub answer-all --rerank \
    --questions eval/questions_public.yaml --out runs/rerank.jsonl
uv run python -m labrag.cli --index-dir .index-pub answer-all --oracle \
    --questions eval/questions_public.yaml --out runs/oracle.jsonl
```

`runs/` is gitignored, because answers over a private corpus quote it back verbatim.
The three files above are what every number in this section is computed from, except
correct and incomplete, which are human grades the files do not carry.

## Setup

- Python 3.10 or newer
- Building the retrieval index runs on CPU and takes seconds on this corpus. The embedding
  model, `BAAI/bge-small-en-v1.5`, downloads once at about 130 MB and then works offline
- Generation calls any OpenAI-compatible endpoint by default, so it needs no GPU.
  Reranking wants one and runs on CPU if there is none. Only `demo.py` with
  `LABGPT_BACKEND=local` requires a GPU, for the model it loads itself

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv && .venv/bin/pip install -e .
```

One optional extra, `search`, for `demo.py`'s web branch:

```bash
uv sync --extra search
```

Everything the retrieval, evaluation and answering path needs is a hard dependency, and
the scripts in `tools/` use the standard library. The web branch is the exception because
it pulls a browser driver, and it runs on the local backend only, so an endpoint-only
deployment never needs it.

## Running it

Build the index first. It reads the corpus, chunks it, embeds the chunks locally and
writes `.index/`:

```bash
uv run python -m labrag.cli index
```

The corpus it reads is `LABGPT_DATA_DIR`, which defaults to the synthetic `data/sample`.
There is no `--data-dir` flag, and `index` prints the corpus path it used on its first
line: an index built from the wrong directory retrieves nothing and does not otherwise
announce it.

Then ask it something. This path needs an endpoint rather than a GPU, so check the
endpoint first:

```bash
uv run python -m labrag.cli selftest                     # one prompt, checks the endpoint
uv run python -m labrag.cli ask "how do I thaw BJ cells"
```

Or the chat loop, over the same index:

```bash
uv run python demo.py
```

It reads `.index/` and nothing else from disk, and prints what it resolved on startup:

```
index: 678 documents, 2745 chunks
rerank: off
model: AzureChatClient gpt-5-chat, 2500 tokens, reasoning_effort=low
```

`exit` or Ctrl-D leaves. Each answer is followed by the sources it cited, and by a count
of factual sentences that carried no citation.

Reranking is off here too. `LABGPT_RERANK=on` turns it on for a session, `auto` defers to
the hardware, and either way the model loads at startup rather than inside the first
question, so a hub that cannot be reached costs one warning and the session continues
without it:

```bash
LABGPT_RERANK=on uv run python demo.py
```

## Serving it to other people

`serve.py` puts the same pipeline behind one endpoint and one page, for a lab that wants
to ask questions without a terminal:

```bash
export LABGPT_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
export LABGPT_LOG_PATH=/srv/labgpt/qa.jsonl
uv run --extra serve uvicorn serve:app --host 0.0.0.0 --port 8080
```

- `GET /` the page, `POST /ask` the endpoint, `GET /health` what the process loaded
- every request carries `X-LabGPT-Token`, checked in constant time. The service refuses to
  start without `LABGPT_API_TOKEN` rather than defaulting to open: a service answering
  from an institutional corpus should not come up unauthenticated because a variable was
  unset
- one worker on purpose. The index is loaded once and held; the wait in a request is the
  endpoint call, which is I/O, so a thread pool serves a lab and more workers would only
  multiply the index in memory

The page renders the four outcomes differently, which is the point rather than decoration:
a sourced answer with its citations, an unsourced one behind an amber banner saying it is
not the lab's documentation, and refusals and withheld drafts in grey. Rendering an
unsourced answer like a sourced one removes the only thing separating general advice from
lab practice.

The endpoint is the server's, configured with the `AZURE_OPENAI_*` variables the service
is started with. Callers send questions and nothing else: no key is typed into the page,
so none travels from a browser and none sits in anyone's `localStorage`. Usage is billed
to whichever subscription the service was started with.

Two secrets still cross the network in the clear if this is served over plain `http://`:
the access token, on every request, and the corpus itself, in every answer. That is
tolerable between a browser and `127.0.0.1` and much less so across a lab network, so put
TLS in front of it before the address is shared.

`LABGPT_LOG_PATH` records every question and answer as JSONL. That is what makes it
possible to see what people actually ask and which questions get refused, and it also
means the file quotes the corpus back verbatim and records who wanted to know what. Keep
it on the server, readable by the account that runs the service, and out of any backup
that leaves the building.

## Where generation happens

Retrieval is always local. Only the generation call changes, so every configuration below
still embeds the query and scores documents on the machine, and sends at most the question
and the retrieved chunks anywhere.

`demo.py` picks between two backends with `LABGPT_BACKEND`:

| `LABGPT_BACKEND` | What answers | Web leg |
|---|---|---|
| `api` (default) | A hosted OpenAI-compatible endpoint, Azure included | no |
| `local` | `LABGPT_MODEL` loaded with transformers, needs a GPU | yes |

The endpoint is the default and the local model is opt-in, named rather than sniffed from
the environment: an unset variable should not quietly move generation from one to the
other. With nothing configured, `demo.py` says what to set and stops rather than failing
on the first question. The CLI has no local path at all; it always calls an endpoint.

### Azure OpenAI, direct or behind an API Management gateway

```bash
export AZURE_OPENAI_GATEWAY=https://<gateway-host>/<product>   # host and product path
export AZURE_OPENAI_TEAM_ID=<team-id>                          # appended to the gateway
export AZURE_OPENAI_MODEL_ID=<deployment-name>                 # NOT the model family name
export AZURE_OPENAI_API_VERSION=<api-version>
export APIM_OPENAI_SUBSCRIPTION_KEY=<key>

uv run python -m labrag.cli selftest
```

`selftest` prints the assembled URL before it sends anything, which is the fastest way to
catch a misconfiguration:

```
{gateway}/{team-id}/openai/deployments/{deployment}/chat/completions?api-version={version}
```

Set `AZURE_OPENAI_ENDPOINT` to the whole base instead if you have it in one piece; the two
forms are equivalent, and a value already carrying `/openai/deployments/...` or an
`?api-version=` query is trimmed back rather than doubled. The key is sent as both
`api-key` and `Ocp-Apim-Subscription-Key`, so one client reaches a direct Azure resource
and a gatewayed one without being told which is in front of it.

**A 401 here usually is not the key.** A gateway answers a path it cannot route with
`"Access denied due to invalid subscription key or wrong API endpoint"`, so a deployment
name that does not exist, or a base URL missing the team segment, produces an error that
reads like bad credentials. Check the URL `selftest` prints before suspecting the key.

Reasoning deployments (`gpt-5`, `o1`, `o3`, `o4`) reject `max_tokens` and any temperature
but their default, which the client detects and corrects on the first call. They also
spend reasoning tokens out of the answer budget and spend them first, so the budget is not
an answer length. The 2500 default is measured rather than guessed: over 101 real
questions with retrieved context, low effort spent a median of 192 reasoning tokens, 448
at the 90th percentile and 704 at the worst successful call, while an earlier 800-token
budget left seventeen questions with nothing written at all. An exhausted budget is raised
as an error rather than recorded, because a batch of blank answers reads as the model
declining to answer. `LABGPT_MAX_TOKENS` and `LABGPT_REASONING_EFFORT` override both the
budget and the effort.

### OpenAI, or any OpenAI-compatible endpoint

```bash
export LABGPT_LLM_BASE_URL=http://localhost:8000/v1    # vLLM, Ollama, a LiteLLM proxy
export LABGPT_LLM_API_KEY=<key>
export LABGPT_LLM_MODEL=<model>
```

When Azure variables are set too, Azure wins by default: on a machine configured for both,
the approved endpoint should be the one that gets institutional text, and that should not
depend on remembering a flag. `--provider openai` overrides it per run.

### The local model

```bash
LABGPT_BACKEND=local LABGPT_MODEL=Qwen/Qwen3-8B uv run python demo.py
```

The only configuration in which no text leaves the machine, which is the reason it exists.
It needs a GPU with room for the model, and `transformers` and `torch`; the api path needs
neither, and imports neither.

## When retrieval finds nothing

Below the abstention threshold there are no sources to ground an answer in. The CLI
refuses. `demo.py` defaults to putting the question to the model anyway with nothing
attached, and labelling what comes back:

- a different system prompt, not the sourced one with its sources removed. It opens by
  saying the lab's documentation does not cover this, refuses to state lab specifics it
  does not have (protocol parameters, catalog numbers, storage locations, who is
  responsible for what), sends exposures and spills to the safety officer instead of
  improvising a procedure, and emits no citations
- a banner before and after the answer
- no citation validation, because nothing was supplied to cite. A `[S#]` emitted anyway is
  reported as invented
- recorded as `answer:unsourced` in the metrics, so the two kinds of answer count apart

This is the intended behaviour for the chat loop: someone at a prompt is better served by
a general answer that says plainly it is general than by a refusal, as long as it cannot
be mistaken for the lab's own documentation. `LABGPT_ABSTAIN_MODE=refuse` restores the
gate-stops-here behaviour, which is what the [Results](#results) were measured with and
what the CLI still does.

The second gate is unaffected in both modes. An answer that *was* given sources and cited
none of them is still withheld: that is a model ignoring its evidence rather than a gap in
the corpus.

## Retrieval index

```bash
uv run python -m labrag.cli chunks    # inspect chunking only, no embedding, no GPU
uv run python -m labrag.cli index     # ingest, embed, write .index/
uv run python -m labrag.cli info      # describe an existing index
uv run python -m labrag.cli search "SMP-17104" --explain   # retrieve, no LLM
uv run python -m labrag.cli ask "how do I thaw BJ cells" --dry-run  # gate + prompt
uv run python -m labrag.cli eval --failures    # score retrieval against a question set
uv run python -m labrag.cli sweep              # compare retrieval configurations
uv run python -m labrag.cli eval --rerank      # with the cross-encoder pass, off by default
uv run python -m labrag.cli answer-all --questions ... --out ...   # batch, to JSONL
```

`chunks` is there so chunk boundaries can be iterated on without paying for embedding,
which is the slow part. An index is a copy of whatever it was built from, so none are
committed.

## Measuring cost per query

Every call records prompt tokens, generated tokens and wall time, and `answer-all` writes
them per question into the output JSONL alongside the answer. `demo.py` writes the same
records to `LABGPT_METRICS` when it is set, tagged by stage, so a session breaks down into
retrieval, the web classifier and the answer itself:

```bash
LABGPT_METRICS=metrics.jsonl uv run python demo.py    # ask some questions, then exit
uv run python labgpt_metrics.py metrics.jsonl         # per-stage breakdown
```

On the local backend those counts come from the input and output tensors and are exact for
the tokenizer in use. On the api path they are whatever the endpoint reports.

Reasoning tokens are recorded separately from the answer. They are billed as completion
tokens while forming no part of the reply, and conflating the two is what made an early
run return a hundred empty answers with no error.

## Configuration

Corpus paths resolve through `labgpt_config.py`. Override with environment variables, or
copy `.env.example` to `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LABGPT_DATA_DIR` | `data/sample` | Corpus directory, read when building an index |
| `LABGPT_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model, or a directory holding it |
| `LABGPT_RERANK` | `off` | `off`, `on`, `auto` (on where a GPU makes it cheap); `demo.py` only |
| `LABGPT_RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder, `demo.py` only |
| `LABGPT_MAX_TOKENS` | `2500` | Answer budget, shared with reasoning tokens |
| `LABGPT_REASONING_EFFORT` | `low` in `demo.py`, endpoint default in the CLI | `minimal`/`low`/`medium`/`high` |

Endpoint selection, read wherever a model is called:

| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | The whole base URL, used as given |
| `AZURE_OPENAI_GATEWAY` | Gateway host and product path, with the team id appended |
| `AZURE_OPENAI_TEAM_ID` | Team segment of a gateway route |
| `AZURE_OPENAI_MODEL_ID` | Deployment name; it becomes a path segment, not a body field |
| `AZURE_OPENAI_API_VERSION` | Required by Azure, no default |
| `APIM_OPENAI_SUBSCRIPTION_KEY` | Sent as `api-key` and as `Ocp-Apim-Subscription-Key` |
| `LABGPT_LLM_BASE_URL` | OpenAI-compatible base, e.g. a vLLM or LiteLLM proxy |
| `LABGPT_LLM_API_KEY` | Key for that endpoint |
| `LABGPT_LLM_MODEL` | Default model for the CLI |

`demo.py` only:

| Variable | Default | Purpose |
|---|---|---|
| `LABGPT_BACKEND` | `api` | `api` or `local`; which model answers |
| `LABGPT_ABSTAIN_MODE` | `answer` | `answer` (unsourced fallback) or `refuse` |
| `LABGPT_INDEX_DIR` | `.index` | Where the chat loop looks for the index |
| `LABGPT_MODEL` | `Qwen/Qwen3-32B` | Local model, `LABGPT_BACKEND=local` only |
| `LABGPT_ASSISTANT_NAME` | `LabGPT` | Name shown in the chat prompt |
| `LABGPT_METRICS` | unset | Path to write per-call metrics JSONL; off when unset |

On a network that blocks huggingface.co, both models have to arrive by hand. Fetch them
elsewhere, then point `LABGPT_EMBED_MODEL` at the directory. The reranker is off by
default, so on such a network it costs nothing until it is asked for. A hub that cannot be reached is
reported as such rather than as a missing package.

## Using a private corpus

Create a directory with the same filenames as `data/sample` (see
[data/sample/README.md](data/sample/README.md) for the shape of each), then:

```bash
export LABGPT_DATA_DIR=/path/to/private/corpus
uv run python -m labrag.cli --index-dir .index-private index
```

**Keep any real corpus outside this repository.** Everything under `data/` is gitignored
by default and allowed back one directory at a time, because naming them individually is
how two private corpora came to be sitting untracked but committable during development.

A private corpus is also a reason to be careful about which endpoint answers it. Sending
it to a personal API key is a larger exposure than committing it would have been.

## Known limitations

- **Paraphrase.** Three questions written to share no vocabulary at all with their
  source documents score 0.333 at success@6, and nothing tried has moved them: not
  reranking with either model, not a larger candidate pool. Both retrieval legs are
  defeated by the same thing, so the next place to look is the embedding model rather than
  the ranking.
- **Three questions is not a demonstrated improvement.** Reranking moves the end-to-end
  score from 90 to 93 of 100 and the confidence intervals overlap. That is why it is
  opt-in rather than on wherever it is cheap: the gain is not established at n=100, and a
  default that varies by machine would put an unestablished difference into runs nobody
  asked to differ.
- **Fusion discards calibrated similarity.** Reciprocal rank fusion reads only ranks, so a
  chunk both legs place in their top handful beats a chunk one leg is certain about. On
  "what should I do if I stick myself with a needle" the correct entry has cosine 0.710
  against a wrong one's 0.498 and still comes second. Leg weights exist and are left at
  1.0; measurement says weighting is the wrong tool and the reranker is the right one.
- **The abstention gate reads cosine, and the reranker's score is better calibrated.**
  `q047` has the right document at rank 4 and is still refused, because that document's
  cosine is 0.498 against a threshold of 0.62. A gate on the reranker's score would have
  the signal it needs, but those logits are not comparable across queries, so it is not a
  drop-in substitution.
- **The abstention threshold is a compromise, not a solution.** Answerable and
  unanswerable questions overlap in cosine, so no threshold separates them. 0.62 is where
  accuracy peaks, chosen on the same questions it is scored on, and the citation gate
  downstream is the second line of defence for what it lets through.
- **One annotator.** The evaluation set is one person's judgement with no second annotator
  and no agreement score, and its labels have been corrected four times.
- **Paper sections are not expanded to whole documents,** unlike protocols. Deliberate,
  and argued in `labrag/documents.py`, but it means a methods answer sees one chunk rather
  than the section.
- **Generated answers are not reproducible on the api path.** A reasoning deployment
  refuses any temperature but its default, and `seed` is documented as best effort, so
  identical inputs can give different answers and any comparison between configurations
  has to treat the generated half as noisy.
- **The unsourced fallback is unmeasured.** Everything in [Results](#results) was run with
  the gate refusing. What `demo.py` now returns below the threshold has not been graded,
  and the labelling that keeps it honest is a prompt instruction rather than a checked
  gate: the citation validator cannot help when there are no citations to validate.
- **The two entry points do not abstain alike, on purpose.** `demo.py` answers below the
  gate and the CLI refuses, so a question can be refused by `ask` and answered by the chat
  loop. That split is intended rather than pending: a person at a prompt is better served
  by a general answer that says it is general than by a refusal, while `answer-all` feeds
  an evaluation whose numbers depend on the gate holding. It does mean the chat loop is
  not the thing [Results](#results) measured.
- **`tools/` builds corpora, and is not covered by anything.** The four scripts that
  produce an evaluation corpus have no tests and reference input files that are not in the
  repository, so a corpus rebuild is checked by reading its output.

## Roadmap

The retrieval work this file used to list as future is done: structure-aware chunking,
hybrid retrieval with rank fusion, validated citations, an abstention gate, and a labelled
evaluation set. What the measurements now point at:

1. A larger embedding model, which is the only untried lever on `paraphrase` and the one
   category no reranker moved at all
2. A second annotator on the evaluation set, so category-level numbers mean something
3. The head-to-head against the original prompt-stuffing assistant, on the private corpus,
   through an approved endpoint. It needs `d681ce6` checked out beside this one, which is
   the last commit where that assistant still ran

## Layout

| Path | Purpose |
|---|---|
| `labrag/` | Retrieval and answering: chunking, ingest, embeddings, BM25, fusion, gates |
| `labrag/cli.py` | `index`, `search`, `ask`, `eval`, `sweep`, `answer-all`, `selftest` |
| `labrag/rerank.py` | Cross-encoder reranking over the fused candidates |
| `labrag/answerer.py` | Chat clients: OpenAI-shaped, Azure, local transformers; the two gates |
| `demo.py` | Interactive chat loop over the index |
| `labgpt_config.py` | Corpus paths, and the local model name |
| `labgpt_metrics.py` | Per-call token and latency instrumentation |
| `search/` | DuckDuckGo search and page fetching for `demo.py`'s web leg |
| `eval/` | Labelled question sets |
| `tools/` | Corpus construction: PMC fetch, clean, merge, and a dump of the lab's own SQLite |
| `data/sample/` | Synthetic corpus |

The prompt-stuffing assistant this grew out of was removed in `a2bc0ee` once the retrieval
path replaced it, and most of it stayed removed: the per-domain scripts, the SQLite build,
`prompt-engineering/`. `demo.py` came back on the retrieval path rather than the old one,
and brought back only what it imports, which is why `search/` holds two modules rather than
six. `tools/export_experiments_db.py` survives because the lab's database is still where the real
protocol text comes from, but nothing here builds one any more.
