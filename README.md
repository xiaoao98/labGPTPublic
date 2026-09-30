# LabGPT

A laboratory knowledge assistant. It answers questions about lab protocols, reagents,
safety guidance, team responsibilities, and the lab's own publications, over a local
retrieval index, with citations on every answer and a refusal when the index does not
support one.

Generation runs either on a local open-weights model, in which case no text leaves the
machine, or on an approved hosted endpoint such as the institutional Azure OpenAI gateway.

The repository ships with a **synthetic corpus** in `data/sample`. Point it at a private
corpus to run it for real.

## How it works today

```
query
  |
  +-- hybrid retrieval over one index
  |     dense: bge-small embeddings, cosine
  |     lexical: BM25
  |     fused with reciprocal rank fusion   -> top k documents
  |
  +-- abstention gate on the retrieval cosine, before a token is spent
  +-- the directory, retrieved separately, appended only above a relevance floor
  |
  +-- local backend only:
  |     classify: does this need web search?   (1 LLM call, gates a real external call)
  |     if yes -> DuckDuckGo search, fetch and summarize pages
  |
  +-- generate with numbered sources and required citations
  +-- validate citations; withhold the answer if none resolve to a real source
```

The web leg belongs to the local path only. An `api` run skips it, and skips the classifier
call that gates it, so it answers from the retrieved corpus and nothing else: no question
text leaves through a second channel, and no question pays for a classifier call whose
answer would be ignored.

Every answer carries `[S#]` markers, each checked after generation against the sources
that were actually supplied. An answer citing a source that was never given is withheld
rather than served, because in this domain invented provenance reads as verified.

## Setup

- Python 3.10 or newer
- Building the retrieval index runs on CPU and takes a few seconds on this corpus
- Answering calls a hosted endpoint by default, which needs no GPU. Set
  `LABGPT_BACKEND=local` to answer on `LABGPT_MODEL` instead (`Qwen/Qwen3-32B` by default),
  which needs a GPU with enough memory for it

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv && .venv/bin/pip install -e .
```

Optional extras: `search` for the web-search branch of the local backend, `ingest` for the corpus
preparation scripts in `buildDB/` and `tools/`, `dev` for pytest. With uv:
`uv sync --extra search`.

## Running it

Build the retrieval index first. It reads the corpus, chunks it, embeds the chunks
locally, and writes `.index/`:

```bash
uv run python -m labrag.cli index
```

The corpus it reads is `LABGPT_DATA_DIR`, which defaults to the synthetic `data/sample`.
There is no `--data-dir` flag; point the variable at the real corpus and rebuild:

```bash
export LABGPT_DATA_DIR=data/eval_lab
uv run python -m labrag.cli index
```

`index` prints the corpus path it used on the first line, which is worth a glance: an index
silently built from `data/sample` retrieves nothing from the real corpus and does not
otherwise announce it. See [Using a private corpus](#using-a-private-corpus).

Then the assistant:

```bash
uv run python demo.py
```

It needs nothing else: the index is the only thing it reads from disk. `experiments.db` is
not part of this path, and neither is `buildDB/build_sample_db.py` that generates it; only
the per-domain scripts below still query SQLite.

`demo.py` answers on a hosted endpoint by default and prints which model it is using on
startup. Configure the endpoint first, or pass `LABGPT_BACKEND=local` to answer on a local
model; see [Choosing the model](#choosing-the-model) below.

The individual domains can be run on their own: `protocolDemo.py`, `safetyDemo.py`,
`memberInfoDemo.py`, `paperDemo.py`. These are the original per-domain scripts and still
stuff their own corpus into the prompt; they are not on the retrieval path.

The embedding model downloads once, about 130 MB, and then runs offline. On a network that
blocks huggingface.co, fetch it elsewhere or point `LABGPT_EMBED_MODEL` at a local copy of
the same model.

## Choosing the model

Two backends, selected by `LABGPT_BACKEND`:

| `LABGPT_BACKEND` | What answers |
|---|---|
| `api` (default) | A hosted OpenAI-compatible endpoint, Azure OpenAI included. Corpus only, no web leg |
| `local` | `LABGPT_MODEL`, loaded with transformers onto the local GPU. Web leg available |

**The endpoint is the default; the local model is opt-in.** That is deliberate rather than
sniffed from the environment: an unset variable should not quietly move generation from one
to the other. With no endpoint configured, `demo.py` says so and stops instead of failing
on the first question.

Retrieval is local either way. Only the generation call changes, so an `api` run still
reads the corpus, embeds the query and scores documents on the machine, and sends only the
question and the retrieved chunks to the endpoint.

### Azure OpenAI, direct or behind an API Management gateway

This is the institutional path. Set the variables, then check them before running
anything long:

```bash
export AZURE_OPENAI_GATEWAY=https://<gateway-host>/<product>   # host and product path
export AZURE_OPENAI_TEAM_ID=<team-id>                          # appended to the gateway
export AZURE_OPENAI_MODEL_ID=<deployment-name>                 # NOT the model family name
export AZURE_OPENAI_API_VERSION=<api-version>
export APIM_OPENAI_SUBSCRIPTION_KEY=<key>

uv run python -m labrag.cli selftest      # one trivial prompt, prints what it will call
uv run python demo.py
```

`selftest` prints the assembled URL before it sends anything, which is the fastest way to
catch a misconfiguration:

```
{gateway}/{team-id}/openai/deployments/{deployment}/chat/completions?api-version={version}
```

If the base URL is already known in full, set `AZURE_OPENAI_ENDPOINT` to it instead of
`AZURE_OPENAI_GATEWAY` plus `AZURE_OPENAI_TEAM_ID`; the two forms are equivalent and the
whole URL is assembled the same way the `openai` SDK's `AzureOpenAI` client assembles it.
A value that already carries `/openai/deployments/...` or an `?api-version=` query is
trimmed back rather than doubled.

**A 401 here usually is not the key.** An API Management gateway answers a path it cannot
route with `"Access denied due to invalid subscription key or wrong API endpoint"`, so a
deployment name that does not exist, or a base URL missing the team segment, produces an
error that reads like bad credentials. Check the URL `selftest` prints first.

Two more things worth knowing about the reasoning deployments (`gpt-5`, `o1`, `o3`, `o4`):
they reject `max_tokens` and any temperature but their default, which the client detects
and corrects on the first call; and they spend their reasoning tokens out of the same
budget as the answer, so too small a budget returns an empty answer rather than an error.
`LABGPT_REASONING_EFFORT` controls the spend, and on this workload answers are quoted from
supplied sources rather than derived, so `low` is usually enough.

### OpenAI, or any OpenAI-compatible endpoint

The same path serves a self-hosted vLLM, Ollama or a LiteLLM proxy:

```bash
export LABGPT_LLM_BASE_URL=http://localhost:8000/v1
export LABGPT_LLM_API_KEY=<key>
export LABGPT_LLM_MODEL=<model>
```

When Azure variables are set too, Azure wins by default: on a machine configured for both,
the approved endpoint should be the one that gets institutional text, and that should not
depend on remembering a flag. Pass `--provider openai` to the CLI to override it
deliberately.

### The local model

Not the default, so it has to be asked for:

```bash
export LABGPT_BACKEND=local
export LABGPT_MODEL=Qwen/Qwen3-8B     # smaller than the 32B default
uv run python demo.py
```

Or for one run: `LABGPT_BACKEND=local uv run python demo.py`.

This is the configuration in which no text leaves the machine at all, which is the reason
the local path exists. It needs a GPU with room for the model, and `transformers` and
`torch` installed; the api path needs neither.

## Retrieval index

```bash
uv run python -m labrag.cli chunks    # inspect chunking only, no embedding, no GPU
uv run python -m labrag.cli index     # ingest, embed, write .index/
uv run python -m labrag.cli info      # describe an existing index
uv run python -m labrag.cli search "SMP-17104" --explain   # retrieve, no LLM
uv run python -m labrag.cli ask "how do I thaw BJ cells" --dry-run  # gate + prompt
uv run python -m labrag.cli ask "how do I thaw BJ cells"    # and call the model
uv run python -m labrag.cli eval --failures    # score against eval/questions.yaml
uv run python -m labrag.cli sweep             # compare retrieval configurations
uv run python -m labrag.cli selftest          # one prompt, to check the endpoint
uv run python -m labrag.cli answer-all --questions eval/questions.yaml --out answers.jsonl
```

`chunks` is there so chunk boundaries can be iterated on without paying for embedding,
which is the slow part. The index is generated, never committed.

`ask`, `selftest` and `answer-all` are the commands that call a model, and they read the
same environment as `demo.py`, with `--provider` and `--model` to override it per run.
`answer-all` writes one JSONL record per question, flushed as it goes, and `--resume`
skips ids already in the output file, so an interrupted batch is cheap to restart.

## Measuring cost per query

Every call to the model records prompt tokens, generated tokens, and wall time. This exists
so that the retrieval work has a measured baseline to be compared against rather than an
estimated one.

```bash
LABGPT_METRICS=metrics.jsonl python demo.py     # run some queries, then exit
python labgpt_metrics.py metrics.jsonl          # per-stage breakdown
```

On the local backend token counts come from the input and output tensors, so they are exact
for the tokenizer in use. On the api path they are whatever the endpoint reports in its
`usage` field, which includes reasoning tokens spent but not shown.

## Configuration

Everything resolves through `labgpt_config.py`. Override with environment variables, or
copy `.env.example` to `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LABGPT_DATA_DIR` | `data/sample` | Corpus directory |
| `LABGPT_DB_PATH` | `experiments.db` | Generated SQLite database |
| `LABGPT_MODEL` | `Qwen/Qwen3-32B` | Hugging Face model id |
| `LABGPT_ASSISTANT_NAME` | `LabGPT` | Name shown in the chat prompt |
| `LABGPT_METRICS` | unset | Path to write per-call metrics JSONL; off when unset |
| `LABGPT_BACKEND` | `api` | `api` or `local`; which model answers |
| `LABGPT_INDEX_DIR` | `.index` | Where `demo.py` looks for the index |
| `LABGPT_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model, or a local directory holding it |
| `LABGPT_REASONING_EFFORT` | unset | `minimal`/`low`/`medium`/`high`; reasoning deployments only |
| `GOOGLE_API_KEY` | unset | Optional, for `search/callGoogleAPI.py` |
| `GOOGLE_SEARCH_ENGINE_ID` | unset | Optional, for `search/callGoogleAPI.py` |

Endpoint variables, read when the backend resolves to `api`:

| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | The whole base URL, used as given |
| `AZURE_OPENAI_GATEWAY` | Gateway host and product path, with the team id appended to it |
| `AZURE_OPENAI_TEAM_ID` | Team segment of a gateway route |
| `AZURE_OPENAI_MODEL_ID` | Deployment name; it becomes a path segment, not a body field |
| `AZURE_OPENAI_API_VERSION` | Required by Azure, no default |
| `APIM_OPENAI_SUBSCRIPTION_KEY` | Sent as `api-key` and as `Ocp-Apim-Subscription-Key` |
| `LABGPT_LLM_BASE_URL` | OpenAI-compatible base, e.g. a vLLM or LiteLLM proxy |
| `LABGPT_LLM_API_KEY` | Key for that endpoint |
| `LABGPT_LLM_MODEL` | Model name for that endpoint |

## Using a private corpus

Create a directory with the same filenames as `data/sample` (see
[data/sample/README.md](data/sample/README.md) for the shape of each), then:

```bash
export LABGPT_DATA_DIR=/path/to/private/corpus
python -m labrag.cli index          # the index is built from the corpus, so rebuild it
python demo.py
```

`buildDB/build_sample_db.py` is needed only for the per-domain scripts, which query
SQLite rather than the index.

**Keep any real corpus outside this repository.** Nothing institutional belongs in version
control here.

## Known limitations

These are real and are worth stating plainly.

- **The abstention threshold is an operating point, not a validated number.** 0.62 is where
  accuracy peaks on the 101-question set, chosen by looking at the same questions it is
  scored on, so it will do worse on unseen ones. The answerable and unanswerable score
  distributions overlap, so no threshold separates them cleanly; the choice trades wrong
  refusals against wrong answers and cannot eliminate either. `labrag/answerer.py` carries
  the table.
- **Generated answers are not reproducible on the api path.** A reasoning deployment
  refuses any temperature but its default, so identical inputs can give different answers,
  and `seed` is documented as best effort. Any comparison between configurations has to
  treat the generated half as noisy.
- **The member relevance floor rejects the obviously unrelated and does no more.** On at
  least one question the right person scores below a wrong one on cosine; only the lexical
  leg of the fused ranking puts them first.
- **No test suite.** `pyproject.toml` points pytest at `tests/`, which does not exist yet.
  Retrieval quality is measured by `labrag.cli eval`; nothing else is.
- **The per-domain scripts are not on the retrieval path.** `protocolDemo.py`,
  `safetyDemo.py`, `memberInfoDemo.py` and `paperDemo.py` still stuff whole corpora into a
  prompt, as the whole pipeline once did.
- **`tools/` and `buildDB/` hold one-off ETL scripts** from the original ingest, some
  duplicated across both directories, some referencing input files that are not in the
  repository. They need consolidating.
- **The Azure path has been run against one gateway only.** Other deployments, other api
  versions and a direct `*.openai.azure.com` resource are assembled the same way but have
  not been exercised.

## Roadmap

The first five items are done: structure-aware chunking, dense plus BM25 retrieval fused
with RRF, validated citation markers, an abstention gate, and a 101-question labeled set
scored with recall@k, success@k, MRR and nDCG. What is left:

1. A `tests/` suite, starting with the citation and sentence-splitting logic, which is
   where the subtle bugs have been
2. A held-out question set, so the abstention threshold can be validated rather than fitted
3. Consolidate `tools/` and `buildDB/`
4. Put the per-domain scripts on the retrieval path, or retire them

## Layout

| Path | Purpose |
|---|---|
| `demo.py` | Main chat loop: retrieve, gate, generate, validate citations |
| `labgpt_config.py` | Paths, model, corpus location |
| `labgpt_metrics.py` | Per-call token and latency instrumentation |
| `labrag/` | Retrieval and answering: chunking, ingest, embeddings, index, BM25, gates, chat clients, CLI |
| `eval/questions.yaml` | 101 labeled questions, including deliberately unanswerable ones |
| `protocolDemo.py` | Protocol and reagent lookup over SQLite |
| `safetyDemo.py` | Safety corpus loading |
| `memberInfoDemo.py` | Team directory loading |
| `paperDemo.py`, `protocolPaperDemo.py` | Publication lookup |
| `search/` | Web search and page fetching |
| `buildDB/` | Corpus to SQLite ingest |
| `tools/` | One-off data preparation utilities |
| `data/sample/` | Synthetic corpus |
