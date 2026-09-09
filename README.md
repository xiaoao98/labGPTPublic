# LabGPT

A laboratory knowledge assistant. It answers questions about lab protocols, reagents,
safety guidance, team responsibilities, and the lab's own publications, running a local
open-weights model so that no institutional data leaves the machine.

The repository ships with a **synthetic corpus** in `data/sample`. Point it at a private
corpus to run it for real.

> **Status: prompt stuffing, not retrieval.** The current pipeline classifies a query and
> then injects whole documents into the context window. There is no embedding index and no
> similarity search yet. Retrieval is the next piece of work; see [Roadmap](#roadmap).
> The README used to claim RAG, which was not accurate.

## How it works today

```
query
  |
  +-- classify: does this need web search?          (LLM call)
  +-- classify: does this need the paper corpus?    (LLM call)
  +-- classify: which data domains?                 (LLM call)
  |
  +-- safety      -> inject the whole safety corpus
  +-- protocol    -> LLM picks an experiment name, SQL fetches that row
  +-- papers      -> LLM picks 5 paper IDs from the whole abstract index
  +-- web         -> DuckDuckGo search, fetch and summarize pages
  +-- members     -> inject the whole directory (unconditionally)
  |
  +-- assemble one large prompt -> generate answer
```

## Requirements

- Python 3.10 or newer
- A GPU with enough memory for the configured model. The default is `Qwen/Qwen3-32B`;
  set `LABGPT_MODEL` to something smaller to run on modest hardware.

```bash
pip install -r requirements.txt
```

## Running it

```bash
python buildDB/build_sample_db.py    # generates experiments.db from the corpus CSVs
python demo.py
```

The individual domains can be run on their own: `protocolDemo.py`, `safetyDemo.py`,
`memberInfoDemo.py`, `paperDemo.py`.

## Configuration

Everything resolves through `labgpt_config.py`. Override with environment variables, or
copy `.env.example` to `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LABGPT_DATA_DIR` | `data/sample` | Corpus directory |
| `LABGPT_DB_PATH` | `experiments.db` | Generated SQLite database |
| `LABGPT_MODEL` | `Qwen/Qwen3-32B` | Hugging Face model id |
| `LABGPT_ASSISTANT_NAME` | `LabGPT` | Name shown in the chat prompt |
| `GOOGLE_API_KEY` | unset | Optional, for `search/callGoogleAPI.py` |
| `GOOGLE_SEARCH_ENGINE_ID` | unset | Optional, for `search/callGoogleAPI.py` |

## Using a private corpus

Create a directory with the same filenames as `data/sample` (see
[data/sample/README.md](data/sample/README.md) for the shape of each), then:

```bash
export LABGPT_DATA_DIR=/path/to/private/corpus
python buildDB/build_sample_db.py
python demo.py
```

**Keep any real corpus outside this repository.** Nothing institutional belongs in version
control here.

## Known limitations

These are real and are worth stating plainly.

- **No retrieval.** Whole documents go into the prompt. A paper query currently assembles
  tens of thousands of tokens of context before generation starts, and the paper "search"
  asks the model to rank the entire abstract index in context, which is O(corpus) per query
  and cannot be scored or thresholded.
- **The directory is injected on every query,** including ones that have nothing to do with
  people.
- **No grounding guarantees.** Answers carry no citations, and there is no mechanism to
  abstain when the context does not contain the answer. For safety questions that matters.
- **No tests, no evaluation set.** Retrieval quality is currently unmeasured.
- **`tools/` and `buildDB/` hold one-off ETL scripts** from the original ingest, some
  duplicated across both directories, some referencing input files that are not in the
  repository. They need consolidating.

## Roadmap

1. Chunk the corpora with structure awareness, so a protocol step is never split from its
   warning
2. Local embeddings plus a lexical BM25 leg, fused with reciprocal rank fusion
3. Citation markers on retrieved chunks, validated after generation
4. An abstention gate in front of generation for low-confidence retrieval
5. A labeled evaluation set with recall@k and nDCG reported per configuration

## Layout

| Path | Purpose |
|---|---|
| `demo.py` | Main chat loop, routes across all domains |
| `labgpt_config.py` | Paths, model, corpus location |
| `protocolDemo.py` | Protocol and reagent lookup over SQLite |
| `safetyDemo.py` | Safety corpus loading |
| `memberInfoDemo.py` | Team directory loading |
| `paperDemo.py`, `protocolPaperDemo.py` | Publication lookup |
| `search/` | Web search and page fetching |
| `buildDB/` | Corpus to SQLite ingest |
| `tools/` | One-off data preparation utilities |
| `data/sample/` | Synthetic corpus |
