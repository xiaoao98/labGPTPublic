# LabGPT

A laboratory knowledge assistant. It answers questions about lab protocols, reagents,
safety guidance, team responsibilities and the lab's own publications.

Indexing, retrieval and reranking run entirely on the machine holding the corpus, so
nothing leaves during search. Generation calls an OpenAI-compatible endpoint, which means
the handful of documents retrieved for a question do leave. Which endpoint that is decides
what a private corpus is exposed to, and is the one deployment choice this repository
cannot make for you.

Retrieval is hybrid: a local embedding model and a hand-written BM25, fused by reciprocal
rank. Every answer cites the documents it came from, every citation is checked against the
sources that were actually supplied, and the system refuses rather than guesses when
retrieval comes back weak.

The repository ships with a **synthetic corpus** in `data/sample`. Point it at a real
corpus to run it for real.

**Measured end to end on 100 labelled questions, answers graded by hand: 90% correct, or
93% with reranking on.** The numbers, and what they do and do not establish, are in
[Results](#results).

## How it works

```
query
  |
  +-- dense   top 20 by cosine over chunk embeddings
  +-- BM25    top 20 by exact term overlap
  |
  +-- reciprocal rank fusion            rank-based, not score-based
  |
  +-- cross-encoder rerank              restores what fusion discarded, GPU only
  |
  +-- expand chunks to documents        small-to-big, protocols served whole
  |
  +-- abstention gate                   cosine below threshold -> refuse, no model call
  |
  +-- generate with numbered sources
  |
  +-- citation validation               an answer with no valid citation is withheld
```

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

**End to end, 90 of 100 correct, or 93 with the reranker on.** Both were graded by
reading every answer.

| | answerable, 81 | unanswerable, 19 | total |
|---|---|---|---|
| default | 71 | 19 | **90 / 100** |
| with reranker | 74 | 19 | **93 / 100** |
| oracle context | 80 | not applicable | — |

Which of the two runs the default is decided by the hardware: reranking is on where it is
cheap and off where it is not, and `eval` prints which one produced a given table. The
nineteen unanswerable questions are refused either way, with nothing invented in either
run.

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

On by default where a GPU is present and off where it is not, since that is the whole of
the argument either way; `--rerank` and `--no-rerank` override.

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
costs about three times what answering does, which is why the default follows the
hardware. On a GPU the model should be a rounding error.

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
- Generation calls any OpenAI-compatible endpoint, so it needs no GPU. Reranking is the
  only step that wants one, and it runs on CPU if there is none

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv && .venv/bin/pip install -e .
```

There are no optional extras. Everything the retrieval and evaluation path needs is a
hard dependency, and the scripts in `tools/` use the standard library.

## Running it

Build the index first. It reads the corpus, chunks it, embeds the chunks locally and
writes `.index/`:

```bash
uv run python -m labrag.cli index
```

Then ask it something. This path needs an endpoint rather than a GPU:

```bash
uv run python -m labrag.cli selftest                     # one prompt, checks the endpoint
uv run python -m labrag.cli ask "how do I thaw BJ cells"
```

## Retrieval index

```bash
uv run python -m labrag.cli chunks    # inspect chunking only, no embedding, no GPU
uv run python -m labrag.cli index     # ingest, embed, write .index/
uv run python -m labrag.cli info      # describe an existing index
uv run python -m labrag.cli search "SMP-17104" --explain   # retrieve, no LLM
uv run python -m labrag.cli ask "how do I thaw BJ cells" --dry-run  # gate + prompt
uv run python -m labrag.cli eval --failures    # score retrieval against a question set
uv run python -m labrag.cli sweep              # compare retrieval configurations
uv run python -m labrag.cli eval --rerank      # with the cross-encoder pass
uv run python -m labrag.cli answer-all --questions ... --out ...   # batch, to JSONL
```

`chunks` is there so chunk boundaries can be iterated on without paying for embedding,
which is the slow part. An index is a copy of whatever it was built from, so none are
committed.

## Measuring cost per query

Every call records prompt tokens, generated tokens and wall time, and `answer-all` writes
them per question into the output JSONL alongside the answer.

Reasoning tokens are recorded separately from the answer. They are billed as completion
tokens while forming no part of the reply, and conflating the two is what made an early
run return a hundred empty answers with no error.

## Configuration

Corpus paths resolve through `labgpt_config.py`. Override with environment variables, or
copy `.env.example` to `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LABGPT_DATA_DIR` | `data/sample` | Corpus directory |
| `LABGPT_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `LABGPT_LLM_API_KEY` | unset | Key for that endpoint |
| `LABGPT_LLM_MODEL` | unset | Default model for the cli |

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
  score from 90 to 93 of 100 and the confidence intervals overlap. It is free on a GPU, so
  it is on, but the size of the gain is not established at n=100.
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
| `labgpt_config.py` | Corpus paths |
| `eval/` | Labelled question sets |
| `tools/` | Corpus construction: PMC fetch, clean, merge, and a dump of the lab's own SQLite |
| `data/sample/` | Synthetic corpus |

The prompt-stuffing assistant this grew out of, its per-domain demo scripts, its SQLite
build and its web-search branch were removed in `a2bc0ee` once the retrieval path replaced
them. `export_experiments_db.py` survives because the lab's database is still where the
real protocol text comes from, but nothing here builds one any more.
