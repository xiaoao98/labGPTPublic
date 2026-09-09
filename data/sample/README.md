# Sample corpus

Everything in this directory is **synthetic**. It exists so the assistant can be run,
tested, and demonstrated without any real institutional data.

The Vance Lab at the Northwind Institute for Biomedical Research does not exist. The
people, protocols, reagents, papers, phone numbers, and email addresses are invented.
Phone numbers use the 555-01xx range reserved for fiction, and email addresses use
`example.edu`, a reserved documentation domain.

The scientific content is plausible but is **not** validated laboratory guidance. Do not
follow any protocol or safety instruction in this directory. It is placeholder text shaped
to exercise the retrieval and chunking code.

## Files

| File | Shape | Replaces |
|---|---|---|
| `members.tsv` | `name<TAB>role<TAB>bio` per line | the lab directory |
| `safety.json` | `[{instruction, input, output}]` | the safety Q&A set |
| `protocols.csv` | `experiment,content` | the protocol table |
| `reagents.csv` | `experiment,content` | the reagent table |
| `papers.json` | `{id: {title, abstract}}` | the publication abstract index |
| `paper_content.json` | `{id: {title, method, result}}` | full paper method and result sections |

## Building the database

The SQLite database is generated, not committed:

```bash
python buildDB/build_sample_db.py
```

That writes `experiments.db` with the `protocols` and `reagents` tables the demos query.

## Pointing at a real corpus

Nothing in the code hardcodes these paths any more. Set them in the environment or copy
`.env.example` to `.env`:

```
LABGPT_DATA_DIR=/path/to/your/private/corpus
```

Keep any real corpus outside this repository.
