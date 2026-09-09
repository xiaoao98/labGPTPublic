# Sample corpus

This directory is **mixed**. Read the table below before assuming anything here is
invented.

`members.tsv` and `memberinfo.txt` hold **real people**: 37 named members of a real
research laboratory, with biographies taken from the laboratory's public web page. Every
other file is synthetic.

The synthetic files describe the Vance Lab at the Northwind Institute for Biomedical
Research, which does not exist. Their people, protocols, reagents, papers, phone numbers,
and email addresses are invented; phone numbers use the 555-01xx range reserved for
fiction and addresses use `example.edu`, a reserved documentation domain.

The synthetic scientific content is plausible but is **not** validated laboratory
guidance. Do not follow any protocol or safety instruction in this directory. It is
placeholder text shaped to exercise the retrieval and chunking code.

## Files

| File | Real or synthetic | Shape |
|---|---|---|
| `members.tsv` | **real people** | `name<TAB>role<TAB>bio` per line |
| `memberinfo.txt` | **real people** | raw source for `members.tsv`, one long tab-run |
| `safety.json` | synthetic | `[{instruction, input, output}]` |
| `protocols.csv` | synthetic | `experiment,content` |
| `reagents.csv` | synthetic | `experiment,content` |
| `papers.json` | synthetic | `{id: {title, abstract}}` |
| `paper_content.json` | synthetic | `{id: {title, method, result}}` |

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
