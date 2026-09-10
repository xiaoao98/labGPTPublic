"""Build a larger, real corpus for evaluating retrieval, from PubMed Central Open Access.

Why this exists: the synthetic corpus in data/sample is 82 chunks. Retrieval metrics on a
corpus that small saturate, because almost any configuration puts the right document in
the top 5. Nothing can be discriminated, so nothing can be tuned. Evaluation needs a
corpus where the retriever has to actually choose.

What it fetches, and what it maps onto:

    research articles   ->  papers.json         title and abstract
                        ->  paper_content.json  methods and results sections
    STAR Protocols      ->  protocols.csv       step-by-step procedures
                        ->  reagents.csv        materials and equipment sections

Members and safety guidance stay synthetic. There is no good public source for either,
and scraping real people into a corpus is not something to do casually.

LICENSING, which is the reason this is a fetch script rather than committed data.

PMC Open Access articles carry a mix of licenses, and the mix matters. Roughly:

    CC BY, CC0      redistribution and reformatting both fine
    CC BY-NC        redistribution fine, commercial use restricted
    CC BY-NC-ND     ND means no derivatives, and reformatting an article into a corpus
                    is a derivative work, so this cannot be redistributed
    NO-CC CODE      no reuse grant at all

STAR Protocols, which is by far the best source of real step-by-step procedures, is
uniformly CC BY-NC-ND. So the corpus this builds is written to a gitignored directory and
never committed. What is committed is the manifest: the exact PMC ids, licenses, titles
and DOIs, which is enough for anyone to rebuild the identical corpus and is also the
provenance record.

The license of every document is recorded so that a CC BY only subset can be selected
later if a redistributable corpus is ever wanted.

Usage:
    uv run python tools/fetch_eval_corpus.py                 # default: ~120 articles
    uv run python tools/fetch_eval_corpus.py --articles 200 --protocols 60
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{}/unicode"

# NCBI allows three requests a second without an API key and answers 429 above that.
REQUEST_DELAY = 0.5
MAX_RETRIES = 4

# BioC coverage of PMC lags deposit by a year or two, and ids above roughly this value
# reliably answer "No result can be found". Skipping them client-side saves a request
# each, which matters when most candidates would otherwise be wasted lookups. It is a
# coverage heuristic, not a rule; raise it as BioC catches up.
BIOC_MAX_PMCID = 12_000_000

# Above this an "abstract" is a conference proceedings volume, not an abstract.
MAX_ABSTRACT_CHARS = 8_000

# Topics chosen to overlap the domains this assistant actually serves, and to overlap
# each other. Near-duplicate subject matter is what makes retrieval hard enough to
# measure; a corpus of unrelated topics would make every configuration look good.
ARTICLE_QUERIES = [
    "exosome isolation ultracentrifugation",
    "extracellular vesicle characterization",
    "pancreatic ductal adenocarcinoma microenvironment",
    "cancer associated fibroblast heterogeneity",
    "single cell RNA sequencing tumor stroma",
    "KRAS pancreatic cancer therapy",
    "tumor metastasis organotropism",
    "macrophage polarization tumor",
    "collagen extracellular matrix tumor",
    "flow cytometry immune profiling tumor",
]

PROTOCOL_QUERIES = [
    '"STAR Protocols"[jour] AND (exosome OR "extracellular vesicle")',
    '"STAR Protocols"[jour] AND (cell culture OR cryopreservation)',
    '"STAR Protocols"[jour] AND (flow cytometry OR immunofluorescence)',
    '"STAR Protocols"[jour] AND ("single cell" OR sequencing)',
    '"STAR Protocols"[jour] AND (western blot OR immunoprecipitation)',
    '"STAR Protocols"[jour] AND (mouse OR xenograft OR organoid)',
]

# BioC coverage lags publication by a year or more, so recent PMC ids resolve to
# "No result can be found". Restricting the search window avoids wasting requests.
DATE_RANGE = "2015:2021[pdat]"

REAGENT_SECTION_RE = re.compile(
    r"materials and equipment|key resources|reagents? and|equipment setup", re.IGNORECASE
)


def _get(url: str, params: dict | None = None) -> str:
    """GET with backoff on 429. NCBI throttles hard and answers 429 rather than queuing."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "labgpt-eval-corpus/0.1"})

    delay = REQUEST_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            delay *= 2
            print(f"      429, backing off {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def search(term: str, retmax: int) -> list[str]:
    body = _get(
        ESEARCH,
        {"db": "pmc", "retmode": "json", "retmax": retmax, "term": f"{term} AND {DATE_RANGE}"},
    )
    return json.loads(body).get("esearchresult", {}).get("idlist", [])


def fetch_bioc(pmc_id: str) -> dict | None:
    """Return the BioC document, or None when PMC has no open-access full text for it."""
    try:
        body = _get(BIOC.format(pmc_id))
    except Exception as exc:
        print(f"    PMC{pmc_id}: fetch failed, {exc}")
        return None
    if body.lstrip().startswith("[Error]"):
        return None
    try:
        collection = json.loads(body)
    except json.JSONDecodeError:
        return None
    documents = collection[0].get("documents") if collection else None
    return documents[0] if documents else None


# --- parsing ------------------------------------------------------------------


def _passages(document: dict, section_type: str) -> list[dict]:
    return [
        p for p in document.get("passages", [])
        if p.get("infons", {}).get("section_type") == section_type
    ]


def _text_of(passages: list[dict]) -> str:
    return "\n".join(p.get("text", "").strip() for p in passages if p.get("text", "").strip())


def parse_article(document: dict) -> dict | None:
    """Pull title, abstract, methods and results out of a BioC document."""
    title_passages = _passages(document, "TITLE")
    title = title_passages[0].get("text", "").strip() if title_passages else ""
    abstract = _text_of(_passages(document, "ABSTRACT"))
    methods = _text_of(_passages(document, "METHODS"))
    results = _text_of(_passages(document, "RESULTS"))

    if not title or not abstract or len(abstract) < 200:
        return None
    # Conference proceedings concatenate hundreds of unrelated abstracts into one
    # ABSTRACT section. They are not useful evaluation documents and one of them alone
    # contributed a 17,000 token "abstract" to an earlier build.
    if len(abstract) > MAX_ABSTRACT_CHARS:
        return None
    if len(methods) < 400 and len(results) < 400:
        return None

    infons = document.get("infons", {})
    return {
        "pmc_id": document.get("id", ""),
        "title": title,
        "abstract": abstract,
        "method": methods,
        "result": results,
        "license": infons.get("license", "unknown"),
        "doi": infons.get("article-id_doi", ""),
        "journal": infons.get("journal", ""),
        "year": infons.get("year", ""),
    }


def parse_protocol(document: dict) -> dict | None:
    """Reconstruct a step-by-step procedure and a reagent list from a protocol article.

    BioC strips the "1." "2." markers from the source, so numbering cannot be read from
    the text. The structure survives in the passage hierarchy instead: a title_2 heading
    opens a stage, and the paragraphs beneath it are its steps. Numbering is rebuilt from
    that order, which is what gives the chunker real multi-step procedures to work on.
    """
    title_passages = _passages(document, "TITLE")
    title = title_passages[0].get("text", "").strip() if title_passages else ""
    if not title:
        return None

    methods = _passages(document, "METHODS")
    if not methods:
        return None

    stages: list[tuple[str, list[str]]] = []
    reagent_lines: list[str] = []
    current_heading, current_steps = None, []
    in_reagent_section = False

    for passage in methods:
        kind = passage.get("infons", {}).get("type", "")
        text = passage.get("text", "").strip()
        if not text:
            continue

        if kind in ("title_1", "title_2"):
            if current_heading and current_steps:
                stages.append((current_heading, current_steps))
            in_reagent_section = bool(REAGENT_SECTION_RE.search(text))
            current_heading, current_steps = (None, []) if in_reagent_section else (text, [])
            continue

        if in_reagent_section:
            reagent_lines.append(text)
        elif current_heading:
            current_steps.append(text)

    if current_heading and current_steps:
        stages.append((current_heading, current_steps))

    # Keep only stages substantial enough to be a real procedure.
    stages = [(h, s) for h, s in stages if len(s) >= 3]
    if not stages:
        return None

    infons = document.get("infons", {})
    return {
        "pmc_id": document.get("id", ""),
        "title": title,
        "stages": stages,
        "reagents": reagent_lines,
        "license": infons.get("license", "unknown"),
        "doi": infons.get("article-id_doi", ""),
        "journal": infons.get("journal", ""),
        "year": infons.get("year", ""),
    }


def render_steps(steps: list[str]) -> str:
    """Rebuild numbering that BioC dropped, so the chunker sees real numbered steps."""
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))


# --- collection ---------------------------------------------------------------


def collect(queries: list[str], want: int, parser, label: str) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    # Most candidates are rejected, either because BioC has no full text or because the
    # article is too thin, so the pool has to be several times the target.
    per_query = max(60, (want * 18) // max(len(queries), 1))
    skipped_coverage = tried = 0

    for query in queries:
        if len(out) >= want:
            break
        try:
            ids = search(query, per_query)
        except Exception as exc:
            print(f"  search failed for {query!r}: {exc}")
            continue
        usable = [i for i in ids if i.isdigit() and int(i) <= BIOC_MAX_PMCID]
        skipped_coverage += len(ids) - len(usable)
        print(f"  {label}: {query[:56]!r} -> {len(ids)} found, {len(usable)} in BioC range")

        for pmc_id in usable:
            if len(out) >= want or pmc_id in seen:
                continue
            seen.add(pmc_id)
            tried += 1
            time.sleep(REQUEST_DELAY)
            document = fetch_bioc(pmc_id)
            if document is None:
                continue
            parsed = parser(document)
            if parsed is None:
                continue
            out.append(parsed)
            print(f"    + PMC{pmc_id}  {parsed['license']:<18} {parsed['title'][:58]}")

    print(f"  {label}: kept {len(out)} of {tried} fetched "
          f"({skipped_coverage} skipped as outside BioC coverage)")
    return out


# --- writing ------------------------------------------------------------------


def write_corpus(out_dir: Path, articles: list[dict], protocols: list[dict],
                 sample_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = {
        a["pmc_id"]: {"title": a["title"], "abstract": a["abstract"]} for a in articles
    }
    (out_dir / "papers.json").write_text(
        json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    contents = {
        a["pmc_id"]: {"title": a["title"], "method": a["method"], "result": a["result"]}
        for a in articles
    }
    (out_dir / "paper_content.json").write_text(
        json.dumps(contents, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (out_dir / "protocols.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for protocol in protocols:
            for heading, steps in protocol["stages"]:
                name = f"{heading} ({protocol['pmc_id']})"
                writer.writerow([name, render_steps(steps)])

    with (out_dir / "reagents.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for protocol in protocols:
            if not protocol["reagents"]:
                continue
            name = f"{protocol['title'][:70]} ({protocol['pmc_id']})"
            writer.writerow([name, render_steps(protocol["reagents"])])

    # Members and safety are not fetched. Copy the synthetic versions across so the
    # eval corpus is a complete corpus and the indexer needs no special casing.
    for filename in ("members.tsv", "safety.json"):
        source = sample_dir / filename
        if source.exists():
            (out_dir / filename).write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "PubMed Central Open Access, via the BioC API",
        "note": (
            "Corpus content is NOT committed. Several sources are CC BY-NC-ND, whose "
            "no-derivatives term this reformatting would breach. Rebuild it with "
            "tools/fetch_eval_corpus.py; the ids below pin the exact set."
        ),
        "counts": {
            "articles": len(articles),
            "protocol_articles": len(protocols),
            "protocol_stages": sum(len(p["stages"]) for p in protocols),
        },
        "licenses": _count(a["license"] for a in articles + protocols),
        "articles": [
            {k: a[k] for k in ("pmc_id", "license", "doi", "journal", "year", "title")}
            for a in articles
        ],
        "protocols": [
            {k: p[k] for k in ("pmc_id", "license", "doi", "journal", "year", "title")}
            for p in protocols
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=int, default=120)
    parser.add_argument("--protocols", type=int, default=40)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "eval")
    parser.add_argument("--sample-dir", type=Path, default=REPO_ROOT / "data" / "sample")
    args = parser.parse_args(argv)

    print(f"fetching up to {args.articles} articles and {args.protocols} protocols")
    print("this makes a few hundred requests and takes a while\n")

    articles = collect(ARTICLE_QUERIES, args.articles, parse_article, "articles")
    print()
    protocols = collect(PROTOCOL_QUERIES, args.protocols, parse_protocol, "protocols")

    if not articles:
        print("\nno articles fetched; nothing written")
        return 1

    manifest = write_corpus(args.out, articles, protocols, args.sample_dir)
    print(f"\nwrote corpus to {args.out}")
    print(f"  articles          {manifest['counts']['articles']}")
    print(f"  protocol articles {manifest['counts']['protocol_articles']}")
    print(f"  protocol stages   {manifest['counts']['protocol_stages']}")
    print("  licenses:")
    for name, count in manifest["licenses"].items():
        print(f"    {name:<16} {count}")
    print(f"\nindex it with:\n  LABGPT_DATA_DIR={args.out} uv run python -m labrag.cli index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
