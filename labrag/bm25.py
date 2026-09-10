"""Okapi BM25, implemented directly.

Hand-rolled rather than pulled from a library, for two reasons. It is about a hundred
lines, and more importantly the tokeniser has to be wrong in the right way for this
corpus. Library defaults strip numbers and punctuation, which on laboratory text throws
away the part that carries the answer.

Dense embeddings and BM25 fail in opposite directions, which is the whole argument for
running both:

  dense  handles paraphrase. "how cold do I spin it" finds "centrifuge at 4 C" without
         sharing a single word, and is hopeless at exact identifiers, mapping SMP-17104
         to a vague neighbourhood of "looks like a catalog number".

  BM25   handles exact tokens. SMP-17104, BSL-2, 100000 x g, 1:10,000, 0.22 um, and every
         cell line and antibody name match exactly or not at all, and it is blind to any
         phrasing that does not share vocabulary with the document.

Tokenisation choices that matter here, all of them departures from the usual default:

  Numbers are kept. "4", "37", "100000" and "2021" are frequently the answer.
  Compound tokens are kept whole *and* split: "smp-17104" also emits "smp" and "17104",
  so a query naming either half still matches.
  The stoplist is deliberately tiny. Dropping "not" would collapse "do not exceed" and
  "do exceed" into the same bag of terms, which in a safety corpus is unacceptable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Alphanumeric runs, allowing internal separators so identifiers survive intact:
# smp-17104, bsl-2, 0.22, 1:10,000, mg/ml
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.:/,][a-z0-9]+)*")
SPLIT_RE = re.compile(r"[-_.:/,]")

# Function words carrying no retrieval signal. Anything with polarity or negation stays,
# because dropping "not" would collapse "do not exceed" and "do exceed" into the same bag.
#
# This list has to be removed lexically rather than left to IDF. On a corpus of a few
# dozen chunks a function word that happens to appear in only a handful of them gets a
# high IDF and can carry a document to rank 1 on its own. That is not a hypothetical:
# "if" alone put the fire-safety and clogged-sink entries above the needlestick entry
# for the query "what should I do if I stick myself with a needle".
STOPWORDS = frozenset(
    """
    a an the of and or to in on at for from by with as is are was were be been being
    this that these those it its i you he she we they what which who whom how when where
    do does did done can could should would will shall may might must there their
    if then than so such about into onto over under again further here once
    me my myself your yours him her them our us
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, fold plurals, keep numbers, emit compounds both whole and split."""
    tokens: list[str] = []
    for match in TOKEN_RE.findall(text.lower()):
        _add(tokens, match)
        # "smp-17104" also contributes "smp" and "17104" so that a query naming
        # either half still matches the document.
        if SPLIT_RE.search(match):
            for part in SPLIT_RE.split(match):
                if part and part != match:
                    _add(tokens, part)
    return tokens


def _add(tokens: list[str], token: str) -> None:
    if token in STOPWORDS:
        return
    # Single characters are noise, except digits: "4 C" and "37 C" are real content.
    if len(token) < 2 and not token.isdigit():
        return
    tokens.append(singular(token))


def singular(token: str) -> str:
    """Conservative plural folding, so "needles" and "needle" are one term.

    Deliberately not a full stemmer. Porter would also strip verb and adjective endings,
    which over-stems domain vocabulary and mangles identifiers, and without an evaluation
    set there is nothing to justify that risk against. This handles the one case that
    demonstrably matters on this corpus and refuses the rest.

    The exclusions are what keep it safe: -ss (class), -us (status), -is (analysis) and
    anything containing a digit (an identifier) are left alone.
    """
    if len(token) <= 3 or any(ch.isdigit() for ch in token):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is", "as")):
        return token[:-1]
    return token


class BM25:
    """Okapi BM25 over an in-memory corpus.

    k1 controls how fast term frequency saturates, b how strongly length normalises.
    1.5 and 0.75 are the standard defaults and there is no reason to tune them before
    there is an evaluation set to tune against.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, tuple] = {}
        self._idf: dict[str, float] = {}
        self._doc_len = None
        self._avgdl = 1.0
        self._n_docs = 0

    def fit(self, documents: list[str]) -> "BM25":
        import numpy as np

        self._n_docs = len(documents)
        if self._n_docs == 0:
            raise ValueError("BM25.fit needs at least one document")

        lengths = np.zeros(self._n_docs, dtype="float32")
        raw: dict[str, list[tuple[int, int]]] = {}
        doc_freq: Counter[str] = Counter()

        for index, text in enumerate(documents):
            tokens = tokenize(text)
            lengths[index] = len(tokens)
            counts = Counter(tokens)
            for term, freq in counts.items():
                raw.setdefault(term, []).append((index, freq))
            doc_freq.update(counts.keys())

        # Freeze postings into arrays so scoring is vectorised.
        self._postings = {
            term: (
                np.fromiter((i for i, _ in plist), dtype="int64", count=len(plist)),
                np.fromiter((f for _, f in plist), dtype="float32", count=len(plist)),
            )
            for term, plist in raw.items()
        }
        self._doc_len = lengths
        mean = float(lengths.mean())
        self._avgdl = mean if mean > 0 else 1.0
        # Lucene's non-negative variant, which avoids negative weights for terms
        # appearing in more than half the corpus.
        self._idf = {
            term: math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        return self

    def score(self, query: str):
        """Return a score per document. Documents sharing no query term score zero."""
        import numpy as np

        scores = np.zeros(self._n_docs, dtype="float32")
        if self._n_docs == 0:
            return scores

        denom_norm = self.k1 * (1.0 - self.b + self.b * self._doc_len / self._avgdl)
        for term in tokenize(query):
            posting = self._postings.get(term)
            if posting is None:
                continue
            indices, freqs = posting
            idf = self._idf[term]
            scores[indices] += idf * (freqs * (self.k1 + 1.0)) / (
                freqs + denom_norm[indices]
            )
        return scores

    def top_k(self, query: str, k: int, allowed=None) -> list[tuple[int, float]]:
        """Top k by score, highest first. Zero-scoring documents are never returned.

        A document with no query term in it is not a weak match, it is not a match, and
        returning it would give the fusion step a rank to reward.
        """
        import numpy as np

        scores = self.score(query)
        if allowed is not None:
            mask = np.zeros(len(scores), dtype=bool)
            mask[list(allowed)] = True
            scores = np.where(mask, scores, 0.0)

        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0.0]
