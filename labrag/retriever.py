"""Hybrid retrieval: two legs, fused by rank, then expanded to documents.

    query
      ├── dense  top N   (cosine over embeddings)
      └── BM25   top N   (exact term overlap)
              │
              ▼
         RRF fusion              rank-based, not score-based
              │
              ▼
       expand to documents       small-to-big, see documents.py
              │
              ▼
        Retrieved[]

Why Reciprocal Rank Fusion rather than a weighted sum of the two scores: cosine
similarity and BM25 are not on a comparable scale and never will be. Cosine is bounded in
[-1, 1] and clusters tightly; BM25 is unbounded and depends on corpus statistics. Any
fixed weighting between them is a constant that has to be re-tuned every time the corpus
or the embedding model changes. RRF reads only the ranks, so it has one parameter and
survives both.

    rrf(chunk) = sum over legs of  weight_leg / (k + rank_in_that_leg)

k defaults to 60, the value from the original paper. It flattens the difference between
the top few ranks, so a chunk that both legs place in their top handful beats a chunk that
one leg loves and the other has never heard of.

That last property is usually what you want and sometimes exactly wrong, so it is worth
stating the failure mode plainly. Because RRF reads only ranks, appearing in both lists
beats being first in one:

    query "what should I do if I stick myself with a needle"
      the sharps-container entry:  bm25 rank 1 (1/61) + dense rank 12 (1/72) = 0.0303
      the needlestick entry:       dense rank 1 (1/61), no bm25 match        = 0.0164

The needlestick entry is the right answer and has a cosine of 0.710 against the sharps
entry's 0.498, but fusion discards that difference and ranks it second. BM25 matched the
sharps entry on the single term "needle".

Hybrid is therefore not uniformly better than dense here. Over an eleven-query spread
across identifiers, paraphrases, people and cross-document questions, the lexical leg was
decisive three times and actively harmful twice, and dense and hybrid disagreed on the top
result in seven of eleven:

    decisive   SMP-17104, "100000 x g", "how long can the vial stay in the water bath"
               (dense missed the last one entirely, ranking a waste-disposal entry first)
    harmful    the needlestick query above, and "who knows flow cytometry", where BM25
               correctly found the person who runs the flow panels and fusion still
               preferred dense's paper about flow cytometry

Leg weights exist so this can be corrected, and default to 1.0 each, which is standard
unweighted RRF. They are deliberately left at the default. Eleven unlabelled queries are
not something to tune on, and measurement shows weighting is the wrong tool for this
failure anyway: flipping the needlestick case needs the lexical weight below about 0.15,
which is indistinguishable from switching BM25 off, and that would give back the three
cases where it was decisive.

The real fix is almost certainly a reranking pass. Fusion throws away the calibrated
difference between a 0.710 match and a 0.498 one; a cross-encoder scoring query and
document together restores exactly that. Settling this is the first job of the evaluation
set in Phase 4, which is what should choose between weights, reranking, and leaving it be.

The raw cosine is carried through fusion untouched. RRF scores encode rank, not
similarity, so the top hit of a search that found nothing relevant still gets a high RRF
score. Abstention in Phase 3 has to threshold on the cosine, never on the fused score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25
from .chunker import Chunk
from .documents import Retrieved, expand_hits

DEFAULT_RRF_K = 60


@dataclass
class ChunkHit:
    """One chunk that survived fusion, with where each leg placed it."""

    chunk: Chunk
    fused_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None

    @property
    def found_by(self) -> str:
        if self.dense_rank is not None and self.lexical_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        return "bm25"

    @property
    def confidence(self) -> float:
        """The calibrated signal for abstention. Cosine, never the fused score.

        A chunk found only by BM25 has no cosine, so it reports 0.0 here rather than
        inventing one. Phase 3 decides what to do about that.
        """
        return self.dense_score if self.dense_score is not None else 0.0


@dataclass
class RetrievalResult:
    query: str
    mode: str
    documents: list[Retrieved] = field(default_factory=list)
    chunk_hits: list[ChunkHit] = field(default_factory=list)

    @property
    def best_cosine(self) -> float:
        return max((h.confidence for h in self.chunk_hits), default=0.0)


class Retriever:
    def __init__(
        self,
        store,
        embedder=None,
        mode: str = "hybrid",
        top_k_dense: int = 20,
        top_k_lexical: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
        final_k: int = 6,
        weight_dense: float = 1.0,
        weight_lexical: float = 1.0,
    ):
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"mode must be dense, bm25 or hybrid, got {mode!r}")
        if mode in ("dense", "hybrid") and embedder is None:
            raise ValueError(f"mode={mode} needs an embedder")

        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.top_k_dense = top_k_dense
        self.top_k_lexical = top_k_lexical
        self.rrf_k = rrf_k
        self.final_k = final_k
        self.weight_dense = weight_dense
        self.weight_lexical = weight_lexical

        # The lexical index is built over the same text that was embedded, breadcrumb
        # included, so both legs see identical inputs and their ranks are comparable.
        self._bm25 = None
        if mode in ("bm25", "hybrid"):
            self._bm25 = BM25().fit([c.embed_text for c in store.chunks])

    # -- legs --

    def _dense(self, query: str, doc_types=None):
        vector = self.embedder.encode_query(query)
        return self.store.search_dense(vector, self.top_k_dense, doc_types=doc_types)

    def _lexical(self, query: str, doc_types=None):
        allowed = None
        if doc_types:
            wanted = set(doc_types)
            allowed = [i for i, c in enumerate(self.store.chunks) if c.doc_type in wanted]
        return self._bm25.top_k(query, self.top_k_lexical, allowed=allowed)

    # -- fusion --

    def _fuse(self, dense, lexical) -> list[ChunkHit]:
        dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
        dense_score = dict(dense)
        lex_rank = {i: r for r, (i, _) in enumerate(lexical, start=1)}
        lex_score = dict(lexical)

        fused: dict[int, float] = {}
        for index, rank in dense_rank.items():
            fused[index] = fused.get(index, 0.0) + self.weight_dense / (self.rrf_k + rank)
        for index, rank in lex_rank.items():
            fused[index] = fused.get(index, 0.0) + self.weight_lexical / (self.rrf_k + rank)

        hits = [
            ChunkHit(
                chunk=self.store.chunks[index],
                fused_score=score,
                dense_rank=dense_rank.get(index),
                dense_score=dense_score.get(index),
                lexical_rank=lex_rank.get(index),
                lexical_score=lex_score.get(index),
            )
            for index, score in fused.items()
        ]
        # Tie-break on cosine then on index so ordering is deterministic across runs,
        # which matters once there is an evaluation set.
        hits.sort(key=lambda h: (-h.fused_score, -h.confidence, h.chunk.chunk_id))
        return hits

    # -- entry point --

    def retrieve(self, query: str, k: int | None = None, doc_types=None,
                 include_linked: bool = True) -> RetrievalResult:
        k = k or self.final_k

        dense = self._dense(query, doc_types) if self.mode in ("dense", "hybrid") else []
        lexical = self._lexical(query, doc_types) if self.mode in ("bm25", "hybrid") else []

        hits = self._fuse(dense, lexical)

        # Expand a few more chunks than requested, because several chunks of one
        # document collapse into a single result and would otherwise short the list.
        documents = expand_hits(
            [(h.chunk, h.fused_score) for h in hits[: max(k * 3, self.top_k_dense)]],
            self.store.documents,
            include_linked=include_linked,
            limit=k,
        )
        return RetrievalResult(
            query=query, mode=self.mode, documents=documents, chunk_hits=hits[:k]
        )
