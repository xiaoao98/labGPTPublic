"""A cross-encoder reranking pass over fused candidates.

The first stage is a bi-encoder: documents are embedded when the index is built, queries
are embedded at search time, and the two never meet until a dot product. That is what
makes it fast enough to score 3,020 chunks in one matrix multiply, and it is also its
limit, because the model encoding a document has no idea what will be asked of it.

A cross-encoder reads the query and the document together and scores the pair. It can see
that "4 C" in this sentence is what "how cold" in the question is asking for. The price is
one forward pass per candidate, so it can only run over a shortlist.

WHY THIS CORPUS WANTS ONE.

Measured over the 81 answerable questions, every gold document is somewhere in the fused
candidate list: success reaches 1.000 by rank 30 and nothing is missing at 40. At the six
documents actually served it is 0.938. Recall is not the problem; ordering is. The five
questions that fail have their gold document sitting at ranks 7, 9, 25, 26 and 29.

Reciprocal rank fusion is why. It reads ranks and discards scores, so a chunk both legs
place in their top handful beats a chunk one leg is certain about. The needlestick case in
retriever.py is the demonstration: the right entry has cosine 0.710 against a wrong one's
0.498 and still comes second. A reranker restores exactly the signal fusion threw away.

SCORES ARE RAW LOGITS, AND ONLY ORDERING IS MEANT BY THEM.

The model is loaded through transformers rather than through sentence-transformers'
CrossEncoder, because that wrapper applies an activation chosen from the model config and
the choice differs between models: bge-reranker's single logit comes back through a
sigmoid, which maps the useful range of -8 to -10 onto 0.0002 to 0.00004 and looks like no
signal at all. Raw logits are monotonic in relevance, which is all a sort needs.

They are not calibrated across queries. A logit of -8 can be the best answer to one
question and a poor one to another, so these must not be compared to a fixed threshold the
way cosine is in the abstention gate. Doing that is the same mistake as thresholding on a
fusion score.
"""

from __future__ import annotations

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """Scores (query, passage) pairs with a cross-encoder. Loads on first use."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, batch_size: int = 32,
                 max_length: int = 512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment problem, not logic
            raise RuntimeError(
                "reranking needs transformers.\n  pip install transformers"
            ) from exc
        print(f"loading reranker {self.model_name} ...")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance logit per passage, in the order given."""
        if not passages:
            return []
        import torch

        self._load()
        out: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            encoded = self._tok([query] * len(batch), batch, padding=True,
                                truncation=True, max_length=self.max_length,
                                return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**encoded).logits.view(-1)
            out.extend(float(x) for x in logits)
        return out
