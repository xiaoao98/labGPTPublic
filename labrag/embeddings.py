"""Local embeddings.

An embedding turns a piece of text into a fixed-length vector of numbers, positioned so
that texts meaning similar things land near each other. That is what lets "how cold do I
spin it" find a chunk that says "centrifuge at 4 C" without sharing a single word.

The default model is BAAI/bge-small-en-v1.5: 33M parameters, 384 dimensions, runs fine on
CPU. It downloads once (about 130 MB) and then works with no network. That matters here,
because the point of running an open-weights model at all is that institutional text never
leaves the machine, and calling a hosted embedding API would quietly undo that.

Two details that are easy to get wrong and expensive to debug:

  Normalisation. Vectors are scaled to unit length, so a dot product between two of them
  is exactly their cosine similarity, always between -1 and 1. Without this, similarity
  scores drift with text length and no fixed abstention threshold can hold.

  The query prefix. bge models are trained asymmetrically: documents are embedded bare,
  but queries are supposed to carry an instruction prefix. Skipping it costs real recall.
  It is applied automatically for bge models and skipped for others.
"""

from __future__ import annotations

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def normalize(matrix):
    """Scale each row to unit length so a dot product is a cosine similarity."""
    import numpy as np

    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector would produce NaN and poison every later comparison.
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype("float32")


class Embedder:
    """Wraps a sentence-transformers model. Loads it once, on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "embeddings need sentence-transformers.\n"
                    "  pip install sentence-transformers"
                ) from exc
            print(f"loading embedding model {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def _uses_bge_prefix(self) -> bool:
        return "bge" in self.model_name.lower()

    def encode_documents(self, texts):
        """Embed corpus chunks. No prefix; documents are embedded bare."""
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return normalize(vectors)

    def encode_query(self, text: str):
        """Embed one query, with the instruction prefix where the model expects it."""
        prefixed = (BGE_QUERY_PREFIX + text) if self._uses_bge_prefix else text
        return normalize(self.model.encode([prefixed], convert_to_numpy=True))[0]
