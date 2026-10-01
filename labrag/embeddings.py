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

import os

# The name is resolved through the environment so the model can be loaded from a directory
# on disk instead of by hub id. Some institutional networks block huggingface.co outright
# (the TLS handshake is reset, which surfaces as a connection error rather than as a
# refusal), and on those machines the only way in is to fetch the model elsewhere and
# point at the copy. Keep it the same model: the index records which one built it, and
# retrieval scores, including the abstention threshold, are not comparable across models.
DEFAULT_MODEL = os.environ.get("LABGPT_EMBED_MODEL") or "BAAI/bge-small-en-v1.5"
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


def resolve_model(recorded: str | None) -> str:
    """The model to load, given the name an index recorded when it was built.

    LABGPT_EMBED_MODEL wins over the recorded name. That inversion is deliberate: the
    recorded name is a hub id, and the variable names a directory holding the same weights
    fetched by hand, which is the only way onto a machine whose network blocks the hub.
    Nothing in the files themselves can prove the two are the same model, so setting the
    variable is the assertion that they are, and the basenames are compared below to catch
    the obvious mistake rather than to verify the claim.
    """
    override = os.environ.get("LABGPT_EMBED_MODEL")
    if not override:
        return recorded or DEFAULT_MODEL
    if recorded and not _same_model_family(override, recorded):
        print(f"warning: LABGPT_EMBED_MODEL is {override}, but this index was built by "
              f"{recorded}.\n"
              f"  Loading the override anyway. Scores from a different embedding model are "
              f"not comparable\n"
              f"  with the vectors in the index, and retrieval will be quietly wrong "
              f"rather than broken.")
    return override


def _same_model_family(override: str, recorded: str) -> bool:
    """Whether a path and a hub id plausibly name the same model, by final path segment."""
    tail = override.rstrip("/").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return tail == recorded.rsplit("/", 1)[-1].lower()


def _looks_like_unreachable_hub(exc: Exception) -> bool:
    """Whether a load failure was the network rather than the model.

    A blocked hub arrives as OSError from transformers, LocalEntryNotFoundError from
    huggingface_hub, or a requests ConnectionError, depending on how far the download got
    before the reset. Matching on the message is unlovely but it is the one thing all
    three share, and the cost of a false positive is a slightly wrong hint on an error
    that was going to be raised anyway.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in (
        "couldn't connect", "could not connect", "connection", "offline",
        "localentrynotfound", "max retries", "name resolution", "timed out",
    ))


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
                # The original message is carried through rather than replaced. This also
                # fires when the package is installed but its own dependencies conflict
                # (transformers pinning huggingface-hub, most often), and "pip install
                # sentence-transformers" is useless advice for that, while the underlying
                # ImportError names the two versions that disagree.
                raise RuntimeError(
                    f"embeddings need a working sentence-transformers: {exc}\n"
                    f"  pip install -U sentence-transformers, or uv sync"
                ) from exc
            print(f"loading embedding model {self.model_name} ...")
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                if not _looks_like_unreachable_hub(exc):
                    raise
                raise RuntimeError(
                    f"could not load {self.model_name}: it is not in the local cache and "
                    f"huggingface.co could not be reached.\n"
                    f"  {type(exc).__name__}: {str(exc).splitlines()[0][:160]}\n"
                    f"\n"
                    f"  On a network that blocks the hub, fetch the model once from one "
                    f"that does not\n"
                    f"  and it stays cached in ~/.cache/huggingface:\n"
                    f"\n"
                    f"      python -c \"from sentence_transformers import "
                    f"SentenceTransformer as S; S('{self.model_name}')\"\n"
                    f"\n"
                    f"  Or copy the model directory over from another machine and point "
                    f"at it:\n"
                    f"\n"
                    f"      export LABGPT_EMBED_MODEL=/path/to/bge-small-en-v1.5\n"
                    f"\n"
                    f"  Use the same model that built the index; scores from different "
                    f"embedding\n"
                    f"  models are not comparable, and the abstention threshold is "
                    f"calibrated to this one."
                ) from exc
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def _uses_bge_prefix(self) -> bool:
        return "bge" in self.model_name.lower()

    def warm(self) -> None:
        """Load the model now, instead of inside whichever request arrives first.

        sentence-transformers loads on first use. In a chat loop that cost lands on the
        first question of the session; in a service it lands on one colleague, who
        experiences the whole system as slow for reasons that have nothing to do with
        their question. Measured on the HTTP path: 14.3 s for the first request against
        7.5 s for every one after it.

        One short encode is enough to force the load, and it exercises the tokenizer and
        the pooling layer as well, so a model directory that is missing a file fails here
        rather than on a question.
        """
        self.encode_query("warm up")

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
