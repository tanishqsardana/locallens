from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BaseEmbedder(ABC):
    model_name: str

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    """
    Lightweight fallback embedder.
    It preserves lexical similarity but is not as semantic as transformer models.
    """

    def __init__(self, dim: int = 384) -> None:
        self.model_name = f"hashing-{dim}"
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self._dim, dtype=np.float32)

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vec
        for token in tokens:
            vec += self._token_vector(token)
        return vec / len(tokens)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.vstack([self._embed_text(text) for text in texts]).astype(np.float32)
        return _l2_normalize(mat)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for transformer embeddings. "
                "Install with: pip install 'video-semantic-search[semantic]'"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


def build_embedder(kind: str, model: str | None = None) -> BaseEmbedder:
    normalized = kind.strip().lower()
    if normalized == "hashing":
        return HashingEmbedder()
    if normalized == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unsupported embedder: {kind}")

