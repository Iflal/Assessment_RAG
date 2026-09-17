"""Sentence Transformers embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class EmbeddingError(RuntimeError):
    """Raised when the local embedding model cannot produce vectors."""


class MiniLMEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, show_progress=len(texts) > self.batch_size)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingError("Question cannot be empty")
        vectors = self._embed([text], show_progress=False)
        return vectors[0]

    def _embed(self, texts: Sequence[str], show_progress: bool) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = self._get_model()
            vectors = model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vectors.tolist()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Embedding failed while processing text. Check the model installation "
                "and available memory, then try again."
            ) from exc

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EmbeddingError(
                "Embedding support requires sentence-transformers. "
                "Install project dependencies first."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model {self.model_name!r}. Check internet "
                "access or the local Hugging Face model cache."
            ) from exc
        return self._model
