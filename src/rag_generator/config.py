"""Application configuration sourced from environment variables and CLI flags."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,509}[A-Za-z0-9]$")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Validated settings shared by ingestion and question answering."""

    chroma_path: Path
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    top_k: int
    min_similarity: float
    groq_model: str
    max_completion_tokens: int

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            chroma_path=Path(os.getenv("RAG_CHROMA_PATH", ".chroma")),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "rag-documents"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 800),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 120),
            embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 32),
            top_k=_env_int("RAG_TOP_K", 4),
            min_similarity=_env_float("RAG_MIN_SIMILARITY", 0.35),
            groq_model=os.getenv("RAG_GROQ_MODEL", "openai/gpt-oss-120b"),
            max_completion_tokens=_env_int("RAG_MAX_COMPLETION_TOKENS", 800),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")
        if not self.groq_model.strip():
            raise ValueError("groq_model cannot be empty")
        if self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be at least 1")
        if not _COLLECTION_NAME.fullmatch(self.collection_name):
            raise ValueError(
                "collection_name must be 3-512 characters, start and end with "
                "an alphanumeric character, and contain only letters, numbers, "
                "periods, underscores, or hyphens"
            )
