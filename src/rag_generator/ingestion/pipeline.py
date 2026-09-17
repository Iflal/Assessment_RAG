"""Orchestration for load -> chunk -> embed -> persist."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_generator.ingestion.chunking import TextChunker
from rag_generator.ingestion.loaders import load_documents
from rag_generator.ingestion.models import TextChunk


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def replace(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> str: ...


@dataclass(frozen=True)
class IngestionReport:
    source: Path
    loaded_files: tuple[str, ...]
    skipped_files: tuple[str, ...]
    warnings: tuple[str, ...]
    chunk_count: int
    active_collection: str


class IngestionService:
    def __init__(
        self,
        chunker: TextChunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, source: Path) -> IngestionReport:
        loaded = load_documents(source)
        chunks = self.chunker.chunk_documents(loaded.documents)
        if not chunks:
            raise ValueError("Documents were loaded but no non-empty chunks were produced")

        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks"
            )
        active_collection = self.vector_store.replace(chunks, embeddings)
        return IngestionReport(
            source=source.resolve(),
            loaded_files=loaded.loaded_files,
            skipped_files=loaded.skipped_files,
            warnings=loaded.warnings,
            chunk_count=len(chunks),
            active_collection=active_collection,
        )

