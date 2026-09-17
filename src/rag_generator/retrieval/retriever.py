"""Top-k cosine-similarity retrieval with an explicit relevance threshold."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_generator.indexing.vector_store import VectorSearchResult


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class SearchStore(Protocol):
    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    similarity: float
    page: int | None = None

    @property
    def citation(self) -> str:
        location = f"{self.source}, chunk {self.chunk_index}"
        if self.page is not None:
            location = f"{self.source}, page {self.page}, chunk {self.chunk_index}"
        return f"{location} ({self.id})"


class SimilarityRetriever:
    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_store: SearchStore,
        top_k: int = 4,
        min_similarity: float = 0.35,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        query_embedding = self.embedder.embed_query(question)
        records = self.vector_store.query(query_embedding, self.top_k)
        retrieved = [self._to_retrieved_chunk(record) for record in records]
        return [
            chunk
            for chunk in retrieved
            if chunk.similarity >= self.min_similarity
        ]

    @staticmethod
    def _to_retrieved_chunk(record: VectorSearchResult) -> RetrievedChunk:
        metadata = record.metadata
        try:
            source = str(metadata["source"])
            chunk_index = int(metadata["chunk_index"])
            page_value = metadata.get("page")
            page = int(page_value) if page_value is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Index record {record.id!r} has invalid citation metadata") from exc

        # Collections are created with cosine distance: 0 means identical and
        # 1 means orthogonal, so cosine similarity is one minus the distance.
        similarity = max(-1.0, min(1.0, 1.0 - record.distance))
        return RetrievedChunk(
            id=record.id,
            text=record.document,
            source=source,
            page=page,
            chunk_index=chunk_index,
            similarity=similarity,
        )

