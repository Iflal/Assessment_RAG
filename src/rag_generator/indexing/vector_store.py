"""Persistent Chroma index with safe active-index switching."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_generator.ingestion.models import TextChunk


class VectorStoreError(RuntimeError):
    """Raised when the local vector index cannot be updated."""


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float


class ChromaVectorStore:
    """Write complete corpora to versioned collections.

    The active manifest changes only after every batch is persisted. This
    prevents a failed ingestion from replacing the last usable collection.
    """

    def __init__(
        self,
        persist_path: Path,
        collection_name: str,
        batch_size: int = 256,
    ) -> None:
        self.persist_path = persist_path.resolve()
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.manifest_path = self.persist_path / "active-index.json"

    def replace(self, chunks: Sequence[TextChunk], embeddings: Sequence[Sequence[float]]) -> str:
        if not chunks:
            raise VectorStoreError("Cannot create an index without chunks")
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts do not match")

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise VectorStoreError(
                "Vector storage requires chromadb. Install project dependencies first."
            ) from exc

        self.persist_path.mkdir(parents=True, exist_ok=True)
        version_name = f"{self.collection_name}-{uuid.uuid4().hex[:12]}"
        previous_name = self._active_collection_name()
        client: Any | None = None

        try:
            client = chromadb.PersistentClient(path=str(self.persist_path))
            collection = client.create_collection(
                name=version_name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
                metadata={"logical_name": self.collection_name},
            )
            for offset in range(0, len(chunks), self.batch_size):
                chunk_batch = chunks[offset : offset + self.batch_size]
                vector_batch = embeddings[offset : offset + self.batch_size]
                collection.add(
                    ids=[chunk.id for chunk in chunk_batch],
                    documents=[chunk.text for chunk in chunk_batch],
                    metadatas=[chunk.metadata() for chunk in chunk_batch],
                    embeddings=[list(vector) for vector in vector_batch],
                )

            self._write_manifest(version_name, len(chunks))
        except Exception as exc:
            if client is not None:
                try:
                    client.delete_collection(version_name)
                except Exception:
                    pass
            raise VectorStoreError(f"Could not persist Chroma index: {exc}") from exc

        if previous_name and previous_name != version_name:
            try:
                client.delete_collection(previous_name)
            except Exception:
                # The new collection is already active. A stale collection is
                # harmless and can be cleaned up on a later maintenance pass.
                pass
        return version_name

    def query(self, query_embedding: Sequence[float], top_k: int) -> list[VectorSearchResult]:
        if not query_embedding:
            raise VectorStoreError("Query embedding cannot be empty")
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1")

        active_collection = self.ensure_ready()

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise VectorStoreError(
                "Vector storage requires chromadb. Install project dependencies first."
            ) from exc

        try:
            client = chromadb.PersistentClient(path=str(self.persist_path))
            collection = client.get_collection(
                name=active_collection,
                embedding_function=None,
            )
            result_count = min(top_k, collection.count())
            if result_count == 0:
                return []
            raw = collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=result_count,
                include=["documents", "metadatas", "distances"],
            )
            return _parse_query_results(raw)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not query Chroma index: {exc}") from exc

    def ensure_ready(self) -> str:
        """Return the active collection or fail before loading an embedding model."""

        active_collection = self._active_collection_name()
        if active_collection is None:
            raise VectorStoreError(
                f"No document index found at {self.persist_path}. Run ingest first."
            )
        return active_collection

    def _active_collection_name(self) -> str | None:
        if not self.manifest_path.exists():
            return None
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            logical_name = data.get("logical_name")
            if logical_name != self.collection_name:
                return None
            name = data.get("collection")
            return name if isinstance(name, str) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_manifest(self, active_collection: str, chunk_count: int) -> None:
        manifest = {
            "collection": active_collection,
            "logical_name": self.collection_name,
            "chunk_count": chunk_count,
        }
        temporary = self.manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)


def _parse_query_results(raw: Any) -> list[VectorSearchResult]:
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
        raise VectorStoreError("Chroma returned inconsistent query result lengths")

    results: list[VectorSearchResult] = []
    for record_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        if document is None or distance is None:
            continue
        results.append(
            VectorSearchResult(
                id=str(record_id),
                document=str(document),
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
        )
    return results
