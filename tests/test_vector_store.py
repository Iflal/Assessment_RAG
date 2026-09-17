from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_generator.indexing.vector_store import ChromaVectorStore, VectorStoreError
from rag_generator.ingestion.models import TextChunk


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def add(self, *, ids, documents, metadatas, embeddings) -> None:
        assert len(ids) == len(documents) == len(metadatas) == len(embeddings)
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def count(self) -> int:
        return len(self.ids)

    def query(self, *, query_embeddings, n_results, include):
        return {
            "ids": [self.ids[:n_results]],
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]],
            "distances": [[0.15] * min(n_results, len(self.ids))],
        }


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.deleted: list[str] = []

    def create_collection(self, *, name, configuration, embedding_function, metadata):
        assert configuration["hnsw"]["space"] == "cosine"
        collection = FakeCollection(name)
        self.collections[name] = collection
        return collection

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        self.collections.pop(name, None)

    def get_collection(self, *, name, embedding_function):
        return self.collections[name]


class VectorStoreTests(unittest.TestCase):
    def test_missing_active_index_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ChromaVectorStore(Path(temporary), "rag-documents")

            with self.assertRaisesRegex(VectorStoreError, "Run ingest first"):
                store.ensure_ready()

    def test_successful_replacement_switches_manifest_and_removes_old_version(self) -> None:
        client = FakeClient()
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)
        chunk = TextChunk(
            id="chunk-one",
            text="Grounded content",
            source="facts.txt",
            file_type="txt",
            chunk_index=0,
            char_start=0,
            char_end=16,
        )

        with tempfile.TemporaryDirectory() as temporary:
            store = ChromaVectorStore(Path(temporary), "rag-documents")
            with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
                first = store.replace([chunk], [[1.0, 0.0]])
                second = store.replace([chunk], [[1.0, 0.0]])

            manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))

        self.assertNotEqual(first, second)
        self.assertEqual(manifest["collection"], second)
        self.assertEqual(manifest["chunk_count"], 1)
        self.assertIn(first, client.deleted)
        self.assertIn(second, client.collections)

    def test_query_uses_the_active_collection(self) -> None:
        client = FakeClient()
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)
        chunk = TextChunk(
            id="chunk-one",
            text="Grounded content",
            source="facts.txt",
            file_type="txt",
            chunk_index=0,
            char_start=0,
            char_end=16,
        )

        with tempfile.TemporaryDirectory() as temporary:
            store = ChromaVectorStore(Path(temporary), "rag-documents")
            with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
                store.replace([chunk], [[1.0, 0.0]])
                results = store.query([1.0, 0.0], top_k=4)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "chunk-one")
        self.assertEqual(results[0].metadata["source"], "facts.txt")
        self.assertEqual(results[0].distance, 0.15)


if __name__ == "__main__":
    unittest.main()
