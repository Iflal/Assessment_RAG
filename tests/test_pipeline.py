from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from rag_generator.ingestion.chunking import TextChunker
from rag_generator.ingestion.models import TextChunk
from rag_generator.ingestion.pipeline import IngestionService


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts = list(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[TextChunk] = []
        self.embeddings: list[list[float]] = []

    def replace(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> str:
        self.chunks = list(chunks)
        self.embeddings = [list(vector) for vector in embeddings]
        return "test-collection-version"


class PipelineTests(unittest.TestCase):
    def test_ingestion_orchestrates_all_stages(self) -> None:
        embedder = FakeEmbedder()
        store = FakeVectorStore()
        service = IngestionService(TextChunker(), embedder, store)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "facts.txt").write_text("Grounded facts live here.", encoding="utf-8")
            report = service.ingest(source)

        self.assertEqual(report.loaded_files, ("facts.txt",))
        self.assertEqual(report.chunk_count, 1)
        self.assertEqual(report.active_collection, "test-collection-version")
        self.assertEqual(embedder.texts, ["Grounded facts live here."])
        self.assertEqual(len(store.chunks), len(store.embeddings))


if __name__ == "__main__":
    unittest.main()

