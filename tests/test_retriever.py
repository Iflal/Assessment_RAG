from __future__ import annotations

import unittest
from collections.abc import Sequence

from rag_generator.indexing.vector_store import VectorSearchResult
from rag_generator.retrieval.retriever import SimilarityRetriever


class FakeQueryEmbedder:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.questions.append(text)
        return [1.0, 0.0]


class FakeSearchStore:
    def __init__(self, records: list[VectorSearchResult]) -> None:
        self.records = records
        self.top_k: int | None = None

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        self.top_k = top_k
        return self.records[:top_k]


class RetrieverTests(unittest.TestCase):
    def test_top_k_search_filters_results_below_similarity_threshold(self) -> None:
        records = [
            VectorSearchResult(
                id="chunk-relevant",
                document="Relevant content",
                metadata={"source": "guide.md", "chunk_index": 2, "page": 3},
                distance=0.20,
            ),
            VectorSearchResult(
                id="chunk-irrelevant",
                document="Irrelevant content",
                metadata={"source": "other.txt", "chunk_index": 0},
                distance=0.80,
            ),
        ]
        embedder = FakeQueryEmbedder()
        store = FakeSearchStore(records)
        retriever = SimilarityRetriever(embedder, store, top_k=2, min_similarity=0.35)

        results = retriever.retrieve("What is relevant?")

        self.assertEqual(embedder.questions, ["What is relevant?"])
        self.assertEqual(store.top_k, 2)
        self.assertEqual([result.id for result in results], ["chunk-relevant"])
        self.assertAlmostEqual(results[0].similarity, 0.80)
        self.assertEqual(results[0].citation, "guide.md, page 3, chunk 2 (chunk-relevant)")

    def test_empty_question_is_rejected_before_embedding(self) -> None:
        embedder = FakeQueryEmbedder()
        retriever = SimilarityRetriever(embedder, FakeSearchStore([]))

        with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
            retriever.retrieve("   ")

        self.assertEqual(embedder.questions, [])


if __name__ == "__main__":
    unittest.main()

