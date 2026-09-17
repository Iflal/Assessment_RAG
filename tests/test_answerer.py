from __future__ import annotations

import unittest
from collections.abc import Sequence

from rag_generator.generation.answerer import NOT_FOUND_RESPONSE, AnswerService
from rag_generator.indexing.vector_store import VectorSearchResult
from rag_generator.retrieval.retriever import RetrievedChunk, SimilarityRetriever


class FakeRetriever:
    min_similarity = 0.35

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self.chunks


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Sequence[RetrievedChunk]]] = []

    def generate(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        self.calls.append((question, chunks))
        return "The supported answer is here [S1]."


class FixedEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class BelowThresholdStore:
    def query(self, query_embedding, top_k):
        return [
            VectorSearchResult(
                id="chunk-low-score",
                document="Unrelated content",
                metadata={"source": "unrelated.md", "chunk_index": 0},
                distance=0.9,
            )
        ]


class AnswerServiceTests(unittest.TestCase):
    def test_no_relevant_chunks_refuses_without_calling_generator(self) -> None:
        generator = RecordingGenerator()
        service = AnswerService(FakeRetriever([]), generator)

        result = service.answer("An unsupported question")

        self.assertEqual(result.answer, NOT_FOUND_RESPONSE)
        self.assertTrue(result.refused_before_generation)
        self.assertEqual(generator.calls, [])
        self.assertIn("Sources: none", result.render())

    def test_grounded_answer_always_renders_source_and_chunk_citations(self) -> None:
        chunk = RetrievedChunk(
            id="chunk-abc",
            text="Supporting text",
            source="policy.pdf",
            page=4,
            chunk_index=1,
            similarity=0.82,
        )
        generator = RecordingGenerator()
        service = AnswerService(FakeRetriever([chunk]), generator)

        rendered = service.answer("What is the policy?").render()

        self.assertEqual(len(generator.calls), 1)
        self.assertIn("[S1]", rendered)
        self.assertIn("policy.pdf, page 4, chunk 1 (chunk-abc)", rendered)
        self.assertIn("similarity 0.820", rendered)

    def test_below_threshold_result_never_reaches_generator(self) -> None:
        retriever = SimilarityRetriever(
            FixedEmbedder(),
            BelowThresholdStore(),
            min_similarity=0.35,
        )
        generator = RecordingGenerator()

        result = AnswerService(retriever, generator).answer("Unsupported topic")

        self.assertTrue(result.refused_before_generation)
        self.assertEqual(generator.calls, [])


if __name__ == "__main__":
    unittest.main()
