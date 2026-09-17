"""Refusal gate and source rendering around retrieval and generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_generator.retrieval.retriever import RetrievedChunk


NOT_FOUND_RESPONSE = "I couldn't find information about this in the provided documents."


class Retriever(Protocol):
    min_similarity: float

    def retrieve(self, question: str) -> list[RetrievedChunk]: ...


class Generator(Protocol):
    def generate(self, question: str, chunks: Sequence[RetrievedChunk]) -> str: ...


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: tuple[RetrievedChunk, ...]
    refused_before_generation: bool = False

    def render(self) -> str:
        if not self.sources:
            return f"{self.answer}\n\nSources: none (no relevant chunk met the similarity threshold)."

        source_lines = [
            f"- [S{number}] {chunk.citation} — similarity {chunk.similarity:.3f}"
            for number, chunk in enumerate(self.sources, start=1)
        ]
        return f"{self.answer}\n\nSources:\n" + "\n".join(source_lines)


class AnswerService:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        chunks = self.retriever.retrieve(question)
        if not chunks:
            return AnswerResult(
                answer=NOT_FOUND_RESPONSE,
                sources=(),
                refused_before_generation=True,
            )

        answer = self.generator.generate(question, chunks)
        return AnswerResult(answer=answer, sources=tuple(chunks))
