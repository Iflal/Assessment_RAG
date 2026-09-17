from __future__ import annotations

import types
import unittest

from rag_generator.generation.groq_client import (
    GenerationError,
    GroqGenerator,
    SYSTEM_PROMPT,
)
from rag_generator.retrieval.retriever import RetrievedChunk


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = types.SimpleNamespace(content="A grounded answer [S1].")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class GroqGeneratorTests(unittest.TestCase):
    def test_prompt_forbids_outside_knowledge_and_labels_context(self) -> None:
        completions = FakeCompletions()
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        generator = GroqGenerator(client=client)
        chunk = RetrievedChunk(
            id="chunk-abc",
            text="The document-backed fact.",
            source="facts.md",
            chunk_index=0,
            similarity=0.9,
        )

        answer = generator.generate("What is the fact?", [chunk])

        self.assertEqual(answer, "A grounded answer [S1].")
        self.assertIn("only the supplied document chunks", SYSTEM_PROMPT)
        self.assertIn("Do not use outside knowledge", SYSTEM_PROMPT)
        messages = completions.kwargs["messages"]
        self.assertIn("[S1]", messages[1]["content"])
        self.assertIn("facts.md, chunk 0 (chunk-abc)", messages[1]["content"])
        self.assertIn("The document-backed fact.", messages[1]["content"])

    def test_authentication_error_is_mapped_without_sdk_details(self) -> None:
        class AuthenticationError(Exception):
            pass

        class FailingCompletions:
            def create(self, **kwargs):
                raise AuthenticationError("secret response details")

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FailingCompletions())
        )
        generator = GroqGenerator(client=client)
        chunk = RetrievedChunk(
            id="chunk-abc",
            text="Fact",
            source="facts.md",
            chunk_index=0,
            similarity=0.9,
        )

        with self.assertRaisesRegex(GenerationError, "Groq authentication failed") as error:
            generator.generate("Question", [chunk])

        self.assertNotIn("secret response details", str(error.exception))


if __name__ == "__main__":
    unittest.main()
