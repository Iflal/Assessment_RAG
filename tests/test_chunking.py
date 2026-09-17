from __future__ import annotations

import unittest

from rag_generator.ingestion.chunking import TextChunker
from rag_generator.ingestion.models import LoadedDocument


class ChunkerTests(unittest.TestCase):
    def test_chunks_are_bounded_and_keep_citation_metadata(self) -> None:
        text = "# First\n\n" + ("First section has useful details. " * 8) + "\n\n# Second\n\n" + (
            "Second section has other facts. " * 8
        )
        document = LoadedDocument("manual.md", text, "md")

        chunks = TextChunker(chunk_size=180, chunk_overlap=40).chunk_documents([document])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 180 for chunk in chunks))
        self.assertTrue(all(chunk.source == "manual.md" for chunk in chunks))
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(chunk.char_start < chunk.char_end for chunk in chunks))

    def test_chunk_ids_are_deterministic(self) -> None:
        document = LoadedDocument("source.txt", "A sufficiently long but simple document.", "txt")
        chunker = TextChunker()

        first = chunker.chunk_documents([document])
        second = chunker.chunk_documents([document])

        self.assertEqual([chunk.id for chunk in first], [chunk.id for chunk in second])

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_zero_overlap_does_not_repeat_characters(self) -> None:
        text = "word " * 80
        document = LoadedDocument("source.txt", text, "txt")

        chunks = TextChunker(chunk_size=100, chunk_overlap=0).chunk_documents([document])

        for previous, current in zip(chunks, chunks[1:]):
            self.assertGreaterEqual(current.char_start, previous.char_end)

    def test_markdown_headers_start_new_chunks(self) -> None:
        first_section = "First section sentence. " * 8
        second_section = "Second section sentence. " * 8
        text = f"# First\n\n{first_section}\n\n## Second\n\n{second_section}"

        chunks = TextChunker(chunk_size=180, chunk_overlap=30).chunk_documents(
            [LoadedDocument("guide.md", text, "md")]
        )

        second_header_chunks = [chunk for chunk in chunks if "## Second" in chunk.text]
        self.assertEqual(len(second_header_chunks), 1)
        self.assertTrue(second_header_chunks[0].text.startswith("## Second"))

    def test_prefers_complete_sentences_before_word_fallback(self) -> None:
        sentences = [
            "Alpha sentence contains several useful words.",
            "Beta sentence also contains several useful words.",
            "Gamma sentence finishes the document cleanly.",
        ]
        text = " ".join(sentences)

        chunks = TextChunker(chunk_size=105, chunk_overlap=0).chunk_documents(
            [LoadedDocument("facts.txt", text, "txt")]
        )

        self.assertEqual(" ".join(chunk.text for chunk in chunks), text)
        self.assertTrue(all(chunk.text.endswith(".") for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
