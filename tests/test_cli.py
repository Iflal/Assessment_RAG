from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from rag_generator.cli import main
from rag_generator.generation.groq_client import GenerationError
from rag_generator.indexing.embeddings import EmbeddingError


class CliErrorHandlingTests(unittest.TestCase):
    def test_empty_folder_is_clear_and_has_no_traceback(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stderr(stderr):
            exit_code = main(["ingest", temporary])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: No files found", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_missing_index_is_clear_and_has_no_traceback(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stderr(stderr):
            exit_code = main(["ask", "A question", "--index-path", temporary])

        self.assertEqual(exit_code, 1)
        self.assertIn("Run ingest first", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_embedding_failure_is_clear_and_has_no_traceback(self) -> None:
        stderr = io.StringIO()
        with patch(
            "rag_generator.cli._run_ingest",
            side_effect=EmbeddingError("Embedding model is unavailable."),
        ), redirect_stderr(stderr):
            exit_code = main(["ingest", str(Path("documents"))])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr.getvalue().strip(), "Error: Embedding model is unavailable.")

    def test_generation_failure_is_clear_and_has_no_traceback(self) -> None:
        stderr = io.StringIO()
        with patch(
            "rag_generator.cli._run_ask",
            side_effect=GenerationError("Groq authentication failed."),
        ), redirect_stderr(stderr):
            exit_code = main(["ask", "A question"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr.getvalue().strip(), "Error: Groq authentication failed.")


if __name__ == "__main__":
    unittest.main()
