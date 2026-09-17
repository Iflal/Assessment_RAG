from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_generator.ingestion.loaders import IngestionInputError, load_documents


class LoaderTests(unittest.TestCase):
    def test_loads_supported_files_recursively_and_warns_for_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notes.txt").write_text("Plain text", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "guide.md").write_text("# Guide\nMarkdown", encoding="utf-8")
            (root / "draft.docx").write_bytes(b"not a real document")

            result = load_documents(root)

        self.assertEqual(result.loaded_files, ("nested/guide.md", "notes.txt"))
        self.assertEqual(result.skipped_files, ("draft.docx",))
        self.assertEqual({doc.source for doc in result.documents}, {"notes.txt", "nested/guide.md"})
        self.assertIn("Skipped unsupported file: draft.docx", result.warnings)

    def test_pdf_pages_keep_one_based_page_numbers(self) -> None:
        class FakePage:
            def __init__(self, text: str | None) -> None:
                self._text = text

            def extract_text(self) -> str | None:
                return self._text

        class FakeReader:
            def __init__(self, _path: Path) -> None:
                self.pages = [FakePage("Page one"), FakePage(None), FakePage("Page three")]

        fake_pypdf = types.SimpleNamespace(PdfReader=FakeReader)
        with tempfile.TemporaryDirectory() as temporary:
            pdf = Path(temporary) / "report.pdf"
            pdf.write_bytes(b"fake")
            with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
                result = load_documents(pdf)

        self.assertEqual([doc.page for doc in result.documents], [1, 3])
        self.assertTrue(all(doc.source == "report.pdf" for doc in result.documents))

    def test_empty_directory_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(IngestionInputError, "No files found"):
                load_documents(Path(temporary))


if __name__ == "__main__":
    unittest.main()
