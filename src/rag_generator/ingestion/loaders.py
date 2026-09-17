"""Runtime document discovery and text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag_generator.ingestion.models import LoadedDocument


SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})


class IngestionInputError(ValueError):
    """Raised when the requested source cannot be ingested."""


class DocumentLoadError(RuntimeError):
    """Raised when a supported document cannot be read safely."""


@dataclass(frozen=True)
class LoadResult:
    documents: tuple[LoadedDocument, ...]
    loaded_files: tuple[str, ...]
    skipped_files: tuple[str, ...]
    warnings: tuple[str, ...]


def discover_files(source: Path) -> list[Path]:
    """Return deterministic file paths from a single file or directory."""

    source = source.expanduser()
    if not source.exists():
        raise IngestionInputError(f"Document path does not exist: {source}")
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise IngestionInputError(f"Document path is not a file or directory: {source}")
    return sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix().casefold(),
    )


def load_documents(source: Path) -> LoadResult:
    """Load all supported files while retaining citation metadata."""

    source = source.expanduser().resolve()
    files = discover_files(source)
    if not files:
        raise IngestionInputError(f"No files found in: {source}")

    documents: list[LoadedDocument] = []
    loaded_files: list[str] = []
    skipped_files: list[str] = []
    warnings: list[str] = []

    for path in files:
        relative_name = _relative_source(path, source)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            skipped_files.append(relative_name)
            warnings.append(f"Skipped unsupported file: {relative_name}")
            continue

        extracted = _load_file(path, relative_name, suffix)
        if not extracted:
            skipped_files.append(relative_name)
            warnings.append(f"Skipped file with no extractable text: {relative_name}")
            continue

        documents.extend(extracted)
        loaded_files.append(relative_name)

    if not documents:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionInputError(
            f"No readable documents found in {source}. Supported extensions: {supported}"
        )

    return LoadResult(
        documents=tuple(documents),
        loaded_files=tuple(loaded_files),
        skipped_files=tuple(skipped_files),
        warnings=tuple(warnings),
    )


def _relative_source(path: Path, source: Path) -> str:
    if source.is_file():
        return source.name
    return path.relative_to(source).as_posix()


def _load_file(path: Path, source_name: str, suffix: str) -> Iterable[LoadedDocument]:
    if suffix in {".txt", ".md"}:
        return _load_text(path, source_name, suffix)
    if suffix == ".pdf":
        return _load_pdf(path, source_name)
    return ()


def _load_text(path: Path, source_name: str, suffix: str) -> tuple[LoadedDocument, ...]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise DocumentLoadError(f"Could not read {source_name}: {exc}") from exc

    if not text.strip():
        return ()
    return (LoadedDocument(source_name, text, suffix.removeprefix(".")),)


def _load_pdf(path: Path, source_name: str) -> tuple[LoadedDocument, ...]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DocumentLoadError(
            "PDF support requires pypdf. Install project dependencies first."
        ) from exc

    try:
        reader = PdfReader(path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    LoadedDocument(
                        source=source_name,
                        text=text,
                        file_type="pdf",
                        page=page_number,
                    )
                )
        return tuple(pages)
    except Exception as exc:
        raise DocumentLoadError(f"Could not extract text from {source_name}: {exc}") from exc

