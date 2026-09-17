"""Data passed between ingestion stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedDocument:
    """Text extracted from one source file or one PDF page."""

    source: str
    text: str
    file_type: str
    page: int | None = None


@dataclass(frozen=True)
class TextChunk:
    """A citation-ready piece of a loaded document."""

    id: str
    text: str
    source: str
    file_type: str
    chunk_index: int
    char_start: int
    char_end: int
    page: int | None = None

    def metadata(self) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "source": self.source,
            "file_type": self.file_type,
            "chunk_id": self.id,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }
        if self.page is not None:
            metadata["page"] = self.page
        return metadata

