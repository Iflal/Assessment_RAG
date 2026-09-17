"""Recursive, structure-aware chunking with deterministic citation metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Iterator

from rag_generator.ingestion.models import LoadedDocument, TextChunk


@dataclass(frozen=True)
class _Separator:
    pattern: re.Pattern[str]
    break_at_match_start: bool = False
    hard_boundary: bool = False


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    hard_break_before: bool = False


# Separators are deliberately ordered from strongest document structure to
# weakest. A lower-priority separator is used only when a higher-level section
# is too large to fit in one chunk.
_SEPARATORS = (
    _Separator(
        re.compile(r"(?m)(?=^#{1,6}[ \t]+)"),
        break_at_match_start=True,
        hard_boundary=True,
    ),
    _Separator(re.compile(r"\n[ \t]*\n+")),
    _Separator(re.compile(r"(?<=[.!?])(?:[\"')\]]+)?\s+")),
    _Separator(re.compile(r"\n+")),
    _Separator(re.compile(r"[ \t]+")),
)


@dataclass(frozen=True)
class TextChunker:
    chunk_size: int = 800
    chunk_overlap: int = 120

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    def chunk_documents(self, documents: Iterable[LoadedDocument]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        per_source_index: dict[tuple[str, int | None], int] = {}

        for document in documents:
            key = (document.source, document.page)
            next_index = per_source_index.get(key, 0)
            atomic_spans = self._recursive_split(document.text, 0, len(document.text), 0)

            for span in self._merge_spans(atomic_spans):
                raw = document.text[span.start : span.end]
                left_trim = len(raw) - len(raw.lstrip())
                right_trim = len(raw.rstrip())
                text = raw.strip()
                if not text:
                    continue

                char_start = span.start + left_trim
                char_end = span.start + right_trim
                chunk_id = _chunk_id(document, next_index, text)
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        text=text,
                        source=document.source,
                        file_type=document.file_type,
                        page=document.page,
                        chunk_index=next_index,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
                next_index += 1
            per_source_index[key] = next_index
        return chunks

    def _recursive_split(
        self,
        text: str,
        start: int,
        end: int,
        separator_index: int,
        hard_break_before: bool = False,
    ) -> list[_Span]:
        """Recursively split an oversized span using weaker boundaries."""

        if end - start <= self.chunk_size:
            return [_Span(start, end, hard_break_before)]

        for index in range(separator_index, len(_SEPARATORS)):
            separator = _SEPARATORS[index]
            boundaries = _boundary_positions(text, start, end, separator)
            if not boundaries:
                continue

            split_spans: list[_Span] = []
            cursor = start
            next_is_hard = hard_break_before
            for boundary in (*boundaries, end):
                if boundary <= cursor:
                    continue
                split_spans.extend(
                    self._recursive_split(
                        text,
                        cursor,
                        boundary,
                        index + 1,
                        next_is_hard,
                    )
                )
                cursor = boundary
                next_is_hard = separator.hard_boundary

            if cursor == end and split_spans:
                return split_spans

        # A single word/token can still exceed the configured size. Fixed-width
        # splitting is the final fallback and guarantees the size limit.
        return [
            _Span(position, min(position + self.chunk_size, end), hard_break_before and position == start)
            for position in range(start, end, self.chunk_size)
        ]

    def _merge_spans(self, spans: list[_Span]) -> Iterator[_Span]:
        """Greedily pack semantic spans and retain whole-span overlap."""

        current: list[_Span] = []
        for span in spans:
            must_flush = bool(current) and (
                span.hard_break_before
                or span.end - current[0].start > self.chunk_size
            )
            if must_flush:
                yield _Span(current[0].start, current[-1].end, current[0].hard_break_before)
                current = [] if span.hard_break_before else self._overlap_tail(current)
                while current and span.end - current[0].start > self.chunk_size:
                    current.pop(0)
            current.append(span)

        if current:
            yield _Span(current[0].start, current[-1].end, current[0].hard_break_before)

    def _overlap_tail(self, spans: list[_Span]) -> list[_Span]:
        if self.chunk_overlap == 0:
            return []

        selected: list[_Span] = []
        previous_end = spans[-1].end
        for span in reversed(spans):
            if previous_end - span.start > self.chunk_overlap:
                break
            selected.append(span)
        selected.reverse()
        return selected


def _boundary_positions(
    text: str,
    start: int,
    end: int,
    separator: _Separator,
) -> tuple[int, ...]:
    boundaries: list[int] = []
    for match in separator.pattern.finditer(text, start, end):
        boundary = match.start() if separator.break_at_match_start else match.end()
        if start < boundary < end:
            boundaries.append(boundary)
    return tuple(dict.fromkeys(boundaries))


def _chunk_id(document: LoadedDocument, index: int, text: str) -> str:
    identity = f"{document.source}\0{document.page or 0}\0{index}\0{text}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"chunk-{digest}"
