"""Small end-to-end evaluation for retrieval, refusal, and generation."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rag_generator.config import Settings
from rag_generator.generation.answerer import AnswerResult, AnswerService
from rag_generator.generation.groq_client import GenerationError, GroqGenerator
from rag_generator.indexing.embeddings import EmbeddingError, MiniLMEmbedder
from rag_generator.indexing.vector_store import ChromaVectorStore, VectorStoreError
from rag_generator.ingestion.chunking import TextChunker
from rag_generator.ingestion.loaders import DocumentLoadError, IngestionInputError
from rag_generator.ingestion.pipeline import IngestionService
from rag_generator.retrieval.retriever import SimilarityRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = PROJECT_ROOT / "tests" / "fixtures" / "eval_docs"
DEFAULT_INDEX = PROJECT_ROOT / ".eval-chroma"


@dataclass(frozen=True)
class EvalCase:
    name: str
    question: str
    required_term_groups: tuple[tuple[str, ...], ...] = ()
    expect_refusal: bool = False


CASES = (
    EvalCase(
        name="supported-file-types",
        question="Which file formats does the portal accept?",
        required_term_groups=(("csv",), ("json",)),
    ),
    EvalCase(
        name="maximum-upload-size",
        question="What is the maximum size of one uploaded file?",
        required_term_groups=(("25 mb", "25 megabyte"),),
    ),
    EvalCase(
        name="successful-import-retention",
        question="How long are files from successful imports retained?",
        required_term_groups=(("45",), ("day",)),
    ),
    EvalCase(
        name="support-hours",
        question="When is the support desk staffed?",
        required_term_groups=(
            ("monday", "weekday"),
            ("friday", "weekday"),
            ("08:30", "8:30"),
            ("17:00", "5:00"),
            ("asia/colombo", "colombo"),
        ),
    ),
    EvalCase(
        name="unanswerable-capital",
        question="What is the capital of Mongolia?",
        expect_refusal=True,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run five end-to-end RAG checks.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return _run_evaluation(args)
    except (
        ValueError,
        IngestionInputError,
        DocumentLoadError,
        EmbeddingError,
        VectorStoreError,
        GenerationError,
    ) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Evaluation cancelled by user.", file=sys.stderr)
        return 130
    except Exception:
        if os.getenv("RAG_DEBUG") == "1":
            raise
        print(
            "Evaluation error: An unexpected failure occurred. "
            "Set RAG_DEBUG=1 and rerun for technical details.",
            file=sys.stderr,
        )
        return 2


def _run_evaluation(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    top_k = args.top_k if args.top_k is not None else settings.top_k
    min_similarity = (
        args.min_similarity
        if args.min_similarity is not None
        else settings.min_similarity
    )
    model = args.model or settings.groq_model

    embedder = MiniLMEmbedder(settings.embedding_model, settings.embedding_batch_size)
    vector_store = ChromaVectorStore(args.index_path, "rag-evaluation")
    ingestion = IngestionService(
        TextChunker(settings.chunk_size, settings.chunk_overlap),
        embedder,
        vector_store,
    )
    report = ingestion.ingest(args.docs)
    print(
        f"Evaluation corpus: {len(report.loaded_files)} file(s), "
        f"{report.chunk_count} chunk(s)"
    )
    print(f"Retrieval: top_k={top_k}, min_similarity={min_similarity:.2f}")
    print()

    answer_service = AnswerService(
        SimilarityRetriever(embedder, vector_store, top_k, min_similarity),
        GroqGenerator(model, settings.max_completion_tokens),
    )

    passed = 0
    for position, case in enumerate(CASES, start=1):
        try:
            result = answer_service.answer(case.question)
            success, reason = _grade(case, result)
        except (EmbeddingError, VectorStoreError, GenerationError, ValueError) as exc:
            result = None
            success, reason = False, str(exc)

        status = "PASS" if success else "FAIL"
        print(f"[{status}] {position}. {case.name}")
        print(f"  Q: {case.question}")
        if result is not None:
            print(f"  A: {_one_line(result.answer)}")
            if result.sources:
                citations = ", ".join(chunk.citation for chunk in result.sources)
                print(f"  Sources: {citations}")
        print(f"  Check: {reason}")
        print()
        passed += int(success)

    total = len(CASES)
    print(f"Summary: {passed}/{total} passed, {total - passed}/{total} failed")
    return 0 if passed == total else 1


def _grade(case: EvalCase, result: AnswerResult) -> tuple[bool, str]:
    if case.expect_refusal:
        refused = result.refused_before_generation or "couldn't find" in result.answer.lower()
        reason = "explicitly refused unsupported question" if refused else "expected refusal"
        return refused, reason

    answer = re.sub(r"\s+", " ", result.answer.lower())
    missing_groups = [
        "/".join(group)
        for group in case.required_term_groups
        if not any(term in answer for term in group)
    ]
    if missing_groups:
        return False, f"missing expected terms: {', '.join(missing_groups)}"
    if not result.sources:
        return False, "answer had no source chunks"
    if "[s" not in result.render().lower():
        return False, "answer had no rendered source citation"
    return True, "expected facts and source citations present"


def _one_line(text: str, limit: int = 500) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
