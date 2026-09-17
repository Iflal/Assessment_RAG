"""Command-line interface for the RAG application."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from rag_generator.config import Settings
from rag_generator.generation.answerer import AnswerService
from rag_generator.generation.groq_client import GenerationError, GroqGenerator
from rag_generator.indexing.embeddings import EmbeddingError, MiniLMEmbedder
from rag_generator.indexing.vector_store import ChromaVectorStore, VectorStoreError
from rag_generator.ingestion.chunking import TextChunker
from rag_generator.ingestion.loaders import DocumentLoadError, IngestionInputError
from rag_generator.ingestion.pipeline import IngestionService
from rag_generator.retrieval.retriever import SimilarityRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Build and query a local document-grounded RAG index.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser(
        "ingest",
        help="Load, chunk, embed, and index a document file or directory.",
    )
    ingest.add_argument("source", type=Path, help="Document file or directory")
    ingest.add_argument("--index-path", type=Path, help="Chroma persistence directory")
    ingest.add_argument("--collection", help="Logical Chroma collection name")
    ingest.add_argument("--chunk-size", type=int, help="Maximum chunk size in characters")
    ingest.add_argument("--chunk-overlap", type=int, help="Chunk overlap in characters")
    ingest.add_argument("--batch-size", type=int, help="Embedding batch size")

    ask = subparsers.add_parser(
        "ask",
        help="Answer a question using only the active document index.",
    )
    ask.add_argument("question", help="Natural-language question")
    ask.add_argument("--index-path", type=Path, help="Chroma persistence directory")
    ask.add_argument("--collection", help="Logical Chroma collection name")
    ask.add_argument("--top-k", type=int, help="Maximum chunks to retrieve")
    ask.add_argument(
        "--min-similarity",
        type=float,
        help="Minimum cosine similarity from 0 to 1",
    )
    ask.add_argument("--model", help="Groq generation model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = _settings_with_cli_overrides(args)
        if args.command == "ingest":
            return _run_ingest(args.source, settings)
        if args.command == "ask":
            return _run_ask(args.question, settings)
    except (
        ValueError,
        IngestionInputError,
        DocumentLoadError,
        EmbeddingError,
        VectorStoreError,
        GenerationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Error: Operation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        if os.getenv("RAG_DEBUG") == "1":
            raise
        print(
            "Error: An unexpected failure occurred. Set RAG_DEBUG=1 for technical details.",
            file=sys.stderr,
        )
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _settings_with_cli_overrides(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    settings = Settings(
        chroma_path=getattr(args, "index_path", None) or base.chroma_path,
        collection_name=getattr(args, "collection", None) or base.collection_name,
        embedding_model=base.embedding_model,
        chunk_size=(
            args.chunk_size
            if getattr(args, "chunk_size", None) is not None
            else base.chunk_size
        ),
        chunk_overlap=(
            args.chunk_overlap
            if getattr(args, "chunk_overlap", None) is not None
            else base.chunk_overlap
        ),
        embedding_batch_size=(
            args.batch_size
            if getattr(args, "batch_size", None) is not None
            else base.embedding_batch_size
        ),
        top_k=args.top_k if getattr(args, "top_k", None) is not None else base.top_k,
        min_similarity=(
            args.min_similarity
            if getattr(args, "min_similarity", None) is not None
            else base.min_similarity
        ),
        groq_model=getattr(args, "model", None) or base.groq_model,
        max_completion_tokens=base.max_completion_tokens,
    )
    settings.validate()
    return settings


def _run_ingest(source: Path, settings: Settings) -> int:
    service = IngestionService(
        chunker=TextChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=MiniLMEmbedder(
            settings.embedding_model,
            settings.embedding_batch_size,
        ),
        vector_store=ChromaVectorStore(
            settings.chroma_path,
            settings.collection_name,
        ),
    )
    report = service.ingest(source)
    for warning in report.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(
        f"Indexed {report.chunk_count} chunks from {len(report.loaded_files)} files "
        f"into {settings.chroma_path.resolve()}"
    )
    print(f"Active collection: {report.active_collection}")
    return 0


def _run_ask(question: str, settings: Settings) -> int:
    embedder = MiniLMEmbedder(
        settings.embedding_model,
        settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(
        settings.chroma_path,
        settings.collection_name,
    )
    vector_store.ensure_ready()
    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
    )
    generator = GroqGenerator(
        model=settings.groq_model,
        max_completion_tokens=settings.max_completion_tokens,
    )
    result = AnswerService(retriever, generator).answer(question)
    print(result.render())
    return 0


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
