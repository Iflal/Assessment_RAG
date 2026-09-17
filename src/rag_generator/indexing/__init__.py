"""Embedding and local vector-store adapters."""

from rag_generator.indexing.embeddings import MiniLMEmbedder
from rag_generator.indexing.vector_store import ChromaVectorStore

__all__ = ["ChromaVectorStore", "MiniLMEmbedder"]

