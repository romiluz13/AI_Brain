"""
Embeddings module for AI Brain Python

Provides embedding generation capabilities using various providers
including OpenAI, Voyage AI, and other embedding services.
"""

from .openai_embedding_provider import OpenAIEmbeddingProvider

__all__ = [
    "OpenAIEmbeddingProvider"
]
