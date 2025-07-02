"""
OpenAI Embedding Provider

Provides embedding generation using OpenAI's text-embedding models
with support for batch processing and caching.
"""

import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..utils.logger import logger


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider with caching and batch processing."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.cache: Dict[str, List[float]] = {}
        
        # Configuration
        self.config = {
            "model": model,
            "dimensions": 1536 if "3-small" in model else 3072,
            "max_batch_size": 100,
            "max_tokens_per_request": 8000,
            "enable_caching": True
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        # Check cache first
        if self.config["enable_caching"]:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return self.cache[cache_key]
        
        # Generate embedding
        try:
            # In a real implementation, this would call OpenAI API
            # For now, return a mock embedding
            embedding = self._generate_mock_embedding(text)
            
            # Cache the result
            if self.config["enable_caching"]:
                self.cache[cache_key] = embedding
            
            logger.debug(f"Generated embedding for: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        # Process in batches
        batch_size = self.config["max_batch_size"]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = []
            for text in batch:
                embedding = await self.generate_embedding(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Small delay to avoid rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for testing purposes."""
        # Create a deterministic but varied embedding based on text
        import random
        random.seed(hash(text) % (2**32))
        
        dimensions = self.config["dimensions"]
        embedding = [random.uniform(-1, 1) for _ in range(dimensions)]
        
        # Normalize the embedding
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimensions")
        
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Compute magnitudes
        magnitude1 = sum(a**2 for a in embedding1) ** 0.5
        magnitude2 = sum(b**2 for b in embedding2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity
    
    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings to query."""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding provider."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "dimensions": self.config["dimensions"],
            "cache_size": len(self.cache),
            "max_batch_size": self.config["max_batch_size"]
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
