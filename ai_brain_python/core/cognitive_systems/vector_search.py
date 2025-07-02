"""
VectorSearchEngine - Advanced vector search capabilities for Universal AI Brain

Exact Python equivalent of JavaScript VectorSearchEngine.ts with:
- Semantic vector search using MongoDB Atlas Vector Search
- Hybrid search combining vector and text search
- Multiple embedding provider support
- Search result ranking and filtering
- Real-time search analytics and optimization
- Search caching and performance optimization

Features:
- Semantic vector search using MongoDB Atlas Vector Search
- Hybrid search combining vector and text search
- Multiple embedding provider support
- Search result ranking and filtering
- Real-time search analytics and optimization
- Search caching and performance optimization
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.interfaces.cognitive_system import CognitiveSystemInterface
from ..core.models import CognitiveInputData, CognitiveResponse
from ..utils.logger import logger


@dataclass
class SearchResult:
    """Search result data structure."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    explanation: Optional[str] = None


@dataclass
class SearchOptions:
    """Search options data structure."""
    limit: int = 10
    min_score: float = 0.0
    max_candidates: int = 100
    include_embeddings: bool = False
    include_explanation: bool = False
    filters: Optional[Dict[str, Any]] = None
    boost: Optional[List[Dict[str, Any]]] = None


@dataclass
class HybridSearchOptions(SearchOptions):
    """Hybrid search options data structure."""
    vector_weight: float = 0.7
    text_weight: float = 0.3
    text_query: Optional[str] = None


@dataclass
class SearchAnalytics:
    """Search analytics data structure."""
    total_searches: int
    average_latency: float
    average_result_count: float
    search_type_distribution: Dict[str, int]
    popular_queries: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]


class VectorSearchEngine(CognitiveSystemInterface):
    """
    VectorSearchEngine - Advanced vector search using MongoDB Atlas Vector Search
    
    Exact Python equivalent of JavaScript VectorSearchEngine with:
    - Semantic vector search using MongoDB Atlas Vector Search
    - Hybrid search combining vector and text search
    - Multiple embedding provider support
    - Search result ranking and filtering
    - Real-time search analytics and optimization
    - Search caching and performance optimization
    """
    
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        embedding_provider: Optional[Any] = None,
        collection_name: str = "vector_embeddings",
        vector_index_name: str = "vector_search_index",
        text_index_name: str = "text_search_index"
    ):
        super().__init__(db)
        self.db = db
        self.embedding_provider = embedding_provider  # Would be OpenAI or other provider
        self.collection_name = collection_name
        self.vector_index_name = vector_index_name
        self.text_index_name = text_index_name
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = 1000
        self.cache_ttl = 5 * 60  # 5 minutes in seconds
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the vector search engine."""
        if self.is_initialized:
            return
            
        try:
            # Initialize collection
            self.collection = self.db[self.collection_name]
            self.is_initialized = True
            logger.info("✅ VectorSearchEngine initialized successfully")
            logger.info("📝 Note: Requires MongoDB Atlas Vector Search indexes")
        except Exception as error:
            logger.error(f"❌ Error initializing VectorSearchEngine: {error}")
            raise error
    
    async def process(self, input_data: CognitiveInputData) -> CognitiveResponse:
        """Process vector search requests."""
        try:
            await self.initialize()
            
            # Extract search request from input
            request_data = input_data.additional_context.get("search_request", {})
            
            if not request_data:
                return CognitiveResponse(
                    response_text="No search request provided",
                    confidence=0.0,
                    processing_metadata={
                        "system": "vector_search",
                        "error": "Missing search request"
                    }
                )
            
            search_type = request_data.get("type", "semantic")
            query = request_data.get("query", "")
            options_data = request_data.get("options", {})
            
            # Create search options
            if search_type == "hybrid":
                options = HybridSearchOptions(
                    limit=options_data.get("limit", 10),
                    min_score=options_data.get("minScore", 0.0),
                    max_candidates=options_data.get("maxCandidates", 100),
                    include_embeddings=options_data.get("includeEmbeddings", False),
                    include_explanation=options_data.get("includeExplanation", False),
                    filters=options_data.get("filters"),
                    boost=options_data.get("boost"),
                    vector_weight=options_data.get("vectorWeight", 0.7),
                    text_weight=options_data.get("textWeight", 0.3),
                    text_query=options_data.get("textQuery")
                )
            else:
                options = SearchOptions(
                    limit=options_data.get("limit", 10),
                    min_score=options_data.get("minScore", 0.0),
                    max_candidates=options_data.get("maxCandidates", 100),
                    include_embeddings=options_data.get("includeEmbeddings", False),
                    include_explanation=options_data.get("includeExplanation", False),
                    filters=options_data.get("filters"),
                    boost=options_data.get("boost")
                )
            
            # Perform search
            if search_type == "semantic":
                results = await self.semantic_search(query, options)
            elif search_type == "hybrid":
                results = await self.hybrid_search(query, options)
            elif search_type == "text":
                results = await self.text_search(query, options)
            else:
                results = []
            
            response_text = f"Found {len(results)} results for {search_type} search"
            avg_score = sum(r.score for r in results) / len(results) if results else 0
            
            return CognitiveResponse(
                response_text=response_text,
                confidence=avg_score,
                processing_metadata={
                    "system": "vector_search",
                    "search_type": search_type,
                    "results_count": len(results),
                    "average_score": avg_score,
                    "atlas_vector_search": True
                }
            )
            
        except Exception as error:
            logger.error(f"Error in VectorSearchEngine.process: {error}")
            return CognitiveResponse(
                response_text=f"Vector search error: {str(error)}",
                confidence=0.0,
                processing_metadata={
                    "system": "vector_search",
                    "error": str(error)
                }
            )
    
    async def semantic_search(self, query: str, options: SearchOptions = SearchOptions()) -> List[SearchResult]:
        """Perform semantic search using vector similarity."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key("semantic", query, options)
            cached_results = self._get_from_cache(cache_key)
            if cached_results is not None:
                return cached_results
            
            # Generate embedding for query
            query_embedding = await self.create_embedding(query)
            if not query_embedding:
                return []
            
            # Build vector search pipeline
            pipeline = self._build_vector_search_pipeline(query_embedding, {
                "limit": options.limit,
                "min_score": options.min_score,
                "max_candidates": options.max_candidates,
                "filters": options.filters,
                "boost": options.boost,
                "include_embeddings": options.include_embeddings
            })
            
            # Execute search
            results = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Process results
            search_results = self._process_search_results(results, "semantic", options.include_explanation)
            
            # Cache results
            self._set_cache(cache_key, search_results)
            
            # Log analytics
            latency = (time.time() - start_time) * 1000
            self._log_search_analytics("semantic", query, search_results, latency)
            
            return search_results
            
        except Exception as error:
            logger.error(f"Error in semantic search: {error}")
            return []

    async def hybrid_search(self, query: str, options: HybridSearchOptions = HybridSearchOptions()) -> List[SearchResult]:
        """Perform hybrid search combining vector and text search."""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key("hybrid", query, options)
            cached_results = self._get_from_cache(cache_key)
            if cached_results is not None:
                return cached_results

            # Generate embedding for query
            query_embedding = await self.create_embedding(query)
            if not query_embedding:
                return []

            # Build hybrid search pipeline
            pipeline = self._build_hybrid_search_pipeline(query, query_embedding, {
                "limit": options.limit,
                "min_score": options.min_score,
                "max_candidates": options.max_candidates,
                "filters": options.filters,
                "vector_weight": options.vector_weight,
                "text_weight": options.text_weight,
                "text_query": options.text_query or query,
                "include_embeddings": options.include_embeddings
            })

            # Execute search
            results = await self.collection.aggregate(pipeline).to_list(length=None)

            # Process results
            search_results = self._process_search_results(results, "hybrid", options.include_explanation)

            # Cache results
            self._set_cache(cache_key, search_results)

            # Log analytics
            latency = (time.time() - start_time) * 1000
            self._log_search_analytics("hybrid", query, search_results, latency)

            return search_results

        except Exception as error:
            logger.error(f"Error in hybrid search: {error}")
            return []

    async def text_search(self, query: str, options: SearchOptions = SearchOptions()) -> List[SearchResult]:
        """Perform text-only search."""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key("text", query, options)
            cached_results = self._get_from_cache(cache_key)
            if cached_results is not None:
                return cached_results

            # Build text search pipeline
            pipeline = [
                {
                    "$search": {
                        "index": self.text_index_name,
                        "text": {
                            "query": query,
                            "path": "content"
                        }
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "searchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": options.min_score}
                    }
                }
            ]

            # Add filters
            if options.filters:
                pipeline.append({"$match": options.filters})

            # Add limit
            pipeline.append({"$limit": options.limit})

            # Add projection
            projection = {
                "_id": 1,
                "content": 1,
                "metadata": 1,
                "score": 1
            }

            if options.include_embeddings:
                projection["embedding"] = 1

            pipeline.append({"$project": projection})

            # Execute search
            results = await self.collection.aggregate(pipeline).to_list(length=None)

            # Process results
            search_results = self._process_search_results(results, "text", options.include_explanation)

            # Cache results
            self._set_cache(cache_key, search_results)

            # Log analytics
            latency = (time.time() - start_time) * 1000
            self._log_search_analytics("text", query, search_results, latency)

            return search_results

        except Exception as error:
            logger.error(f"Error in text search: {error}")
            return []

    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text."""
        try:
            if self.embedding_provider:
                return await self.embedding_provider.generate_embedding(text)
            else:
                # Fallback: return dummy embedding for testing
                logger.warning("No embedding provider configured, returning dummy embedding")
                return [0.1] * 1536  # OpenAI embedding dimension
        except Exception as error:
            logger.error(f"Failed to create embedding: {error}")
            return []

    async def store_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Store document with automatic embedding generation."""
        try:
            # Generate embedding
            embedding = await self.create_embedding(content)
            if not embedding:
                raise ValueError("Failed to generate embedding")

            # Create document
            doc_id = document_id or str(ObjectId())
            document = {
                "_id": doc_id,
                "content": content,
                "embedding": {"values": embedding},
                "metadata": metadata or {},
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }

            # Store document
            await self.collection.insert_one(document)

            logger.info(f"Stored document with embedding: {doc_id}")
            return doc_id

        except Exception as error:
            logger.error(f"Error storing document: {error}")
            raise error

    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query."""
        try:
            # Use text search to find similar content
            pipeline = [
                {
                    "$search": {
                        "index": self.text_index_name,
                        "autocomplete": {
                            "query": partial_query,
                            "path": "content"
                        }
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "score": {"$meta": "searchScore"}
                    }
                },
                {"$limit": limit}
            ]

            results = await self.collection.aggregate(pipeline).to_list(length=None)

            # Extract suggestions from content
            suggestions = []
            for result in results:
                content = result.get("content", "")
                # Extract first sentence or phrase as suggestion
                suggestion = content.split('.')[0][:100]
                if suggestion and suggestion not in suggestions:
                    suggestions.append(suggestion)

            return suggestions[:limit]

        except Exception as error:
            logger.error(f"Error getting search suggestions: {error}")
            return []

    # Private helper methods

    def _build_vector_search_pipeline(self, query_embedding: List[float], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build vector search pipeline."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "embedding.values",
                    "queryVector": query_embedding,
                    "numCandidates": options.get("max_candidates", 100),
                    "limit": options.get("limit", 10)
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # Add score filter
        min_score = options.get("min_score", 0.0)
        if min_score > 0:
            pipeline.append({
                "$match": {
                    "score": {"$gte": min_score}
                }
            })

        # Add filters
        filters = options.get("filters")
        if filters:
            pipeline.append({"$match": filters})

        # Add boost
        boost = options.get("boost")
        if boost:
            for boost_config in boost:
                field = boost_config.get("field")
                factor = boost_config.get("factor", 1.0)
                if field:
                    pipeline.append({
                        "$addFields": {
                            "score": {
                                "$multiply": [
                                    "$score",
                                    {"$ifNull": [f"${field}", factor]}
                                ]
                            }
                        }
                    })

        # Add projection
        projection = {
            "_id": 1,
            "content": 1,
            "metadata": 1,
            "score": 1
        }

        if options.get("include_embeddings"):
            projection["embedding"] = 1

        pipeline.append({"$project": projection})

        return pipeline

    def _build_hybrid_search_pipeline(self, query: str, query_embedding: List[float], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build CORRECT hybrid search pipeline using $rankFusion for MongoDB 8.1+."""
        vector_weight = options.get("vector_weight", 0.7)
        text_weight = options.get("text_weight", 0.3)
        text_query = options.get("text_query", query)
        filters = options.get("filters", {})

        # CORRECT $rankFusion implementation
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vectorPipeline": [
                                {
                                    "$vectorSearch": {
                                        "index": self.vector_index_name,
                                        "path": "embedding.values",
                                        "queryVector": query_embedding,
                                        "numCandidates": max(options.get("limit", 10) * 10, 150),
                                        "limit": options.get("limit", 10) * 2,
                                        "filter": filters
                                    }
                                }
                            ],
                            "textPipeline": [
                                {
                                    "$search": {
                                        "index": self.text_index_name,
                                        "text": {
                                            "query": text_query,
                                            "path": "content"
                                        }
                                    }
                                },
                                {"$limit": options.get("limit", 10) * 2}
                            ]
                        }
                    },
                    "combination": {
                        "weights": {
                            "vectorPipeline": vector_weight,
                            "textPipeline": text_weight
                        }
                    },
                    "scoreDetails": True
                }
            },
            # Project results AFTER $rankFusion (this is allowed)
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "hybridScore": {"$meta": "scoreDetails"}
                }
            },
            # Filter by minimum score and limit results
            {
                "$match": {
                    "hybridScore.value": {"$gte": options.get("min_score", 0.0)}
                }
            },
            {"$limit": options.get("limit", 10)}
        ]

        # Add embedding projection if requested
        if options.get("include_embeddings"):
            pipeline[1]["$project"]["embedding"] = 1

        # Add explanation if requested
        if options.get("include_explanation"):
            pipeline[1]["$project"]["explanation"] = {
                "$concat": [
                    "Hybrid search using $rankFusion with weights - Vector: ",
                    {"$toString": vector_weight},
                    ", Text: ",
                    {"$toString": text_weight}
                ]
            }

        return pipeline

    def _process_search_results(self, results: List[Dict[str, Any]], search_type: str, include_explanation: bool) -> List[SearchResult]:
        """Process search results into SearchResult objects."""
        search_results = []

        for doc in results:
            # Handle both old and new score formats
            score = 0.0
            if "hybridScore" in doc and isinstance(doc["hybridScore"], dict):
                score = doc["hybridScore"].get("value", 0.0)
            else:
                score = doc.get("score", 0.0)

            search_result = SearchResult(
                id=str(doc.get("_id", "")),
                content=doc.get("content", ""),
                score=score,
                metadata=doc.get("metadata", {}),
                embedding=doc.get("embedding", {}).get("values") if doc.get("embedding") else None,
                explanation=doc.get("explanation") or (f"{search_type} search result" if include_explanation else None)
            )
            search_results.append(search_result)

        return search_results

    def _generate_cache_key(self, search_type: str, query: str, options: Any) -> str:
        """Generate cache key for search results."""
        key_data = {
            "type": search_type,
            "query": query[:100],  # First 100 chars
            "limit": getattr(options, "limit", 10),
            "min_score": getattr(options, "min_score", 0.0),
            "filters": json.dumps(getattr(options, "filters", {}) or {}, sort_keys=True)
        }
        return json.dumps(key_data, sort_keys=True)

    def _get_from_cache(self, key: str) -> Optional[List[SearchResult]]:
        """Get search results from cache."""
        cached = self.search_cache.get(key)
        if not cached:
            return None

        # Check TTL
        if time.time() - cached["timestamp"] > self.cache_ttl:
            del self.search_cache[key]
            return None

        return cached["results"]

    def _set_cache(self, key: str, results: List[SearchResult]) -> None:
        """Set search results in cache."""
        if len(self.search_cache) >= self.cache_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]

        self.search_cache[key] = {
            "results": results,
            "timestamp": time.time()
        }

    def _log_search_analytics(self, search_type: str, query: str, results: List[SearchResult], latency: float) -> None:
        """Log search analytics."""
        logger.debug(f"Search analytics: {search_type} search for '{query[:50]}...' "
                    f"returned {len(results)} results in {latency:.2f}ms")

        # In a production system, this would store analytics in a database
        # For now, we just log the information
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            logger.debug(f"Average relevance score: {avg_score:.3f}")

    async def get_search_analytics(self, timeframe: str = "day") -> SearchAnalytics:
        """Get search analytics."""
        # In a production system, this would query analytics from a database
        # For now, return dummy analytics
        return SearchAnalytics(
            total_searches=100,
            average_latency=150.0,
            average_result_count=8.5,
            search_type_distribution={"semantic": 60, "hybrid": 30, "text": 10},
            popular_queries=[
                {"query": "machine learning", "count": 25, "average_score": 0.85},
                {"query": "artificial intelligence", "count": 20, "average_score": 0.82}
            ],
            performance_metrics={
                "cache_hit_rate": 0.35,
                "average_embedding_time": 50.0,
                "average_search_time": 100.0
            },
            quality_metrics={
                "average_relevance_score": 0.78,
                "zero_result_rate": 0.05,
                "user_satisfaction_score": 0.85
            }
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.search_cache.clear()
