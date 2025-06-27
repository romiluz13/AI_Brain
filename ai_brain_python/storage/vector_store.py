"""
Vector Store for AI Brain Python

Provides vector storage and similarity search capabilities using MongoDB Atlas Vector Search:
- Embedding storage and retrieval
- Similarity search with various metrics
- Hybrid search combining vector and text search
- Batch operations for performance
- Memory consolidation and cleanup
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from ai_brain_python.storage.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class VectorSearchConfig(BaseModel):
    """Configuration for MongoDB Atlas Vector Search operations."""

    embedding_dimension: int = Field(default=1536, description="Dimension of embedding vectors (OpenAI ada-002)")
    similarity_metric: str = Field(default="cosine", description="Similarity metric (cosine, euclidean, dotProduct)")

    # Atlas Vector Search index configuration
    vector_index_name: str = Field(default="ai_brain_vector_index", description="Atlas Vector Search index name")
    text_index_name: str = Field(default="ai_brain_text_index", description="Atlas Search text index name")

    # Search parameters
    num_candidates: int = Field(default=150, description="Number of candidates for vector search (Atlas optimized)")
    default_limit: int = Field(default=10, description="Default limit for search results")
    min_score_threshold: float = Field(default=0.0, description="Minimum similarity score threshold")

    # Atlas-specific optimizations
    use_atlas_search: bool = Field(default=True, description="Use Atlas Search for hybrid queries")
    enable_exact_search: bool = Field(default=False, description="Enable exact search for high precision")


class VectorDocument(BaseModel):
    """Model for vector document storage."""
    
    content: str = Field(description="Text content of the document")
    embedding: List[float] = Field(description="Vector embedding of the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    user_id: Optional[str] = Field(default=None, description="User ID for scoped search")
    source: Optional[str] = Field(default=None, description="Source of the document")
    document_type: str = Field(default="memory", description="Type of document")
    relevance_score: float = Field(default=1.0, description="Relevance score for ranking")
    access_count: int = Field(default=0, description="Number of times accessed")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class SearchResult(BaseModel):
    """Model for search result."""
    
    document: VectorDocument
    similarity_score: float = Field(description="Similarity score from vector search")
    rank: int = Field(description="Rank in search results")


class VectorStore:
    """Vector store implementation using MongoDB Atlas Vector Search."""
    
    def __init__(self, mongodb_client: MongoDBClient, config: VectorSearchConfig):
        """Initialize vector store with MongoDB client and configuration."""
        self.mongodb_client = mongodb_client
        self.config = config
        self.collection_name = "semantic_memory"
    
    async def store_vector(
        self, 
        content: str, 
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        source: Optional[str] = None,
        document_type: str = "memory"
    ) -> str:
        """Store a vector document."""
        try:
            # Validate embedding dimension
            if len(embedding) != self.config.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} does not match "
                    f"configured dimension {self.config.embedding_dimension}"
                )
            
            # Create vector document
            vector_doc = VectorDocument(
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                user_id=user_id,
                source=source,
                document_type=document_type,
                relevance_score=1.0,
                access_count=0
            )
            
            # Convert to dict for storage
            doc_dict = vector_doc.model_dump(exclude_none=True)
            
            # Store in MongoDB
            document_id = await self.mongodb_client.insert_one(
                self.collection_name, 
                doc_dict
            )
            
            logger.debug(f"Stored vector document: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing vector document: {e}")
            raise
    
    async def store_vectors_batch(
        self, 
        documents: List[VectorDocument]
    ) -> List[str]:
        """Store multiple vector documents in batch."""
        try:
            # Validate all embeddings
            for doc in documents:
                if len(doc.embedding) != self.config.embedding_dimension:
                    raise ValueError(
                        f"Embedding dimension {len(doc.embedding)} does not match "
                        f"configured dimension {self.config.embedding_dimension}"
                    )
            
            # Convert to dicts for storage
            doc_dicts = [doc.model_dump(exclude_none=True) for doc in documents]
            
            # Store in MongoDB
            document_ids = await self.mongodb_client.insert_many(
                self.collection_name, 
                doc_dicts
            )
            
            logger.debug(f"Stored {len(documents)} vector documents in batch")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error storing vector documents in batch: {e}")
            raise
    
    async def similarity_search(
        self, 
        query_embedding: List[float],
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        document_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """Perform similarity search using vector embeddings."""
        try:
            # Validate query embedding dimension
            if len(query_embedding) != self.config.embedding_dimension:
                raise ValueError(
                    f"Query embedding dimension {len(query_embedding)} does not match "
                    f"configured dimension {self.config.embedding_dimension}"
                )
            
            limit = limit or self.config.default_limit
            min_score = min_score or self.config.min_score_threshold
            
            # Build aggregation pipeline for Atlas Vector Search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.config.vector_index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": self.config.num_candidates,
                        "limit": limit,
                        "exact": self.config.enable_exact_search
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Add filters
            match_conditions = {}
            
            if user_id:
                match_conditions["user_id"] = user_id
            
            if document_type:
                match_conditions["document_type"] = document_type
            
            if metadata_filter:
                for key, value in metadata_filter.items():
                    match_conditions[f"metadata.{key}"] = value
            
            if min_score > 0:
                match_conditions["similarity_score"] = {"$gte": min_score}
            
            if match_conditions:
                pipeline.append({"$match": match_conditions})
            
            # Execute search
            results = await self.mongodb_client.aggregate(self.collection_name, pipeline)
            
            # Convert results to SearchResult objects
            search_results = []
            for i, result in enumerate(results):
                # Remove MongoDB-specific fields
                result.pop("_id", None)
                similarity_score = result.pop("similarity_score", 0.0)
                
                # Create VectorDocument
                vector_doc = VectorDocument(**result)
                
                # Create SearchResult
                search_result = SearchResult(
                    document=vector_doc,
                    similarity_score=similarity_score,
                    rank=i + 1
                )
                search_results.append(search_result)
            
            # Update access counts for retrieved documents
            if search_results:
                await self._update_access_counts([r.document for r in search_results])
            
            logger.debug(f"Vector similarity search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    async def hybrid_search(
        self, 
        query_text: str,
        query_embedding: List[float],
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        text_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[SearchResult]:
        """Perform hybrid search combining text and vector search."""
        try:
            limit = limit or self.config.default_limit
            
            # Perform text search
            text_pipeline = [
                {
                    "$search": {
                        "index": "text_index",
                        "text": {
                            "query": query_text,
                            "path": "content"
                        }
                    }
                },
                {
                    "$addFields": {
                        "text_score": {"$meta": "searchScore"}
                    }
                },
                {"$limit": limit * 2}  # Get more candidates for hybrid ranking
            ]
            
            # Add user filter if specified
            if user_id:
                text_pipeline.append({"$match": {"user_id": user_id}})
            
            # Perform vector search
            vector_results = await self.similarity_search(
                query_embedding=query_embedding,
                limit=limit * 2,
                user_id=user_id
            )
            
            # Perform text search
            text_results = await self.mongodb_client.aggregate(
                self.collection_name, 
                text_pipeline
            )
            
            # Combine and rank results
            combined_results = {}
            
            # Add vector results
            for result in vector_results:
                doc_id = id(result.document)  # Use object id as key
                combined_results[doc_id] = {
                    "document": result.document,
                    "vector_score": result.similarity_score,
                    "text_score": 0.0
                }
            
            # Add text results
            for result in text_results:
                result.pop("_id", None)
                text_score = result.pop("text_score", 0.0)
                vector_doc = VectorDocument(**result)
                doc_id = id(vector_doc)
                
                if doc_id in combined_results:
                    combined_results[doc_id]["text_score"] = text_score
                else:
                    combined_results[doc_id] = {
                        "document": vector_doc,
                        "vector_score": 0.0,
                        "text_score": text_score
                    }
            
            # Calculate hybrid scores and rank
            hybrid_results = []
            for doc_id, scores in combined_results.items():
                hybrid_score = (
                    vector_weight * scores["vector_score"] + 
                    text_weight * scores["text_score"]
                )
                
                hybrid_results.append({
                    "document": scores["document"],
                    "hybrid_score": hybrid_score,
                    "vector_score": scores["vector_score"],
                    "text_score": scores["text_score"]
                })
            
            # Sort by hybrid score and limit results
            hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            hybrid_results = hybrid_results[:limit]
            
            # Convert to SearchResult objects
            search_results = []
            for i, result in enumerate(hybrid_results):
                search_result = SearchResult(
                    document=result["document"],
                    similarity_score=result["hybrid_score"],
                    rank=i + 1
                )
                search_results.append(search_result)
            
            logger.debug(f"Hybrid search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
    
    async def get_document_by_id(self, document_id: str) -> Optional[VectorDocument]:
        """Retrieve a document by its ID."""
        try:
            from bson import ObjectId
            
            result = await self.mongodb_client.find_one(
                self.collection_name,
                {"_id": ObjectId(document_id)}
            )
            
            if result:
                result.pop("_id", None)
                return VectorDocument(**result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document by ID: {e}")
            raise
    
    async def update_document(
        self, 
        document_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update a document by its ID."""
        try:
            from bson import ObjectId
            
            # Remove fields that shouldn't be updated directly
            updates.pop("_id", None)
            updates.pop("created_at", None)
            
            success = await self.mongodb_client.update_one(
                self.collection_name,
                {"_id": ObjectId(document_id)},
                {"$set": updates}
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by its ID."""
        try:
            from bson import ObjectId
            
            success = await self.mongodb_client.delete_one(
                self.collection_name,
                {"_id": ObjectId(document_id)}
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    async def _update_access_counts(self, documents: List[VectorDocument]) -> None:
        """Update access counts for retrieved documents."""
        try:
            # This would require document IDs, which we don't have in the current structure
            # For now, we'll skip this optimization
            pass
            
        except Exception as e:
            logger.error(f"Error updating access counts: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            total_documents = await self.mongodb_client.count_documents(
                self.collection_name, 
                {}
            )
            
            # Get documents by type
            type_pipeline = [
                {"$group": {"_id": "$document_type", "count": {"$sum": 1}}}
            ]
            type_stats = await self.mongodb_client.aggregate(
                self.collection_name, 
                type_pipeline
            )
            
            # Get average relevance score
            avg_pipeline = [
                {"$group": {"_id": None, "avg_relevance": {"$avg": "$relevance_score"}}}
            ]
            avg_stats = await self.mongodb_client.aggregate(
                self.collection_name, 
                avg_pipeline
            )
            
            avg_relevance = avg_stats[0]["avg_relevance"] if avg_stats else 0.0
            
            return {
                "total_documents": total_documents,
                "documents_by_type": {stat["_id"]: stat["count"] for stat in type_stats},
                "average_relevance_score": avg_relevance,
                "embedding_dimension": self.config.embedding_dimension,
                "similarity_metric": self.config.similarity_metric
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store statistics: {e}")
            raise
