"""
Storage Manager for AI Brain Python

Coordinates MongoDB and Redis operations for optimal performance:
- Unified interface for storage operations
- Intelligent caching strategies
- Data consistency management
- Performance optimization
- Health monitoring
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ai_brain_python.storage.mongodb_client import MongoDBClient, MongoDBConfig
from ai_brain_python.storage.vector_store import VectorStore, VectorSearchConfig
from ai_brain_python.storage.cache_manager import CacheManager, CacheConfig

logger = logging.getLogger(__name__)


class StorageConfig(BaseModel):
    """Configuration for the storage manager."""
    
    mongodb: MongoDBConfig
    redis: CacheConfig
    vector_search: VectorSearchConfig
    
    # Cache strategies
    enable_caching: bool = Field(default=True, description="Enable Redis caching")
    cache_cognitive_states: bool = Field(default=True, description="Cache cognitive states")
    cache_search_results: bool = Field(default=True, description="Cache search results")
    cache_ttl_cognitive: int = Field(default=3600, description="TTL for cognitive state cache")
    cache_ttl_search: int = Field(default=1800, description="TTL for search result cache")
    
    # Performance settings
    batch_size: int = Field(default=100, description="Batch size for bulk operations")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")


class StorageManager:
    """Unified storage manager for AI Brain cognitive data."""
    
    def __init__(self, config: StorageConfig):
        """Initialize storage manager with configuration."""
        self.config = config
        
        # Initialize storage components
        self.mongodb_client = MongoDBClient(config.mongodb)
        self.cache_manager = CacheManager(config.redis) if config.enable_caching else None
        self.vector_store = VectorStore(self.mongodb_client, config.vector_search)
        
        self._initialization_lock = asyncio.Lock()
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize all storage components."""
        async with self._initialization_lock:
            if self._is_initialized:
                return
            
            try:
                logger.info("Initializing storage manager")
                
                # Initialize MongoDB
                await self.mongodb_client.connect()
                
                # Initialize Redis cache if enabled
                if self.cache_manager:
                    await self.cache_manager.connect()
                
                self._is_initialized = True
                logger.info("Storage manager initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize storage manager: {e}")
                raise
    
    async def shutdown(self) -> None:
        """Shutdown all storage components."""
        async with self._initialization_lock:
            if not self._is_initialized:
                return
            
            try:
                logger.info("Shutting down storage manager")
                
                # Shutdown cache manager
                if self.cache_manager:
                    await self.cache_manager.disconnect()
                
                # Shutdown MongoDB
                await self.mongodb_client.disconnect()
                
                self._is_initialized = False
                logger.info("Storage manager shutdown complete")
                
            except Exception as e:
                logger.error(f"Error during storage manager shutdown: {e}")
    
    # Cognitive State Operations
    
    async def store_cognitive_state(
        self, 
        system_id: str,
        user_id: str,
        state: Dict[str, Any],
        cache_ttl: Optional[int] = None
    ) -> str:
        """Store cognitive system state with caching."""
        try:
            # Store in MongoDB
            state_doc = {
                "system_id": system_id,
                "user_id": user_id,
                "state": state,
                "timestamp": datetime.utcnow()
            }
            
            document_id = await self.mongodb_client.insert_one(
                "cognitive_states", 
                state_doc
            )
            
            # Cache if enabled
            if self.cache_manager and self.config.cache_cognitive_states:
                cache_ttl = cache_ttl or self.config.cache_ttl_cognitive
                await self.cache_manager.cache_cognitive_state(
                    system_id, 
                    user_id, 
                    state, 
                    ttl=cache_ttl
                )
            
            logger.debug(f"Stored cognitive state for {system_id}:{user_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing cognitive state: {e}")
            raise
    
    async def get_cognitive_state(
        self, 
        system_id: str, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cognitive system state with cache fallback."""
        try:
            # Try cache first if enabled
            if self.cache_manager and self.config.cache_cognitive_states:
                cached_state = await self.cache_manager.get_cognitive_state(system_id, user_id)
                if cached_state:
                    logger.debug(f"Retrieved cognitive state from cache: {system_id}:{user_id}")
                    return cached_state
            
            # Fallback to MongoDB
            result = await self.mongodb_client.find_one(
                "cognitive_states",
                {"system_id": system_id, "user_id": user_id},
                {"state": 1, "timestamp": 1}
            )
            
            if result:
                state = result["state"]
                
                # Update cache if enabled
                if self.cache_manager and self.config.cache_cognitive_states:
                    await self.cache_manager.cache_cognitive_state(
                        system_id, 
                        user_id, 
                        state,
                        ttl=self.config.cache_ttl_cognitive
                    )
                
                logger.debug(f"Retrieved cognitive state from MongoDB: {system_id}:{user_id}")
                return state
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cognitive state: {e}")
            raise
    
    async def get_cognitive_state_history(
        self, 
        system_id: str, 
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get cognitive state history for a system and user."""
        try:
            results = await self.mongodb_client.find_many(
                "cognitive_states",
                {"system_id": system_id, "user_id": user_id},
                sort=[("timestamp", -1)],
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting cognitive state history: {e}")
            raise
    
    # Vector Storage Operations
    
    async def store_memory(
        self, 
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        source: Optional[str] = None
    ) -> str:
        """Store memory with vector embedding."""
        try:
            document_id = await self.vector_store.store_vector(
                content=content,
                embedding=embedding,
                metadata=metadata,
                user_id=user_id,
                source=source
            )
            
            # Invalidate related search caches
            if self.cache_manager and user_id:
                pattern = f"semantic_memory:*"
                await self.cache_manager.delete_pattern(pattern)
            
            logger.debug(f"Stored memory: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def search_memories(
        self, 
        query_embedding: List[float],
        user_id: Optional[str] = None,
        limit: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Search memories with caching."""
        try:
            # Generate cache key
            query_hash = None
            if self.cache_manager and self.config.cache_search_results and use_cache:
                query_data = {
                    "embedding": query_embedding,
                    "user_id": user_id,
                    "limit": limit
                }
                query_hash = hashlib.md5(str(query_data).encode()).hexdigest()
                
                # Try cache first
                cached_results = await self.cache_manager.get_semantic_memory_result(query_hash)
                if cached_results:
                    logger.debug("Retrieved search results from cache")
                    return cached_results
            
            # Perform vector search
            search_results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit
            )
            
            # Convert to dict format
            results = [
                {
                    "document": result.document.model_dump(),
                    "similarity_score": result.similarity_score,
                    "rank": result.rank
                }
                for result in search_results
            ]
            
            # Cache results if enabled
            if (self.cache_manager and self.config.cache_search_results and 
                query_hash and use_cache):
                await self.cache_manager.cache_semantic_memory_result(
                    query_hash, 
                    results,
                    ttl=self.config.cache_ttl_search
                )
            
            logger.debug(f"Found {len(results)} memories")
            return results
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
    
    async def hybrid_search_memories(
        self, 
        query_text: str,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid text and vector search."""
        try:
            search_results = await self.vector_store.hybrid_search(
                query_text=query_text,
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit
            )
            
            # Convert to dict format
            results = [
                {
                    "document": result.document.model_dump(),
                    "similarity_score": result.similarity_score,
                    "rank": result.rank
                }
                for result in search_results
            ]
            
            logger.debug(f"Hybrid search found {len(results)} memories")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise
    
    # Generic Storage Operations
    
    async def store_document(
        self, 
        collection_name: str, 
        document: Dict[str, Any]
    ) -> str:
        """Store a document in specified collection."""
        try:
            document_id = await self.mongodb_client.insert_one(collection_name, document)
            logger.debug(f"Stored document in {collection_name}: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document in {collection_name}: {e}")
            raise
    
    async def get_document(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get a document from specified collection."""
        try:
            result = await self.mongodb_client.find_one(collection_name, filter_dict)
            return result
            
        except Exception as e:
            logger.error(f"Error getting document from {collection_name}: {e}")
            raise
    
    async def update_document(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any]
    ) -> bool:
        """Update a document in specified collection."""
        try:
            success = await self.mongodb_client.update_one(
                collection_name, 
                filter_dict, 
                update_dict
            )
            
            # Invalidate related caches
            if self.cache_manager:
                # This is a simple invalidation strategy
                # In production, you might want more sophisticated cache invalidation
                await self.cache_manager.delete_pattern(f"*{collection_name}*")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating document in {collection_name}: {e}")
            raise
    
    # Cache Management
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cached data for a user."""
        if not self.cache_manager:
            return 0
        
        try:
            count = await self.cache_manager.invalidate_user_cache(user_id)
            logger.debug(f"Invalidated {count} cache entries for user {user_id}")
            return count
            
        except Exception as e:
            logger.error(f"Error invalidating user cache: {e}")
            raise
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        if not self.cache_manager:
            return 0
        
        try:
            pattern = pattern or "*"
            count = await self.cache_manager.delete_pattern(pattern)
            logger.debug(f"Cleared {count} cache entries")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise
    
    # Health and Monitoring
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            # Check MongoDB
            mongodb_health = await self.mongodb_client.health_check()
            health_status["components"]["mongodb"] = mongodb_health
            
            # Check Redis if enabled
            if self.cache_manager:
                redis_health = await self.cache_manager.health_check()
                health_status["components"]["redis"] = redis_health
            else:
                health_status["components"]["redis"] = {"status": "disabled"}
            
            # Check vector store
            try:
                vector_stats = await self.vector_store.get_statistics()
                health_status["components"]["vector_store"] = {
                    "status": "healthy",
                    "statistics": vector_stats
                }
            except Exception as e:
                health_status["components"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Determine overall status
            component_statuses = [
                comp.get("status", "unknown") 
                for comp in health_status["components"].values()
            ]
            
            if any(status == "unhealthy" for status in component_statuses):
                health_status["status"] = "unhealthy"
            elif any(status == "unknown" for status in component_statuses):
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "mongodb": {},
                "vector_store": {},
                "cache": {}
            }
            
            # MongoDB statistics
            for collection_name in self.mongodb_client.collections.values():
                count = await self.mongodb_client.count_documents(collection_name, {})
                stats["mongodb"][collection_name] = count
            
            # Vector store statistics
            vector_stats = await self.vector_store.get_statistics()
            stats["vector_store"] = vector_stats
            
            # Cache statistics (if available)
            if self.cache_manager:
                cache_health = await self.cache_manager.health_check()
                stats["cache"] = cache_health
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
