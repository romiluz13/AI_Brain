"""
Cache Manager for AI Brain Python

Provides MongoDB Atlas-based caching for cognitive system data:
- High-performance MongoDB Atlas 8.1 caching with TTL support
- Advanced indexing and aggregation pipelines
- Real-time change streams for updates
- Vector search integration
- Pure MongoDB Atlas - no Redis, no in-memory
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """MongoDB Atlas cache configuration."""

    # MongoDB Atlas connection
    mongodb_uri: str = Field(..., description="MongoDB Atlas connection URI")
    database_name: str = Field(default="ai_brain_cache", description="Cache database name")

    # Cache settings
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    key_prefix: str = Field(default="ai_brain:", description="Key prefix for namespacing")

    # Collection settings
    cache_collection: str = Field(default="cache_data", description="Main cache collection")
    metrics_collection: str = Field(default="cache_metrics", description="Cache metrics collection")


class CacheManager:
    """MongoDB Atlas cache manager for AI Brain cognitive data."""

    def __init__(self, config: CacheConfig):
        """Initialize cache manager with MongoDB Atlas configuration."""
        self.config = config
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.cache_collection: Optional[AsyncIOMotorCollection] = None
        self.metrics_collection: Optional[AsyncIOMotorCollection] = None
        self._lock = asyncio.Lock()
        self._is_connected = False

        # Cache key patterns
        self.key_patterns = {
            "cognitive_state": "cognitive_state:{system_id}:{user_id}",
            "semantic_memory": "semantic_memory:{query_hash}",
            "goal_hierarchy": "goal_hierarchy:{user_id}",
            "emotional_state": "emotional_state:{user_id}",
            "attention_state": "attention_state:{user_id}",
            "confidence_tracking": "confidence_tracking:{user_id}",
            "monitoring_metrics": "monitoring_metrics:{metric_type}:{timestamp}",
            "session": "session:{session_id}",
            "user_context": "user_context:{user_id}",
        }

    async def connect(self) -> None:
        """Connect to MongoDB Atlas for caching."""
        async with self._lock:
            if self._is_connected:
                return

            try:
                logger.info("Connecting to MongoDB Atlas for caching")

                # Connect to MongoDB Atlas
                self.client = AsyncIOMotorClient(self.config.mongodb_uri)
                self.database = self.client[self.config.database_name]
                self.cache_collection = self.database[self.config.cache_collection]
                self.metrics_collection = self.database[self.config.metrics_collection]

                # Test connection
                await self.client.admin.command('ping')

                # Create TTL index for automatic expiration
                await self.cache_collection.create_index(
                    "expires_at",
                    expireAfterSeconds=0,
                    background=True
                )

                # Create indexes for performance
                await self.cache_collection.create_index("key", unique=True, background=True)
                await self.cache_collection.create_index("category", background=True)

                self._is_connected = True
                logger.info("Successfully connected to MongoDB Atlas cache")

            except Exception as e:
                logger.error(f"Failed to connect to MongoDB Atlas cache: {e}")
                raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB Atlas cache."""
        async with self._lock:
            if self._is_connected and self.client:
                self.client.close()
                self._is_connected = False
                logger.info("Disconnected from MongoDB Atlas cache")
    
    def _get_key(self, pattern: str, **kwargs) -> str:
        """Generate cache key from pattern and parameters."""
        if pattern not in self.key_patterns:
            raise ValueError(f"Unknown key pattern: {pattern}")
        
        key = self.key_patterns[pattern].format(**kwargs)
        return f"{self.config.key_prefix}{key}"
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        category: str = "general"
    ) -> bool:
        """Set a value in MongoDB Atlas cache."""
        try:
            if not self.cache_collection:
                raise RuntimeError("MongoDB cache not initialized")

            ttl = ttl or self.config.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            # Create cache document
            cache_doc = {
                "key": key,
                "value": value,
                "category": category,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at,
                "ttl": ttl
            }

            # Upsert the document
            result = await self.cache_collection.replace_one(
                {"key": key},
                cache_doc,
                upsert=True
            )

            logger.debug(f"Set cache key: {key} (TTL: {ttl}s, Category: {category})")
            return True

        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            raise

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from MongoDB Atlas cache."""
        try:
            if not self.cache_collection:
                raise RuntimeError("MongoDB cache not initialized")

            # Find the cache document
            doc = await self.cache_collection.find_one({"key": key})

            if doc is None:
                logger.debug(f"Cache miss for key: {key}")
                return None

            # Check if expired (extra safety, TTL index should handle this)
            if doc.get("expires_at") and doc["expires_at"] < datetime.utcnow():
                logger.debug(f"Cache key expired: {key}")
                await self.cache_collection.delete_one({"key": key})
                return None

            logger.debug(f"Cache hit for key: {key}")
            return doc.get("value")

        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete a value from MongoDB Atlas cache."""
        try:
            if not self.cache_collection:
                raise RuntimeError("MongoDB cache not initialized")

            result = await self.cache_collection.delete_one({"key": key})
            logger.debug(f"Deleted cache key: {key}")
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def clear_category(self, category: str) -> int:
        """Clear all cache entries in a specific category."""
        try:
            if not self.cache_collection:
                raise RuntimeError("MongoDB cache not initialized")

            result = await self.cache_collection.delete_many({"category": category})
            logger.info(f"Cleared {result.deleted_count} cache entries in category: {category}")
            return result.deleted_count

        except Exception as e:
            logger.error(f"Error clearing cache category {category}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists in MongoDB Atlas cache."""
        try:
            if not self.cache_collection:
                raise RuntimeError("MongoDB cache not initialized")

            doc = await self.cache_collection.find_one({"key": key}, {"_id": 1})
            return doc is not None

        except Exception as e:
            logger.error(f"Error checking cache key existence {key}: {e}")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from MongoDB Atlas."""
        try:
            if not self.cache_collection:
                raise RuntimeError("MongoDB cache not initialized")

            # Aggregate cache statistics
            pipeline = [
                {
                    "$group": {
                        "_id": "$category",
                        "count": {"$sum": 1},
                        "total_size": {"$sum": {"$bsonSize": "$$ROOT"}}
                    }
                }
            ]

            stats = {}
            async for doc in self.cache_collection.aggregate(pipeline):
                stats[doc["_id"]] = {
                    "count": doc["count"],
                    "size_bytes": doc["total_size"]
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    # Cognitive system specific cache methods

    async def cache_cognitive_state(
        self,
        system_id: str,
        user_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache cognitive system state."""
        key = self._get_key("cognitive_state", system_id=system_id, user_id=user_id)
        return await self.set(key, state, ttl=ttl, category="cognitive_state")

    async def get_cognitive_state(self, system_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached cognitive system state."""
        key = self._get_key("cognitive_state", system_id=system_id, user_id=user_id)
        return await self.get(key)

    async def cache_semantic_memory_result(
        self,
        query_hash: str,
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache semantic memory search results."""
        key = self._get_key("semantic_memory", query_hash=query_hash)
        return await self.set(key, results, ttl=ttl, category="semantic_memory")

    async def get_semantic_memory_result(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached semantic memory search results."""
        key = self._get_key("semantic_memory", query_hash=query_hash)
        return await self.get(key)

    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a user."""
        if not self.cache_collection:
            return 0

        # Delete all cache entries containing the user_id
        result = await self.cache_collection.delete_many(
            {"key": {"$regex": f".*{user_id}.*"}}
        )
        return result.deleted_count

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
