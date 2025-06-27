"""
Cache Manager for AI Brain Python

Provides Redis-based caching for cognitive system data:
- High-performance caching with TTL support
- Pub/Sub for real-time updates
- Distributed locking for coordination
- Cache invalidation strategies
- Performance monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Redis cache configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    database: int = Field(default=0, description="Redis database number")
    
    # Connection pool settings
    max_connections: int = Field(default=100, description="Maximum connections in pool")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(default=5.0, description="Connection timeout")
    
    # Cache settings
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    key_prefix: str = Field(default="ai_brain:", description="Key prefix for namespacing")
    
    # Serialization
    encoding: str = Field(default="utf-8", description="String encoding")
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """Get Redis connection parameters."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.database,
            "encoding": self.encoding,
            "decode_responses": True,
            "max_connections": self.max_connections,
            "retry_on_timeout": self.retry_on_timeout,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
        }
        
        if self.password:
            kwargs["password"] = self.password
        
        return kwargs


class CacheManager:
    """Redis-based cache manager for AI Brain cognitive data."""
    
    def __init__(self, config: CacheConfig):
        """Initialize cache manager with Redis configuration."""
        self.config = config
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self._connection_lock = asyncio.Lock()
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
        """Establish connection to Redis."""
        async with self._connection_lock:
            if self._is_connected:
                return
            
            try:
                logger.info(f"Connecting to Redis: {self.config.host}:{self.config.port}")
                
                # Create connection pool
                self.redis_pool = redis.ConnectionPool(**self.config.get_connection_kwargs())
                self.redis_client = redis.Redis(connection_pool=self.redis_pool)
                
                # Test connection
                await self.redis_client.ping()
                self._is_connected = True
                
                logger.info("Successfully connected to Redis")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        async with self._connection_lock:
            if self.redis_client and self._is_connected:
                await self.redis_client.close()
                if self.redis_pool:
                    await self.redis_pool.disconnect()
                self._is_connected = False
                logger.info("Disconnected from Redis")
    
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
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set a value in cache."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            ttl = ttl or self.config.default_ttl
            
            # Set value with options
            result = await self.redis_client.set(
                key, 
                serialized_value, 
                ex=ttl,
                nx=nx,
                xx=xx
            )
            
            logger.debug(f"Set cache key: {key} (TTL: {ttl}s)")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            value = await self.redis_client.get(key)
            
            if value is None:
                return None
            
            # Try to deserialize as JSON, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            raise
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            result = await self.redis_client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            raise
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            result = await self.redis_client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error checking cache key existence {key}: {e}")
            raise
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            result = await self.redis_client.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting expiration for cache key {key}: {e}")
            raise
    
    async def ttl(self, key: str) -> int:
        """Get time to live for a key."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            result = await self.redis_client.ttl(key)
            return result
            
        except Exception as e:
            logger.error(f"Error getting TTL for cache key {key}: {e}")
            raise
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            result = await self.redis_client.incrby(key, amount)
            return result
            
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            raise
    
    async def set_hash(self, key: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set a hash in cache."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            # Serialize hash values
            serialized_mapping = {}
            for field, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[field] = json.dumps(value, default=str)
                else:
                    serialized_mapping[field] = str(value)
            
            result = await self.redis_client.hset(key, mapping=serialized_mapping)
            
            if ttl:
                await self.redis_client.expire(key, ttl)
            
            logger.debug(f"Set hash cache key: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting hash cache key {key}: {e}")
            raise
    
    async def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a hash from cache."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            hash_data = await self.redis_client.hgetall(key)
            
            if not hash_data:
                return None
            
            # Deserialize hash values
            result = {}
            for field, value in hash_data.items():
                try:
                    result[field] = json.loads(value)
                except json.JSONDecodeError:
                    result[field] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting hash cache key {key}: {e}")
            raise
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            keys = await self.redis_client.keys(pattern)
            if keys:
                result = await self.redis_client.delete(*keys)
                logger.debug(f"Deleted {result} keys matching pattern: {pattern}")
                return result
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting keys with pattern {pattern}: {e}")
            raise
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish a message to a channel."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            # Serialize message
            if isinstance(message, (dict, list)):
                serialized_message = json.dumps(message, default=str)
            else:
                serialized_message = str(message)
            
            result = await self.redis_client.publish(channel, serialized_message)
            logger.debug(f"Published message to channel: {channel}")
            return result
            
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            raise
    
    async def subscribe(self, channels: List[str]) -> redis.client.PubSub:
        """Subscribe to channels."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(*channels)
            
            logger.debug(f"Subscribed to channels: {channels}")
            return pubsub
            
        except Exception as e:
            logger.error(f"Error subscribing to channels {channels}: {e}")
            raise
    
    async def acquire_lock(
        self, 
        lock_name: str, 
        timeout: int = 10,
        blocking_timeout: Optional[int] = None
    ) -> Optional[redis.lock.Lock]:
        """Acquire a distributed lock."""
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")
            
            lock_key = f"{self.config.key_prefix}lock:{lock_name}"
            lock = self.redis_client.lock(
                lock_key, 
                timeout=timeout,
                blocking_timeout=blocking_timeout
            )
            
            acquired = await lock.acquire()
            if acquired:
                logger.debug(f"Acquired lock: {lock_name}")
                return lock
            
            return None
            
        except Exception as e:
            logger.error(f"Error acquiring lock {lock_name}: {e}")
            raise
    
    async def release_lock(self, lock: redis.lock.Lock) -> bool:
        """Release a distributed lock."""
        try:
            result = await lock.release()
            logger.debug("Released lock")
            return result
            
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
            raise
    
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
        return await self.set(key, state, ttl=ttl)
    
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
        return await self.set(key, results, ttl=ttl)
    
    async def get_semantic_memory_result(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached semantic memory search results."""
        key = self._get_key("semantic_memory", query_hash=query_hash)
        return await self.get(key)
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a user."""
        pattern = f"{self.config.key_prefix}*:{user_id}*"
        return await self.delete_pattern(pattern)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        try:
            if not self.redis_client:
                return {"status": "disconnected", "error": "No client connection"}
            
            # Ping Redis
            await self.redis_client.ping()
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
                "uptime": info.get("uptime_in_seconds"),
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
