"""
MemoryCollection - MongoDB collection for semantic memory management

Manages memory records with vector embeddings, similarity search,
and memory decay mechanisms using MongoDB Atlas Vector Search.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...utils.logger import logger


class MemoryCollection:
    """MongoDB collection for semantic memory management with vector search."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.semantic_memory
    
    async def create_indexes(self) -> None:
        """Create indexes for memory collection."""
        indexes = [
            # Primary indexes
            {"memoryId": 1},
            {"agentId": 1, "timestamp": -1},
            {"sessionId": 1, "timestamp": -1},
            
            # Memory type indexes
            {"type": 1, "timestamp": -1},
            {"importance": 1, "timestamp": -1},
            {"category": 1},
            
            # Content indexes
            {"content.summary": "text"},
            {"content.tags": 1},
            
            # Decay indexes
            {"decay.lastAccessed": -1},
            {"decay.accessCount": -1},
            {"decay.strength": 1},
            
            # Vector search index (Atlas Vector Search)
            # Note: This would be created through Atlas UI or API
            # {"embedding": "vectorSearch"},
            
            # TTL index for automatic cleanup (365 days)
            {"timestamp": 1}
        ]
        
        for index in indexes:
            try:
                if "timestamp" in index:
                    await self.collection.create_index(
                        list(index.items()),
                        expireAfterSeconds=60 * 60 * 24 * 365
                    )
                elif index.get("content.summary") == "text":
                    await self.collection.create_index([("content.summary", "text")])
                else:
                    await self.collection.create_index(list(index.items()))
            except Exception as error:
                logger.warning(f"Could not create memory index {index}: {error}")
    
    async def store_memory(self, memory: Dict[str, Any]) -> ObjectId:
        """Store memory record."""
        result = await self.collection.insert_one(memory)
        return result.inserted_id
    
    async def get_memory(self, memory_id: ObjectId) -> Optional[Dict[str, Any]]:
        """Get memory by ID."""
        return await self.collection.find_one({"memoryId": memory_id})
    
    async def search_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories using text search."""
        cursor = self.collection.find({
            "agentId": agent_id,
            "$text": {"$search": query}
        }).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def vector_search_memories(
        self,
        agent_id: str,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories using vector similarity (Atlas Vector Search)."""
        # This would use MongoDB Atlas Vector Search
        # For now, return text-based search as fallback
        return await self.search_memories(agent_id, "", limit)
    
    async def update_memory_access(self, memory_id: ObjectId) -> None:
        """Update memory access tracking for decay calculation."""
        await self.collection.update_one(
            {"memoryId": memory_id},
            {
                "$set": {"decay.lastAccessed": datetime.utcnow()},
                "$inc": {"decay.accessCount": 1}
            }
        )
    
    async def get_memories_for_decay(
        self,
        agent_id: str,
        decay_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Get memories that should be decayed."""
        cursor = self.collection.find({
            "agentId": agent_id,
            "decay.strength": {"$lte": decay_threshold}
        }).sort("decay.strength", 1)
        
        return await cursor.to_list(length=None)
    
    async def apply_memory_decay(self, memory_id: ObjectId, new_strength: float) -> None:
        """Apply decay to memory strength."""
        await self.collection.update_one(
            {"memoryId": memory_id},
            {"$set": {"decay.strength": new_strength}}
        )
