"""
Working Memory Collection - MongoDB collection for session-based memory management

Handles storage and retrieval of working memory data, session isolation, and memory analytics.
Provides session-specific memory management with TTL and priority-based eviction.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class WorkingMemoryCollection:
    """
    Working Memory Collection for MongoDB operations
    
    Manages working memory data with session isolation, TTL, and priority-based eviction.
    Provides session-specific memory management and analytics.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.working_memory
    
    async def create_indexes(self) -> None:
        """Create indexes for working memory collection."""
        try:
            # Primary indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("sessionId", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_session_timestamp_index", background=True)
            
            await self.collection.create_index([
                ("memoryId", ASCENDING)
            ], name="memory_id_index", background=True, unique=True)
            
            # Session-based indexes
            await self.collection.create_index([
                ("sessionId", ASCENDING),
                ("priority", DESCENDING)
            ], name="session_priority_index", background=True)
            
            await self.collection.create_index([
                ("sessionId", ASCENDING),
                ("accessCount", DESCENDING)
            ], name="session_access_index", background=True)
            
            await self.collection.create_index([
                ("sessionId", ASCENDING),
                ("lastAccessed", DESCENDING)
            ], name="session_last_accessed_index", background=True)
            
            # TTL and expiration indexes
            await self.collection.create_index([
                ("expiresAt", ASCENDING)
            ], name="expires_at_index", background=True)
            
            await self.collection.create_index([
                ("expiresAt", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=0, background=True)
            
            # Priority and eviction indexes
            await self.collection.create_index([
                ("priority", ASCENDING),
                ("accessCount", ASCENDING)
            ], name="eviction_priority_index", background=True)
            
            await self.collection.create_index([
                ("isPromoted", ASCENDING),
                ("accessCount", DESCENDING)
            ], name="promotion_index", background=True)
            
            # Content search indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("sessionId", ASCENDING),
                ("content", "text")
            ], name="content_search_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("createdAt", DESCENDING)
            ], name="agent_analytics_index", background=True)
            
            logger.info("✅ WorkingMemoryCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating WorkingMemoryCollection indexes: {error}")
            raise error
    
    async def store_working_memory(self, memory_record: Dict[str, Any]) -> ObjectId:
        """Store a working memory record."""
        try:
            result = await self.collection.insert_one(memory_record)
            logger.debug(f"Working memory stored: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error storing working memory: {error}")
            raise error
    
    async def find_working_memories(
        self,
        agent_id: str,
        session_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find working memories for a session."""
        try:
            match_filter = {
                "agentId": agent_id,
                "sessionId": session_id,
                "expiresAt": {"$gt": datetime.utcnow()}  # Only non-expired memories
            }
            
            if query:
                match_filter["$text"] = {"$search": query}
            
            cursor = self.collection.find(match_filter).sort([
                ("priority", DESCENDING),
                ("lastAccessed", DESCENDING)
            ]).limit(limit)
            
            memories = await cursor.to_list(length=limit)
            return memories
            
        except Exception as error:
            logger.error(f"Error finding working memories: {error}")
            raise error
    
    async def get_session_memory_count(self, session_id: str) -> int:
        """Get memory count for a session."""
        try:
            count = await self.collection.count_documents({
                "sessionId": session_id,
                "expiresAt": {"$gt": datetime.utcnow()}
            })
            return count
        except Exception as error:
            logger.error(f"Error getting session memory count: {error}")
            raise error
    
    async def find_low_priority_memories(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find low priority memories for eviction."""
        try:
            cursor = self.collection.find({
                "sessionId": session_id,
                "expiresAt": {"$gt": datetime.utcnow()}
            }).sort([
                ("priority", ASCENDING),
                ("accessCount", ASCENDING),
                ("lastAccessed", ASCENDING)
            ]).limit(limit)
            
            memories = await cursor.to_list(length=limit)
            return memories
            
        except Exception as error:
            logger.error(f"Error finding low priority memories: {error}")
            raise error
    
    async def find_high_access_memories(
        self,
        session_id: str,
        access_threshold: int = 3,
        importance_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find high-access memories for promotion."""
        try:
            cursor = self.collection.find({
                "sessionId": session_id,
                "accessCount": {"$gte": access_threshold},
                "priority": {"$gte": importance_threshold},
                "isPromoted": {"$ne": True},
                "expiresAt": {"$gt": datetime.utcnow()}
            }).sort("accessCount", DESCENDING)
            
            memories = await cursor.to_list(length=None)
            return memories
            
        except Exception as error:
            logger.error(f"Error finding high access memories: {error}")
            raise error
    
    async def update_memory_access(self, memory_id: ObjectId) -> None:
        """Update memory access count and timestamp."""
        try:
            await self.collection.update_one(
                {"memoryId": memory_id},
                {
                    "$inc": {"accessCount": 1},
                    "$set": {"lastAccessed": datetime.utcnow()}
                }
            )
        except Exception as error:
            logger.error(f"Error updating memory access: {error}")
            raise error
    
    async def mark_memory_promoted(self, memory_id: ObjectId) -> None:
        """Mark memory as promoted to long-term storage."""
        try:
            await self.collection.update_one(
                {"memoryId": memory_id},
                {"$set": {"isPromoted": True}}
            )
        except Exception as error:
            logger.error(f"Error marking memory as promoted: {error}")
            raise error
    
    async def delete_working_memory(self, memory_id: ObjectId) -> None:
        """Delete a working memory."""
        try:
            await self.collection.delete_one({"memoryId": memory_id})
        except Exception as error:
            logger.error(f"Error deleting working memory: {error}")
            raise error
    
    async def cleanup_expired_memories(self) -> int:
        """Clean up expired memories and return count."""
        try:
            result = await self.collection.delete_many({
                "expiresAt": {"$lte": datetime.utcnow()}
            })
            return result.deleted_count
        except Exception as error:
            logger.error(f"Error cleaning up expired memories: {error}")
            raise error
    
    async def get_working_memory_analytics(
        self,
        agent_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get working memory analytics."""
        try:
            match_filter = {"agentId": agent_id}
            if session_id:
                match_filter["sessionId"] = session_id
            
            # Aggregation pipeline for analytics
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": "$sessionId",
                        "memoryCount": {"$sum": 1},
                        "avgPriority": {"$avg": "$priority"},
                        "avgAccessCount": {"$avg": "$accessCount"},
                        "totalAccessCount": {"$sum": "$accessCount"},
                        "promotedCount": {"$sum": {"$cond": ["$isPromoted", 1, 0]}},
                        "expiredCount": {"$sum": {"$cond": [{"$lte": ["$expiresAt", datetime.utcnow()]}, 1, 0]}},
                        "activeCount": {"$sum": {"$cond": [{"$gt": ["$expiresAt", datetime.utcnow()]}, 1, 0]}}
                    }
                }
            ]
            
            results = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Calculate overall analytics
            total_memories = sum(r["memoryCount"] for r in results)
            total_access = sum(r["totalAccessCount"] for r in results)
            total_promoted = sum(r["promotedCount"] for r in results)
            total_active = sum(r["activeCount"] for r in results)
            
            if total_memories > 0:
                avg_priority = sum(r["avgPriority"] * r["memoryCount"] for r in results) / total_memories
                promotion_rate = total_promoted / total_memories
            else:
                avg_priority = promotion_rate = 0.0
            
            return {
                "sessions": results,
                "totalMemories": total_memories,
                "totalAccess": total_access,
                "totalPromoted": total_promoted,
                "totalActive": total_active,
                "avgPriority": round(avg_priority, 3),
                "promotionRate": round(promotion_rate, 3),
                "sessionCount": len(results)
            }
            
        except Exception as error:
            logger.error(f"Error getting working memory analytics: {error}")
            raise error
    
    async def get_working_memory_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get working memory statistics."""
        try:
            match_filter = {}
            if agent_id:
                match_filter["agentId"] = agent_id
            
            # Get basic stats
            total_memories = await self.collection.count_documents(match_filter)
            
            # Get active memories (non-expired)
            active_filter = {**match_filter, "expiresAt": {"$gt": datetime.utcnow()}}
            active_memories = await self.collection.count_documents(active_filter)
            
            # Get promoted memories
            promoted_filter = {**match_filter, "isPromoted": True}
            promoted_memories = await self.collection.count_documents(promoted_filter)
            
            # Get recent memories (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_filter = {**match_filter, "createdAt": {"$gte": recent_time}}
            recent_memories = await self.collection.count_documents(recent_filter)
            
            return {
                "totalMemories": total_memories,
                "activeMemories": active_memories,
                "promotedMemories": promoted_memories,
                "recentMemories": recent_memories,
                "expiredMemories": total_memories - active_memories,
                "promotionRate": round(promoted_memories / total_memories, 3) if total_memories > 0 else 0.0
            }
            
        except Exception as error:
            logger.error(f"Error getting working memory stats: {error}")
            raise error
