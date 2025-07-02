"""
Memory Decay Stats Collection - MongoDB collection for memory decay analytics and statistics

Handles storage and retrieval of memory decay processing data, decay analytics, and optimization insights.
Provides comprehensive memory evolution tracking and decay pattern analysis.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class MemoryDecayStatsCollection:
    """
    Memory Decay Stats Collection for MongoDB operations
    
    Manages memory decay processing data, decay analytics, and optimization insights.
    Provides comprehensive memory evolution tracking and decay pattern analysis.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.memory_decay_stats
    
    async def create_indexes(self) -> None:
        """Create indexes for memory decay stats collection."""
        try:
            # Primary indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_timestamp_index", background=True)
            
            await self.collection.create_index([
                ("sessionId", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="session_timestamp_index", background=True, sparse=True)
            
            # Decay processing indexes
            await self.collection.create_index([
                ("decayId", ASCENDING)
            ], name="decay_id_index", background=True, unique=True)
            
            await self.collection.create_index([
                ("memoryTypes", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="memory_types_index", background=True)
            
            await self.collection.create_index([
                ("processedMemories", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="processed_memories_index", background=True)
            
            # Decay metrics indexes
            await self.collection.create_index([
                ("decayedMemories", DESCENDING)
            ], name="decayed_memories_index", background=True)
            
            await self.collection.create_index([
                ("cleanedMemories", DESCENDING)
            ], name="cleaned_memories_index", background=True)
            
            await self.collection.create_index([
                ("preservedMemories", DESCENDING)
            ], name="preserved_memories_index", background=True)
            
            # Statistics indexes
            await self.collection.create_index([
                ("decayStatistics.decayRate", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="decay_rate_index", background=True)
            
            await self.collection.create_index([
                ("decayStatistics.cleanupRate", DESCENDING)
            ], name="cleanup_rate_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("memoryTypes", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="decay_analytics_index", background=True)
            
            await self.collection.create_index([
                ("forceCleanup", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="force_cleanup_index", background=True)
            
            # TTL index for automatic cleanup (180 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=15552000, background=True)
            
            logger.info("✅ MemoryDecayStatsCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating MemoryDecayStatsCollection indexes: {error}")
            raise error
    
    async def record_decay_processing(self, decay_record: Dict[str, Any]) -> ObjectId:
        """Record a memory decay processing session."""
        try:
            result = await self.collection.insert_one(decay_record)
            logger.debug(f"Memory decay processing recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording decay processing: {error}")
            raise error
    
    async def get_decay_analytics(
        self,
        agent_id: str,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get memory decay analytics for an agent."""
        try:
            # Time range for analytics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # Aggregation pipeline for decay analytics
            pipeline = [
                {
                    "$match": {
                        "agentId": agent_id,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "totalProcessingSessions": {"$sum": 1},
                        "totalProcessedMemories": {"$sum": "$processedMemories"},
                        "totalDecayedMemories": {"$sum": "$decayedMemories"},
                        "totalCleanedMemories": {"$sum": "$cleanedMemories"},
                        "totalPreservedMemories": {"$sum": "$preservedMemories"},
                        "avgDecayRate": {"$avg": "$decayStatistics.decayRate"},
                        "avgCleanupRate": {"$avg": "$decayStatistics.cleanupRate"},
                        "memoryTypes": {"$addToSet": "$memoryTypes"},
                        "forceCleanupSessions": {"$sum": {"$cond": ["$forceCleanup", 1, 0]}},
                        "preserveRecentSessions": {"$sum": {"$cond": ["$preserveRecent", 1, 0]}}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                analytics = result[0]
                analytics.pop("_id", None)
                
                # Calculate efficiency metrics
                total_processed = analytics.get("totalProcessedMemories", 0)
                total_decayed = analytics.get("totalDecayedMemories", 0)
                total_cleaned = analytics.get("totalCleanedMemories", 0)
                total_preserved = analytics.get("totalPreservedMemories", 0)
                
                if total_processed > 0:
                    decay_efficiency = total_decayed / total_processed
                    cleanup_efficiency = total_cleaned / total_processed
                    preservation_rate = total_preserved / total_processed
                else:
                    decay_efficiency = cleanup_efficiency = preservation_rate = 0.0
                
                # Flatten memory types
                all_memory_types = []
                for type_list in analytics.get("memoryTypes", []):
                    if isinstance(type_list, list):
                        all_memory_types.extend(type_list)
                
                unique_memory_types = list(set(all_memory_types))
                
                analytics.update({
                    "decayEfficiency": round(decay_efficiency, 3),
                    "cleanupEfficiency": round(cleanup_efficiency, 3),
                    "preservationRate": round(preservation_rate, 3),
                    "uniqueMemoryTypes": unique_memory_types,
                    "memoryTypesCount": len(unique_memory_types),
                    "avgDecayRate": round(analytics.get("avgDecayRate", 0.0), 3),
                    "avgCleanupRate": round(analytics.get("avgCleanupRate", 0.0), 3)
                })
                
                # Remove raw memory types array
                analytics.pop("memoryTypes", None)
                
                return analytics
            else:
                return {
                    "totalProcessingSessions": 0,
                    "totalProcessedMemories": 0,
                    "totalDecayedMemories": 0,
                    "totalCleanedMemories": 0,
                    "totalPreservedMemories": 0,
                    "avgDecayRate": 0.0,
                    "avgCleanupRate": 0.0,
                    "decayEfficiency": 0.0,
                    "cleanupEfficiency": 0.0,
                    "preservationRate": 0.0,
                    "uniqueMemoryTypes": [],
                    "memoryTypesCount": 0,
                    "forceCleanupSessions": 0,
                    "preserveRecentSessions": 0
                }
                
        except Exception as error:
            logger.error(f"Error getting decay analytics: {error}")
            raise error
    
    async def get_decay_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory decay statistics."""
        try:
            match_filter = {}
            if agent_id:
                match_filter["agentId"] = agent_id
            
            # Get basic stats
            total_sessions = await self.collection.count_documents(match_filter)
            
            # Get recent stats (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_filter = {**match_filter, "timestamp": {"$gte": recent_time}}
            recent_sessions = await self.collection.count_documents(recent_filter)
            
            # Get force cleanup sessions
            force_cleanup_filter = {**match_filter, "forceCleanup": True}
            force_cleanup_sessions = await self.collection.count_documents(force_cleanup_filter)
            
            # Get decay performance stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "totalProcessedMemories": {"$sum": "$processedMemories"},
                        "totalDecayedMemories": {"$sum": "$decayedMemories"},
                        "totalCleanedMemories": {"$sum": "$cleanedMemories"},
                        "totalPreservedMemories": {"$sum": "$preservedMemories"},
                        "avgDecayRate": {"$avg": "$decayStatistics.decayRate"},
                        "avgCleanupRate": {"$avg": "$decayStatistics.cleanupRate"},
                        "maxDecayRate": {"$max": "$decayStatistics.decayRate"},
                        "maxCleanupRate": {"$max": "$decayStatistics.cleanupRate"},
                        "uniqueMemoryTypes": {"$addToSet": "$memoryTypes"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                
                # Calculate efficiency metrics
                total_processed = stats.get("totalProcessedMemories", 0)
                total_decayed = stats.get("totalDecayedMemories", 0)
                total_cleaned = stats.get("totalCleanedMemories", 0)
                total_preserved = stats.get("totalPreservedMemories", 0)
                
                if total_processed > 0:
                    overall_decay_rate = total_decayed / total_processed
                    overall_cleanup_rate = total_cleaned / total_processed
                    overall_preservation_rate = total_preserved / total_processed
                else:
                    overall_decay_rate = overall_cleanup_rate = overall_preservation_rate = 0.0
                
                # Flatten memory types
                all_memory_types = []
                for type_list in stats.get("uniqueMemoryTypes", []):
                    if isinstance(type_list, list):
                        all_memory_types.extend(type_list)
                
                unique_types = list(set(all_memory_types))
                
                return {
                    "totalSessions": total_sessions,
                    "recentSessions": recent_sessions,
                    "forceCleanupSessions": force_cleanup_sessions,
                    "totalProcessedMemories": total_processed,
                    "totalDecayedMemories": total_decayed,
                    "totalCleanedMemories": total_cleaned,
                    "totalPreservedMemories": total_preserved,
                    "overallDecayRate": round(overall_decay_rate, 3),
                    "overallCleanupRate": round(overall_cleanup_rate, 3),
                    "overallPreservationRate": round(overall_preservation_rate, 3),
                    "avgDecayRate": round(stats.get("avgDecayRate", 0.0), 3),
                    "avgCleanupRate": round(stats.get("avgCleanupRate", 0.0), 3),
                    "maxDecayRate": round(stats.get("maxDecayRate", 0.0), 3),
                    "maxCleanupRate": round(stats.get("maxCleanupRate", 0.0), 3),
                    "uniqueMemoryTypes": unique_types,
                    "memoryTypesCount": len(unique_types)
                }
            else:
                return {
                    "totalSessions": total_sessions,
                    "recentSessions": recent_sessions,
                    "forceCleanupSessions": force_cleanup_sessions,
                    "totalProcessedMemories": 0,
                    "totalDecayedMemories": 0,
                    "totalCleanedMemories": 0,
                    "totalPreservedMemories": 0,
                    "overallDecayRate": 0.0,
                    "overallCleanupRate": 0.0,
                    "overallPreservationRate": 0.0,
                    "avgDecayRate": 0.0,
                    "avgCleanupRate": 0.0,
                    "maxDecayRate": 0.0,
                    "maxCleanupRate": 0.0,
                    "uniqueMemoryTypes": [],
                    "memoryTypesCount": 0
                }
                
        except Exception as error:
            logger.error(f"Error getting decay stats: {error}")
            raise error
    
    async def find_decay_patterns(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find memory decay patterns for analysis."""
        try:
            match_filter = {"agentId": agent_id}
            if memory_type:
                match_filter["memoryTypes"] = memory_type
            
            cursor = self.collection.find(match_filter).sort("timestamp", DESCENDING).limit(limit)
            patterns = await cursor.to_list(length=limit)
            
            return patterns
            
        except Exception as error:
            logger.error(f"Error finding decay patterns: {error}")
            raise error
