"""
EmotionalStateCollection - MongoDB collection for agent emotional states

Exact Python equivalent of JavaScript EmotionalStateCollection.ts with:
- Time-series collection for emotional state tracking
- TTL indexes for automatic emotional decay
- Aggregation pipelines for emotional analytics
- Real-time emotional state monitoring
- Emotional trigger analysis and pattern detection

Features:
- Time-series optimization for emotional data
- TTL indexes for automatic emotional decay simulation
- Complex aggregation pipelines for emotional analytics
- Real-time change streams for emotional monitoring
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from ..base_collection import BaseCollection
from ...core.types import EmotionalState, EmotionalStateFilter, EmotionalStateUpdateData, EmotionalAnalyticsOptions
from ...utils.logger import logger


class EmotionalStateCollection(BaseCollection[EmotionalState]):
    """
    EmotionalStateCollection - Manages agent emotional states with time-series optimization
    
    Exact Python equivalent of JavaScript EmotionalStateCollection with:
    - Time-series collections for optimal emotional state storage
    - TTL indexes for automatic emotional decay simulation
    - Complex aggregation pipelines for emotional analytics
    - Real-time change streams for emotional monitoring
    """
    
    @property
    def collection_name(self) -> str:
        return "agent_emotional_states"
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)
        self.initialize_collection()
    
    async def create_indexes(self) -> None:
        """Create indexes optimized for emotional state queries and time-series operations."""
        try:
            # Time-series optimization indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agentId_timestamp_desc", background=True)
            
            # TTL index for automatic emotional decay
            await self.collection.create_index([
                ("expiresAt", ASCENDING)
            ], name="emotional_decay_ttl", expireAfterSeconds=0, background=True)
            
            # Emotional analytics indexes
            await self.collection.create_index([
                ("emotions.primary", ASCENDING),
                ("emotions.intensity", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="emotion_intensity_analysis", background=True)
            
            # Context-based emotional triggers
            await self.collection.create_index([
                ("context.triggerType", ASCENDING),
                ("emotions.valence", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="trigger_valence_analysis", background=True)
            
            # Session-based emotional tracking
            await self.collection.create_index([
                ("sessionId", ASCENDING),
                ("timestamp", ASCENDING)
            ], name="session_emotional_timeline", background=True, sparse=True)
            
            logger.info("✅ EmotionalStateCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating EmotionalStateCollection indexes: {error}")
            raise error
    
    async def record_emotional_state(self, emotional_state: Dict[str, Any]) -> ObjectId:
        """Record a new emotional state with automatic decay calculation."""
        # Calculate expiration time based on decay parameters
        expires_at = datetime.utcnow() + timedelta(
            minutes=emotional_state["decay"]["baselineReturn"]
        )
        
        state_with_expiry = {
            **emotional_state,
            "expiresAt": expires_at,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        
        result = await self.collection.insert_one(state_with_expiry)
        return result.inserted_id
    
    async def get_current_emotional_state(
        self, 
        agent_id: str, 
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get current emotional state for an agent (most recent non-expired)."""
        filter_dict = {
            "agentId": agent_id,
            "$or": [
                {"expiresAt": {"$gt": datetime.utcnow()}},
                {"expiresAt": {"$exists": False}}
            ]
        }
        
        if session_id:
            filter_dict["sessionId"] = session_id
        
        return await self.collection.find_one(
            filter_dict,
            sort=[("timestamp", DESCENDING)]
        )
    
    async def get_emotional_timeline(
        self,
        agent_id: str,
        options: Optional[EmotionalAnalyticsOptions] = None
    ) -> List[Dict[str, Any]]:
        """Get emotional timeline for an agent."""
        if options is None:
            options = EmotionalAnalyticsOptions()
        
        filter_dict = {"agentId": agent_id}
        
        if options.time_range:
            filter_dict["timestamp"] = {
                "$gte": options.time_range["start"],
                "$lte": options.time_range["end"]
            }
        
        if not options.include_decayed:
            filter_dict["$or"] = [
                {"expiresAt": {"$gt": datetime.utcnow()}},
                {"expiresAt": {"$exists": False}}
            ]
        
        if options.emotion_types:
            filter_dict["emotions.primary"] = {"$in": options.emotion_types}
        
        if options.min_intensity:
            filter_dict["emotions.intensity"] = {"$gte": options.min_intensity}
        
        cursor = self.collection.find(filter_dict).sort([("timestamp", ASCENDING)])
        return await cursor.to_list(length=None)
    
    async def analyze_emotional_patterns(self, agent_id: str, days: int = 7) -> Dict[str, Any]:
        """Analyze emotional patterns using MongoDB aggregation."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Dominant emotions analysis
        dominant_emotions_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": "$emotions.primary",
                    "frequency": {"$sum": 1},
                    "avgIntensity": {"$avg": "$emotions.intensity"},
                    "totalIntensity": {"$sum": "$emotions.intensity"}
                }
            },
            {
                "$sort": {"totalIntensity": -1}
            },
            {
                "$project": {
                    "emotion": "$_id",
                    "frequency": 1,
                    "avgIntensity": {"$round": ["$avgIntensity", 3]},
                    "_id": 0
                }
            }
        ]
        
        dominant_emotions = await self.collection.aggregate(dominant_emotions_pipeline).to_list(length=None)
        
        # Emotional stability (variance in valence)
        stability_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avgValence": {"$avg": "$emotions.valence"},
                    "valenceValues": {"$push": "$emotions.valence"}
                }
            },
            {
                "$project": {
                    "stability": {
                        "$subtract": [
                            1,
                            {
                                "$divide": [
                                    {"$stdDevPop": "$valenceValues"},
                                    2  # Max possible std dev for valence range [-1, 1]
                                ]
                            }
                        ]
                    }
                }
            }
        ]
        
        stability_result = await self.collection.aggregate(stability_pipeline).to_list(length=1)
        emotional_stability = stability_result[0]["stability"] if stability_result else 0
        
        # Trigger analysis
        trigger_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": "$context.triggerType",
                    "frequency": {"$sum": 1},
                    "avgValence": {"$avg": "$emotions.valence"}
                }
            },
            {
                "$project": {
                    "trigger": "$_id",
                    "frequency": 1,
                    "avgValence": {"$round": ["$avgValence", 3]},
                    "_id": 0
                }
            },
            {
                "$sort": {"frequency": -1}
            }
        ]
        
        trigger_analysis = await self.collection.aggregate(trigger_pipeline).to_list(length=None)
        
        # Temporal patterns (by hour of day)
        temporal_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": {"$hour": "$timestamp"},
                    "avgValence": {"$avg": "$emotions.valence"},
                    "avgArousal": {"$avg": "$emotions.arousal"}
                }
            },
            {
                "$project": {
                    "hour": "$_id",
                    "avgValence": {"$round": ["$avgValence", 3]},
                    "avgArousal": {"$round": ["$avgArousal", 3]},
                    "_id": 0
                }
            },
            {
                "$sort": {"hour": 1}
            }
        ]
        
        temporal_patterns = await self.collection.aggregate(temporal_pipeline).to_list(length=None)
        
        return {
            "dominantEmotions": dominant_emotions,
            "emotionalStability": emotional_stability,
            "triggerAnalysis": trigger_analysis,
            "temporalPatterns": temporal_patterns
        }
    
    async def cleanup_expired_states(self) -> int:
        """Clean up expired emotional states (manual cleanup for testing)."""
        result = await self.collection.delete_many({
            "expiresAt": {"$lte": datetime.utcnow()}
        })
        return result.deleted_count
    
    async def get_emotional_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get emotional state statistics."""
        filter_dict = {"agentId": agent_id} if agent_id else {}
        now = datetime.utcnow()
        
        stats_pipeline = [
            {"$match": filter_dict},
            {
                "$group": {
                    "_id": None,
                    "totalStates": {"$sum": 1},
                    "activeStates": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$or": [
                                        {"$gt": ["$expiresAt", now]},
                                        {"$eq": ["$expiresAt", None]}
                                    ]
                                },
                                1,
                                0
                            ]
                        }
                    },
                    "expiredStates": {
                        "$sum": {
                            "$cond": [
                                {"$lte": ["$expiresAt", now]},
                                1,
                                0
                            ]
                        }
                    },
                    "avgIntensity": {"$avg": "$emotions.intensity"},
                    "avgValence": {"$avg": "$emotions.valence"}
                }
            }
        ]
        
        stats_result = await self.collection.aggregate(stats_pipeline).to_list(length=1)
        
        return stats_result[0] if stats_result else {
            "totalStates": 0,
            "activeStates": 0,
            "expiredStates": 0,
            "avgIntensity": 0,
            "avgValence": 0
        }
