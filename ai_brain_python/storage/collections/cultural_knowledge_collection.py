"""
Cultural Knowledge Collection - MongoDB collection for cultural context and knowledge management

Handles storage and retrieval of cultural knowledge, context awareness, and cultural adaptation data.
Provides comprehensive cultural analytics and cross-cultural communication insights.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class CulturalKnowledgeCollection:
    """
    Cultural Knowledge Collection for MongoDB operations
    
    Manages cultural context data, cultural knowledge, and cross-cultural communication patterns.
    Provides analytics for cultural adaptation and awareness.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.cultural_knowledge
    
    async def create_indexes(self) -> None:
        """Create indexes for cultural knowledge collection."""
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
            
            # Cultural context indexes
            await self.collection.create_index([
                ("culturalContext.primaryCulture", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="primary_culture_index", background=True)
            
            await self.collection.create_index([
                ("culturalContext.communicationStyle", ASCENDING)
            ], name="communication_style_index", background=True)
            
            await self.collection.create_index([
                ("culturalContext.languagePreferences", ASCENDING)
            ], name="language_preferences_index", background=True)
            
            # Adaptation indexes
            await self.collection.create_index([
                ("adaptationScore", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="adaptation_score_index", background=True)
            
            await self.collection.create_index([
                ("culturalSensitivity", DESCENDING)
            ], name="cultural_sensitivity_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("culturalContext.primaryCulture", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_culture_analytics_index", background=True)
            
            # TTL index for automatic cleanup (90 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=7776000, background=True)
            
            logger.info("✅ CulturalKnowledgeCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating CulturalKnowledgeCollection indexes: {error}")
            raise error
    
    async def record_cultural_interaction(self, cultural_record: Dict[str, Any]) -> ObjectId:
        """Record a cultural interaction."""
        try:
            result = await self.collection.insert_one(cultural_record)
            logger.debug(f"Cultural interaction recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording cultural interaction: {error}")
            raise error
    
    async def get_cultural_analytics(
        self,
        agent_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get cultural analytics for an agent."""
        try:
            # Default time range: last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            if options:
                if options.get("start_time"):
                    start_time = options["start_time"]
                if options.get("end_time"):
                    end_time = options["end_time"]
            
            # Aggregation pipeline for cultural analytics
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
                        "totalInteractions": {"$sum": 1},
                        "avgAdaptationScore": {"$avg": "$adaptationScore"},
                        "avgCulturalSensitivity": {"$avg": "$culturalSensitivity"},
                        "cultures": {"$addToSet": "$culturalContext.primaryCulture"},
                        "communicationStyles": {"$addToSet": "$culturalContext.communicationStyle"},
                        "adaptationRecommendations": {"$push": "$adaptationRecommendations"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                analytics = result[0]
                analytics.pop("_id", None)
                
                # Flatten adaptation recommendations
                all_recommendations = []
                for rec_list in analytics.get("adaptationRecommendations", []):
                    if isinstance(rec_list, list):
                        all_recommendations.extend(rec_list)
                
                # Get top recommendations
                recommendation_counts = {}
                for rec in all_recommendations:
                    recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
                
                top_recommendations = sorted(
                    recommendation_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                analytics["topRecommendations"] = [rec[0] for rec in top_recommendations]
                analytics["adaptationRecommendations"] = len(all_recommendations)
                
                return analytics
            else:
                return {
                    "totalInteractions": 0,
                    "avgAdaptationScore": 0.0,
                    "avgCulturalSensitivity": 0.0,
                    "cultures": [],
                    "communicationStyles": [],
                    "topRecommendations": []
                }
                
        except Exception as error:
            logger.error(f"Error getting cultural analytics: {error}")
            raise error
    
    async def get_cultural_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cultural knowledge statistics."""
        try:
            match_filter = {}
            if agent_id:
                match_filter["agentId"] = agent_id
            
            # Get basic stats
            total_interactions = await self.collection.count_documents(match_filter)
            
            # Get recent stats (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_filter = {**match_filter, "timestamp": {"$gte": recent_time}}
            recent_interactions = await self.collection.count_documents(recent_filter)
            
            # Get cultural diversity stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "uniqueCultures": {"$addToSet": "$culturalContext.primaryCulture"},
                        "uniqueStyles": {"$addToSet": "$culturalContext.communicationStyle"},
                        "avgAdaptation": {"$avg": "$adaptationScore"},
                        "avgSensitivity": {"$avg": "$culturalSensitivity"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                return {
                    "totalInteractions": total_interactions,
                    "recentInteractions": recent_interactions,
                    "uniqueCultures": len(stats.get("uniqueCultures", [])),
                    "uniqueCommunicationStyles": len(stats.get("uniqueStyles", [])),
                    "avgAdaptationScore": round(stats.get("avgAdaptation", 0.0), 3),
                    "avgCulturalSensitivity": round(stats.get("avgSensitivity", 0.0), 3),
                    "cultures": stats.get("uniqueCultures", []),
                    "communicationStyles": stats.get("uniqueStyles", [])
                }
            else:
                return {
                    "totalInteractions": total_interactions,
                    "recentInteractions": recent_interactions,
                    "uniqueCultures": 0,
                    "uniqueCommunicationStyles": 0,
                    "avgAdaptationScore": 0.0,
                    "avgCulturalSensitivity": 0.0,
                    "cultures": [],
                    "communicationStyles": []
                }
                
        except Exception as error:
            logger.error(f"Error getting cultural stats: {error}")
            raise error
    
    async def find_cultural_patterns(
        self,
        agent_id: str,
        culture: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find cultural interaction patterns."""
        try:
            match_filter = {"agentId": agent_id}
            if culture:
                match_filter["culturalContext.primaryCulture"] = culture
            
            cursor = self.collection.find(match_filter).sort("timestamp", DESCENDING).limit(limit)
            patterns = await cursor.to_list(length=limit)
            
            return patterns
            
        except Exception as error:
            logger.error(f"Error finding cultural patterns: {error}")
            raise error
