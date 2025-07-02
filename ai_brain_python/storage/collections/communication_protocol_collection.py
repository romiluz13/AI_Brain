"""
Communication Protocol Collection - MongoDB collection for communication pattern management

Handles storage and retrieval of communication protocols, interaction patterns, and communication analytics.
Provides insights for communication optimization and protocol adaptation.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class CommunicationProtocolCollection:
    """
    Communication Protocol Collection for MongoDB operations
    
    Manages communication protocols, interaction patterns, and communication analytics.
    Provides insights for communication optimization and protocol adaptation.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.communication_protocols
    
    async def create_indexes(self) -> None:
        """Create indexes for communication protocol collection."""
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
            
            # Protocol indexes
            await self.collection.create_index([
                ("protocolType", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="protocol_type_index", background=True)
            
            await self.collection.create_index([
                ("communicationStyle", ASCENDING)
            ], name="communication_style_index", background=True)
            
            await self.collection.create_index([
                ("adaptationLevel", DESCENDING)
            ], name="adaptation_level_index", background=True)
            
            # Pattern indexes
            await self.collection.create_index([
                ("interactionPatterns.patternType", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="pattern_type_index", background=True)
            
            await self.collection.create_index([
                ("interactionPatterns.effectiveness", DESCENDING)
            ], name="pattern_effectiveness_index", background=True)
            
            # Performance indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("protocolType", ASCENDING),
                ("adaptationLevel", DESCENDING)
            ], name="agent_protocol_performance_index", background=True)
            
            await self.collection.create_index([
                ("communicationMetrics.clarity", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="communication_clarity_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("communicationStyle", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="communication_analytics_index", background=True)
            
            # TTL index for automatic cleanup (90 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=7776000, background=True)
            
            logger.info("✅ CommunicationProtocolCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating CommunicationProtocolCollection indexes: {error}")
            raise error
    
    async def record_communication_protocol(self, protocol_record: Dict[str, Any]) -> ObjectId:
        """Record a communication protocol interaction."""
        try:
            result = await self.collection.insert_one(protocol_record)
            logger.debug(f"Communication protocol recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording communication protocol: {error}")
            raise error
    
    async def get_communication_analytics(
        self,
        agent_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get communication analytics for an agent."""
        try:
            # Default time range: last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            if options:
                if options.get("start_time"):
                    start_time = options["start_time"]
                if options.get("end_time"):
                    end_time = options["end_time"]
            
            # Aggregation pipeline for communication analytics
            pipeline = [
                {
                    "$match": {
                        "agentId": agent_id,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$protocolType",
                        "interactionCount": {"$sum": 1},
                        "avgAdaptationLevel": {"$avg": "$adaptationLevel"},
                        "avgClarity": {"$avg": "$communicationMetrics.clarity"},
                        "avgEffectiveness": {"$avg": "$communicationMetrics.effectiveness"},
                        "avgEngagement": {"$avg": "$communicationMetrics.engagement"},
                        "communicationStyles": {"$addToSet": "$communicationStyle"},
                        "patterns": {"$push": "$interactionPatterns"},
                        "optimizationRecommendations": {"$push": "$optimizationRecommendations"}
                    }
                },
                {
                    "$project": {
                        "protocolType": "$_id",
                        "interactionCount": 1,
                        "avgAdaptationLevel": {"$round": ["$avgAdaptationLevel", 3]},
                        "avgClarity": {"$round": ["$avgClarity", 3]},
                        "avgEffectiveness": {"$round": ["$avgEffectiveness", 3]},
                        "avgEngagement": {"$round": ["$avgEngagement", 3]},
                        "communicationStyles": 1,
                        "patterns": 1,
                        "optimizationRecommendations": 1,
                        "_id": 0
                    }
                }
            ]
            
            results = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Calculate overall analytics
            total_interactions = sum(r["interactionCount"] for r in results)
            
            if total_interactions > 0:
                overall_adaptation = sum(r["avgAdaptationLevel"] * r["interactionCount"] for r in results) / total_interactions
                overall_clarity = sum(r["avgClarity"] * r["interactionCount"] for r in results) / total_interactions
                overall_effectiveness = sum(r["avgEffectiveness"] * r["interactionCount"] for r in results) / total_interactions
            else:
                overall_adaptation = overall_clarity = overall_effectiveness = 0.0
            
            # Get top optimization recommendations
            all_recommendations = []
            for result in results:
                for rec_list in result.get("optimizationRecommendations", []):
                    if isinstance(rec_list, list):
                        all_recommendations.extend(rec_list)
            
            recommendation_counts = {}
            for rec in all_recommendations:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
            
            top_optimizations = sorted(
                recommendation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Analyze interaction patterns
            all_patterns = []
            for result in results:
                for pattern_list in result.get("patterns", []):
                    if isinstance(pattern_list, list):
                        all_patterns.extend(pattern_list)
            
            pattern_effectiveness = {}
            for pattern in all_patterns:
                if isinstance(pattern, dict) and "patternType" in pattern:
                    pattern_type = pattern["patternType"]
                    effectiveness = pattern.get("effectiveness", 0.0)
                    if pattern_type not in pattern_effectiveness:
                        pattern_effectiveness[pattern_type] = []
                    pattern_effectiveness[pattern_type].append(effectiveness)
            
            # Calculate average effectiveness per pattern
            pattern_stats = {}
            for pattern_type, effectiveness_list in pattern_effectiveness.items():
                pattern_stats[pattern_type] = {
                    "avgEffectiveness": round(sum(effectiveness_list) / len(effectiveness_list), 3),
                    "usageCount": len(effectiveness_list)
                }
            
            return {
                "protocolTypes": results,
                "totalInteractions": total_interactions,
                "overallAdaptationLevel": round(overall_adaptation, 3),
                "overallClarity": round(overall_clarity, 3),
                "overallEffectiveness": round(overall_effectiveness, 3),
                "protocolTypesCount": len(results),
                "topOptimizations": [opt[0] for opt in top_optimizations],
                "patternStats": pattern_stats
            }
                
        except Exception as error:
            logger.error(f"Error getting communication analytics: {error}")
            raise error
    
    async def get_communication_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get communication protocol statistics."""
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
            
            # Get protocol diversity and performance stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "uniqueProtocolTypes": {"$addToSet": "$protocolType"},
                        "uniqueCommunicationStyles": {"$addToSet": "$communicationStyle"},
                        "avgAdaptationLevel": {"$avg": "$adaptationLevel"},
                        "avgClarity": {"$avg": "$communicationMetrics.clarity"},
                        "avgEffectiveness": {"$avg": "$communicationMetrics.effectiveness"},
                        "maxAdaptationLevel": {"$max": "$adaptationLevel"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                return {
                    "totalInteractions": total_interactions,
                    "recentInteractions": recent_interactions,
                    "uniqueProtocolTypes": len(stats.get("uniqueProtocolTypes", [])),
                    "uniqueCommunicationStyles": len(stats.get("uniqueCommunicationStyles", [])),
                    "avgAdaptationLevel": round(stats.get("avgAdaptationLevel", 0.0), 3),
                    "avgClarity": round(stats.get("avgClarity", 0.0), 3),
                    "avgEffectiveness": round(stats.get("avgEffectiveness", 0.0), 3),
                    "maxAdaptationLevel": round(stats.get("maxAdaptationLevel", 0.0), 3),
                    "protocolTypes": stats.get("uniqueProtocolTypes", []),
                    "communicationStyles": stats.get("uniqueCommunicationStyles", [])
                }
            else:
                return {
                    "totalInteractions": total_interactions,
                    "recentInteractions": recent_interactions,
                    "uniqueProtocolTypes": 0,
                    "uniqueCommunicationStyles": 0,
                    "avgAdaptationLevel": 0.0,
                    "avgClarity": 0.0,
                    "avgEffectiveness": 0.0,
                    "maxAdaptationLevel": 0.0,
                    "protocolTypes": [],
                    "communicationStyles": []
                }
                
        except Exception as error:
            logger.error(f"Error getting communication stats: {error}")
            raise error
