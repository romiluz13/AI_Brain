"""
Metacognitive State Collection - MongoDB collection for metacognitive awareness and self-monitoring

Handles storage and retrieval of metacognitive states, self-assessment data, and cognitive analytics.
Provides insights for metacognitive strategy optimization and self-awareness enhancement.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class MetacognitiveStateCollection:
    """
    Metacognitive State Collection for MongoDB operations
    
    Manages metacognitive states, self-assessment data, and cognitive analytics.
    Provides insights for metacognitive strategy optimization and self-awareness enhancement.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.metacognitive_states
    
    async def create_indexes(self) -> None:
        """Create indexes for metacognitive state collection."""
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
            
            # Metacognitive indexes
            await self.collection.create_index([
                ("awarenessId", ASCENDING)
            ], name="awareness_id_index", background=True, unique=True)
            
            await self.collection.create_index([
                ("currentTask", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="current_task_index", background=True)
            
            await self.collection.create_index([
                ("cognitiveLoad", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="cognitive_load_index", background=True)
            
            # Cognitive state indexes
            await self.collection.create_index([
                ("cognitiveState.clarity", DESCENDING)
            ], name="cognitive_clarity_index", background=True)
            
            await self.collection.create_index([
                ("cognitiveState.mode", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="cognitive_mode_index", background=True)
            
            await self.collection.create_index([
                ("cognitiveState.confidence", DESCENDING)
            ], name="cognitive_confidence_index", background=True)
            
            # Self-assessment indexes
            await self.collection.create_index([
                ("selfAssessment.overallAssessment", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="self_assessment_index", background=True)
            
            await self.collection.create_index([
                ("selfAssessment.performanceAccuracy", DESCENDING)
            ], name="performance_accuracy_index", background=True)
            
            # Strategy indexes
            await self.collection.create_index([
                ("metacognitiveStrategies.strategyId", ASCENDING)
            ], name="strategy_id_index", background=True)
            
            await self.collection.create_index([
                ("metacognitiveStrategies.effectiveness", DESCENDING)
            ], name="strategy_effectiveness_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("currentTask", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="metacognitive_analytics_index", background=True)
            
            # TTL index for automatic cleanup (90 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=7776000, background=True)
            
            logger.info("✅ MetacognitiveStateCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating MetacognitiveStateCollection indexes: {error}")
            raise error
    
    async def record_metacognitive_state(self, metacognitive_record: Dict[str, Any]) -> ObjectId:
        """Record a metacognitive state."""
        try:
            result = await self.collection.insert_one(metacognitive_record)
            logger.debug(f"Metacognitive state recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording metacognitive state: {error}")
            raise error
    
    async def get_metacognitive_analytics(
        self,
        agent_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get metacognitive analytics for an agent."""
        try:
            # Default time range: last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            if options:
                if options.get("start_time"):
                    start_time = options["start_time"]
                if options.get("end_time"):
                    end_time = options["end_time"]
            
            # Aggregation pipeline for metacognitive analytics
            pipeline = [
                {
                    "$match": {
                        "agentId": agent_id,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$currentTask",
                        "stateCount": {"$sum": 1},
                        "avgCognitiveLoad": {"$avg": "$cognitiveLoad"},
                        "avgCognitiveClarity": {"$avg": "$cognitiveState.clarity"},
                        "avgCognitiveConfidence": {"$avg": "$cognitiveState.confidence"},
                        "avgSelfAssessment": {"$avg": "$selfAssessment.overallAssessment"},
                        "avgPerformanceAccuracy": {"$avg": "$selfAssessment.performanceAccuracy"},
                        "cognitiveModes": {"$addToSet": "$cognitiveState.mode"},
                        "strategies": {"$push": "$metacognitiveStrategies"},
                        "optimizationRecommendations": {"$push": "$optimizationRecommendations"}
                    }
                },
                {
                    "$project": {
                        "currentTask": "$_id",
                        "stateCount": 1,
                        "avgCognitiveLoad": {"$round": ["$avgCognitiveLoad", 3]},
                        "avgCognitiveClarity": {"$round": ["$avgCognitiveClarity", 3]},
                        "avgCognitiveConfidence": {"$round": ["$avgCognitiveConfidence", 3]},
                        "avgSelfAssessment": {"$round": ["$avgSelfAssessment", 3]},
                        "avgPerformanceAccuracy": {"$round": ["$avgPerformanceAccuracy", 3]},
                        "cognitiveModes": 1,
                        "strategies": 1,
                        "optimizationRecommendations": 1,
                        "_id": 0
                    }
                }
            ]
            
            results = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Calculate overall analytics
            total_states = sum(r["stateCount"] for r in results)
            
            if total_states > 0:
                overall_cognitive_load = sum(r["avgCognitiveLoad"] * r["stateCount"] for r in results) / total_states
                overall_clarity = sum(r["avgCognitiveClarity"] * r["stateCount"] for r in results) / total_states
                overall_confidence = sum(r["avgCognitiveConfidence"] * r["stateCount"] for r in results) / total_states
                overall_self_assessment = sum(r["avgSelfAssessment"] * r["stateCount"] for r in results) / total_states
            else:
                overall_cognitive_load = overall_clarity = overall_confidence = overall_self_assessment = 0.0
            
            # Analyze strategies
            all_strategies = []
            for result in results:
                for strategy_list in result.get("strategies", []):
                    if isinstance(strategy_list, list):
                        all_strategies.extend(strategy_list)
            
            strategy_effectiveness = {}
            strategy_usage = {}
            for strategy in all_strategies:
                if isinstance(strategy, dict):
                    strategy_id = strategy.get("strategyId", "unknown")
                    effectiveness = strategy.get("effectiveness", 0.0)
                    
                    if strategy_id not in strategy_effectiveness:
                        strategy_effectiveness[strategy_id] = []
                        strategy_usage[strategy_id] = 0
                    
                    strategy_effectiveness[strategy_id].append(effectiveness)
                    strategy_usage[strategy_id] += 1
            
            # Calculate average effectiveness per strategy
            strategy_stats = {}
            for strategy_id, effectiveness_list in strategy_effectiveness.items():
                strategy_stats[strategy_id] = {
                    "avgEffectiveness": round(sum(effectiveness_list) / len(effectiveness_list), 3),
                    "usageCount": strategy_usage[strategy_id]
                }
            
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
            
            return {
                "tasks": results,
                "totalStates": total_states,
                "overallCognitiveLoad": round(overall_cognitive_load, 3),
                "overallCognitiveClarity": round(overall_clarity, 3),
                "overallCognitiveConfidence": round(overall_confidence, 3),
                "overallSelfAssessment": round(overall_self_assessment, 3),
                "taskCount": len(results),
                "strategyStats": strategy_stats,
                "topOptimizations": [opt[0] for opt in top_optimizations]
            }
                
        except Exception as error:
            logger.error(f"Error getting metacognitive analytics: {error}")
            raise error
    
    async def get_metacognitive_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metacognitive statistics."""
        try:
            match_filter = {}
            if agent_id:
                match_filter["agentId"] = agent_id
            
            # Get basic stats
            total_states = await self.collection.count_documents(match_filter)
            
            # Get recent stats (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_filter = {**match_filter, "timestamp": {"$gte": recent_time}}
            recent_states = await self.collection.count_documents(recent_filter)
            
            # Get high cognitive load states
            high_load_filter = {**match_filter, "cognitiveLoad": {"$gte": 0.8}}
            high_load_states = await self.collection.count_documents(high_load_filter)
            
            # Get metacognitive performance stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "uniqueTasks": {"$addToSet": "$currentTask"},
                        "uniqueCognitiveModes": {"$addToSet": "$cognitiveState.mode"},
                        "avgCognitiveLoad": {"$avg": "$cognitiveLoad"},
                        "avgCognitiveClarity": {"$avg": "$cognitiveState.clarity"},
                        "avgSelfAssessment": {"$avg": "$selfAssessment.overallAssessment"},
                        "maxCognitiveLoad": {"$max": "$cognitiveLoad"},
                        "strategies": {"$push": "$metacognitiveStrategies"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                
                # Count total strategies
                all_strategies = []
                for strategy_list in stats.get("strategies", []):
                    if isinstance(strategy_list, list):
                        all_strategies.extend(strategy_list)
                
                return {
                    "totalStates": total_states,
                    "recentStates": recent_states,
                    "highCognitiveLoadStates": high_load_states,
                    "uniqueTasks": len(stats.get("uniqueTasks", [])),
                    "uniqueCognitiveModes": len(stats.get("uniqueCognitiveModes", [])),
                    "avgCognitiveLoad": round(stats.get("avgCognitiveLoad", 0.0), 3),
                    "avgCognitiveClarity": round(stats.get("avgCognitiveClarity", 0.0), 3),
                    "avgSelfAssessment": round(stats.get("avgSelfAssessment", 0.0), 3),
                    "maxCognitiveLoad": round(stats.get("maxCognitiveLoad", 0.0), 3),
                    "totalStrategies": len(all_strategies),
                    "tasks": stats.get("uniqueTasks", []),
                    "cognitiveModes": stats.get("uniqueCognitiveModes", [])
                }
            else:
                return {
                    "totalStates": total_states,
                    "recentStates": recent_states,
                    "highCognitiveLoadStates": high_load_states,
                    "uniqueTasks": 0,
                    "uniqueCognitiveModes": 0,
                    "avgCognitiveLoad": 0.0,
                    "avgCognitiveClarity": 0.0,
                    "avgSelfAssessment": 0.0,
                    "maxCognitiveLoad": 0.0,
                    "totalStrategies": 0,
                    "tasks": [],
                    "cognitiveModes": []
                }
                
        except Exception as error:
            logger.error(f"Error getting metacognitive stats: {error}")
            raise error
