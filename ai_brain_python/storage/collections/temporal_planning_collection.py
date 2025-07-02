"""
Temporal Planning Collection - MongoDB collection for time-aware planning and scheduling

Handles storage and retrieval of temporal plans, scheduling data, and time-based analytics.
Provides insights for temporal optimization and planning effectiveness.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class TemporalPlanningCollection:
    """
    Temporal Planning Collection for MongoDB operations
    
    Manages temporal plans, scheduling data, and time-based analytics.
    Provides insights for temporal optimization and planning effectiveness.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.temporal_plans
    
    async def create_indexes(self) -> None:
        """Create indexes for temporal planning collection."""
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
            
            # Plan indexes
            await self.collection.create_index([
                ("planType", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="plan_type_index", background=True)
            
            await self.collection.create_index([
                ("planStatus", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="plan_status_index", background=True)
            
            await self.collection.create_index([
                ("priority", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="priority_index", background=True)
            
            # Temporal indexes
            await self.collection.create_index([
                ("timeHorizon.startTime", ASCENDING),
                ("timeHorizon.endTime", ASCENDING)
            ], name="time_horizon_index", background=True)
            
            await self.collection.create_index([
                ("scheduledTasks.scheduledTime", ASCENDING)
            ], name="scheduled_time_index", background=True)
            
            await self.collection.create_index([
                ("deadlines.deadline", ASCENDING)
            ], name="deadline_index", background=True)
            
            # Performance indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("planType", ASCENDING),
                ("executionMetrics.efficiency", DESCENDING)
            ], name="agent_plan_performance_index", background=True)
            
            await self.collection.create_index([
                ("executionMetrics.completionRate", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="completion_rate_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("planType", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="temporal_analytics_index", background=True)
            
            # TTL index for automatic cleanup (365 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=31536000, background=True)
            
            logger.info("✅ TemporalPlanningCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating TemporalPlanningCollection indexes: {error}")
            raise error
    
    async def record_temporal_plan(self, plan_record: Dict[str, Any]) -> ObjectId:
        """Record a temporal plan."""
        try:
            result = await self.collection.insert_one(plan_record)
            logger.debug(f"Temporal plan recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording temporal plan: {error}")
            raise error
    
    async def get_temporal_analytics(
        self,
        agent_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get temporal planning analytics for an agent."""
        try:
            # Default time range: last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            if options:
                if options.get("start_time"):
                    start_time = options["start_time"]
                if options.get("end_time"):
                    end_time = options["end_time"]
            
            # Aggregation pipeline for temporal analytics
            pipeline = [
                {
                    "$match": {
                        "agentId": agent_id,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$planType",
                        "planCount": {"$sum": 1},
                        "avgPriority": {"$avg": "$priority"},
                        "avgEfficiency": {"$avg": "$executionMetrics.efficiency"},
                        "avgCompletionRate": {"$avg": "$executionMetrics.completionRate"},
                        "avgTimeAccuracy": {"$avg": "$executionMetrics.timeAccuracy"},
                        "planStatuses": {"$addToSet": "$planStatus"},
                        "optimizationRecommendations": {"$push": "$optimizationRecommendations"}
                    }
                },
                {
                    "$project": {
                        "planType": "$_id",
                        "planCount": 1,
                        "avgPriority": {"$round": ["$avgPriority", 3]},
                        "avgEfficiency": {"$round": ["$avgEfficiency", 3]},
                        "avgCompletionRate": {"$round": ["$avgCompletionRate", 3]},
                        "avgTimeAccuracy": {"$round": ["$avgTimeAccuracy", 3]},
                        "planStatuses": 1,
                        "optimizationRecommendations": 1,
                        "_id": 0
                    }
                }
            ]
            
            results = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Calculate overall analytics
            total_plans = sum(r["planCount"] for r in results)
            
            if total_plans > 0:
                overall_efficiency = sum(r["avgEfficiency"] * r["planCount"] for r in results) / total_plans
                overall_completion = sum(r["avgCompletionRate"] * r["planCount"] for r in results) / total_plans
                overall_time_accuracy = sum(r["avgTimeAccuracy"] * r["planCount"] for r in results) / total_plans
            else:
                overall_efficiency = overall_completion = overall_time_accuracy = 0.0
            
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
                "planTypes": results,
                "totalPlans": total_plans,
                "overallEfficiency": round(overall_efficiency, 3),
                "overallCompletionRate": round(overall_completion, 3),
                "overallTimeAccuracy": round(overall_time_accuracy, 3),
                "planTypesCount": len(results),
                "topOptimizations": [opt[0] for opt in top_optimizations]
            }
                
        except Exception as error:
            logger.error(f"Error getting temporal analytics: {error}")
            raise error
    
    async def get_temporal_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get temporal planning statistics."""
        try:
            match_filter = {}
            if agent_id:
                match_filter["agentId"] = agent_id
            
            # Get basic stats
            total_plans = await self.collection.count_documents(match_filter)
            
            # Get recent stats (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_filter = {**match_filter, "timestamp": {"$gte": recent_time}}
            recent_plans = await self.collection.count_documents(recent_filter)
            
            # Get active plans
            active_filter = {**match_filter, "planStatus": {"$in": ["active", "in_progress"]}}
            active_plans = await self.collection.count_documents(active_filter)
            
            # Get plan diversity and performance stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "uniquePlanTypes": {"$addToSet": "$planType"},
                        "uniquePlanStatuses": {"$addToSet": "$planStatus"},
                        "avgPriority": {"$avg": "$priority"},
                        "avgEfficiency": {"$avg": "$executionMetrics.efficiency"},
                        "avgCompletionRate": {"$avg": "$executionMetrics.completionRate"},
                        "maxPriority": {"$max": "$priority"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                return {
                    "totalPlans": total_plans,
                    "recentPlans": recent_plans,
                    "activePlans": active_plans,
                    "uniquePlanTypes": len(stats.get("uniquePlanTypes", [])),
                    "uniquePlanStatuses": len(stats.get("uniquePlanStatuses", [])),
                    "avgPriority": round(stats.get("avgPriority", 0.0), 3),
                    "avgEfficiency": round(stats.get("avgEfficiency", 0.0), 3),
                    "avgCompletionRate": round(stats.get("avgCompletionRate", 0.0), 3),
                    "maxPriority": round(stats.get("maxPriority", 0.0), 3),
                    "planTypes": stats.get("uniquePlanTypes", []),
                    "planStatuses": stats.get("uniquePlanStatuses", [])
                }
            else:
                return {
                    "totalPlans": total_plans,
                    "recentPlans": recent_plans,
                    "activePlans": active_plans,
                    "uniquePlanTypes": 0,
                    "uniquePlanStatuses": 0,
                    "avgPriority": 0.0,
                    "avgEfficiency": 0.0,
                    "avgCompletionRate": 0.0,
                    "maxPriority": 0.0,
                    "planTypes": [],
                    "planStatuses": []
                }
                
        except Exception as error:
            logger.error(f"Error getting temporal stats: {error}")
            raise error
    
    async def find_active_plans(
        self,
        agent_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find active temporal plans for an agent."""
        try:
            match_filter = {
                "agentId": agent_id,
                "planStatus": {"$in": ["active", "in_progress"]}
            }
            
            cursor = self.collection.find(match_filter).sort("priority", DESCENDING).limit(limit)
            plans = await cursor.to_list(length=limit)
            
            return plans
            
        except Exception as error:
            logger.error(f"Error finding active plans: {error}")
            raise error
