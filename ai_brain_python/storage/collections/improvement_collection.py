"""
Improvement Collection - MongoDB collection for self-improvement and optimization tracking

Handles storage and retrieval of improvement records, optimization data, and performance analytics.
Provides insights for continuous learning and system enhancement.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class ImprovementCollection:
    """
    Improvement Collection for MongoDB operations
    
    Manages improvement records, optimization data, and performance analytics.
    Provides insights for continuous learning and system enhancement.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.improvements
    
    async def create_indexes(self) -> None:
        """Create indexes for improvement collection."""
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
            
            # Improvement indexes
            await self.collection.create_index([
                ("improvementId", ASCENDING)
            ], name="improvement_id_index", background=True, unique=True)
            
            await self.collection.create_index([
                ("optimizationScore", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="optimization_score_index", background=True)
            
            await self.collection.create_index([
                ("improvementAreas", ASCENDING)
            ], name="improvement_areas_index", background=True)
            
            # Performance indexes
            await self.collection.create_index([
                ("performanceGains.metric", ASCENDING),
                ("performanceGains.gain", DESCENDING)
            ], name="performance_gains_index", background=True)
            
            await self.collection.create_index([
                ("performanceMetrics.accuracy", DESCENDING)
            ], name="performance_accuracy_index", background=True)
            
            await self.collection.create_index([
                ("performanceMetrics.efficiency", DESCENDING)
            ], name="performance_efficiency_index", background=True)
            
            # Optimization indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("optimizationScore", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_optimization_performance_index", background=True)
            
            await self.collection.create_index([
                ("optimizationGoals", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="optimization_goals_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("improvementAreas", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="improvement_analytics_index", background=True)
            
            # TTL index for automatic cleanup (365 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=31536000, background=True)
            
            logger.info("✅ ImprovementCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating ImprovementCollection indexes: {error}")
            raise error
    
    async def record_improvement(self, improvement_record: Dict[str, Any]) -> ObjectId:
        """Record an improvement analysis."""
        try:
            result = await self.collection.insert_one(improvement_record)
            logger.debug(f"Improvement record created: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording improvement: {error}")
            raise error
    
    async def get_improvement_analytics(
        self,
        agent_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get improvement analytics for an agent."""
        try:
            # Default time range: last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            if options:
                if options.get("start_time"):
                    start_time = options["start_time"]
                if options.get("end_time"):
                    end_time = options["end_time"]
            
            # Aggregation pipeline for improvement analytics
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
                        "totalImprovements": {"$sum": 1},
                        "avgOptimizationScore": {"$avg": "$optimizationScore"},
                        "maxOptimizationScore": {"$max": "$optimizationScore"},
                        "minOptimizationScore": {"$min": "$optimizationScore"},
                        "improvementAreas": {"$addToSet": "$improvementAreas"},
                        "optimizationGoals": {"$addToSet": "$optimizationGoals"},
                        "performanceGains": {"$push": "$performanceGains"},
                        "learningInsights": {"$push": "$learningInsights"},
                        "recommendations": {"$push": "$recommendations"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                analytics = result[0]
                analytics.pop("_id", None)
                
                # Flatten improvement areas
                all_areas = []
                for area_list in analytics.get("improvementAreas", []):
                    if isinstance(area_list, list):
                        all_areas.extend(area_list)
                
                # Count area frequency
                area_counts = {}
                for area in all_areas:
                    area_counts[area] = area_counts.get(area, 0) + 1
                
                top_areas = sorted(
                    area_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                # Analyze performance gains
                all_gains = []
                for gain_dict in analytics.get("performanceGains", []):
                    if isinstance(gain_dict, dict):
                        for metric, gain in gain_dict.items():
                            if isinstance(gain, (int, float)):
                                all_gains.append({"metric": metric, "gain": gain})
                
                # Calculate average gains by metric
                metric_gains = {}
                for gain in all_gains:
                    metric = gain["metric"]
                    if metric not in metric_gains:
                        metric_gains[metric] = []
                    metric_gains[metric].append(gain["gain"])
                
                avg_gains = {}
                for metric, gains in metric_gains.items():
                    avg_gains[metric] = round(sum(gains) / len(gains), 3)
                
                # Get top recommendations
                all_recommendations = []
                for rec_list in analytics.get("recommendations", []):
                    if isinstance(rec_list, list):
                        all_recommendations.extend(rec_list)
                
                recommendation_counts = {}
                for rec in all_recommendations:
                    if isinstance(rec, dict) and "recommendation" in rec:
                        rec_text = rec["recommendation"]
                        recommendation_counts[rec_text] = recommendation_counts.get(rec_text, 0) + 1
                
                top_recommendations = sorted(
                    recommendation_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                analytics.update({
                    "topImprovementAreas": [area[0] for area in top_areas],
                    "averagePerformanceGains": avg_gains,
                    "topRecommendations": [rec[0] for rec in top_recommendations],
                    "improvementAreasCount": len(set(all_areas)),
                    "totalPerformanceGains": len(all_gains)
                })
                
                # Clean up raw data
                analytics.pop("improvementAreas", None)
                analytics.pop("performanceGains", None)
                analytics.pop("recommendations", None)
                
                return analytics
            else:
                return {
                    "totalImprovements": 0,
                    "avgOptimizationScore": 0.0,
                    "maxOptimizationScore": 0.0,
                    "minOptimizationScore": 0.0,
                    "topImprovementAreas": [],
                    "averagePerformanceGains": {},
                    "topRecommendations": [],
                    "improvementAreasCount": 0,
                    "totalPerformanceGains": 0
                }
                
        except Exception as error:
            logger.error(f"Error getting improvement analytics: {error}")
            raise error
    
    async def get_improvement_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get improvement statistics."""
        try:
            match_filter = {}
            if agent_id:
                match_filter["agentId"] = agent_id
            
            # Get basic stats
            total_improvements = await self.collection.count_documents(match_filter)
            
            # Get recent stats (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_filter = {**match_filter, "timestamp": {"$gte": recent_time}}
            recent_improvements = await self.collection.count_documents(recent_filter)
            
            # Get high-optimization improvements
            high_opt_filter = {**match_filter, "optimizationScore": {"$gte": 0.8}}
            high_optimization_improvements = await self.collection.count_documents(high_opt_filter)
            
            # Get improvement performance stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "avgOptimizationScore": {"$avg": "$optimizationScore"},
                        "maxOptimizationScore": {"$max": "$optimizationScore"},
                        "uniqueImprovementAreas": {"$addToSet": "$improvementAreas"},
                        "uniqueOptimizationGoals": {"$addToSet": "$optimizationGoals"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                
                # Count unique areas and goals
                all_areas = []
                for area_list in stats.get("uniqueImprovementAreas", []):
                    if isinstance(area_list, list):
                        all_areas.extend(area_list)
                
                all_goals = []
                for goal_list in stats.get("uniqueOptimizationGoals", []):
                    if isinstance(goal_list, list):
                        all_goals.extend(goal_list)
                
                return {
                    "totalImprovements": total_improvements,
                    "recentImprovements": recent_improvements,
                    "highOptimizationImprovements": high_optimization_improvements,
                    "avgOptimizationScore": round(stats.get("avgOptimizationScore", 0.0), 3),
                    "maxOptimizationScore": round(stats.get("maxOptimizationScore", 0.0), 3),
                    "uniqueImprovementAreas": len(set(all_areas)),
                    "uniqueOptimizationGoals": len(set(all_goals)),
                    "improvementAreas": list(set(all_areas)),
                    "optimizationGoals": list(set(all_goals))
                }
            else:
                return {
                    "totalImprovements": total_improvements,
                    "recentImprovements": recent_improvements,
                    "highOptimizationImprovements": high_optimization_improvements,
                    "avgOptimizationScore": 0.0,
                    "maxOptimizationScore": 0.0,
                    "uniqueImprovementAreas": 0,
                    "uniqueOptimizationGoals": 0,
                    "improvementAreas": [],
                    "optimizationGoals": []
                }
                
        except Exception as error:
            logger.error(f"Error getting improvement stats: {error}")
            raise error
