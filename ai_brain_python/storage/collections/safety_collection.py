"""
Safety Collection - MongoDB collection for safety monitoring and risk assessment

Handles storage and retrieval of safety assessments, risk evaluations, and safety analytics.
Provides comprehensive safety monitoring and compliance tracking.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class SafetyCollection:
    """
    Safety Collection for MongoDB operations
    
    Manages safety assessments, risk evaluations, and safety analytics.
    Provides comprehensive safety monitoring and compliance tracking.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.safety_assessments
    
    async def create_indexes(self) -> None:
        """Create indexes for safety collection."""
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
            
            # Safety assessment indexes
            await self.collection.create_index([
                ("assessmentId", ASCENDING)
            ], name="assessment_id_index", background=True, unique=True)
            
            await self.collection.create_index([
                ("riskScore", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="risk_score_index", background=True)
            
            await self.collection.create_index([
                ("actionRequired", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="action_required_index", background=True)
            
            # Violation indexes
            await self.collection.create_index([
                ("violations.severity", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="violation_severity_index", background=True)
            
            await self.collection.create_index([
                ("violations.policyId", ASCENDING)
            ], name="violation_policy_index", background=True)
            
            # Risk level indexes
            await self.collection.create_index([
                ("riskLevel", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="risk_level_index", background=True)
            
            # Performance indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("riskScore", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_risk_performance_index", background=True)
            
            await self.collection.create_index([
                ("mitigationStrategies.action", ASCENDING)
            ], name="mitigation_action_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("riskLevel", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="safety_analytics_index", background=True)
            
            # TTL index for automatic cleanup (180 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=15552000, background=True)
            
            logger.info("✅ SafetyCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating SafetyCollection indexes: {error}")
            raise error
    
    async def record_safety_assessment(self, safety_record: Dict[str, Any]) -> ObjectId:
        """Record a safety assessment."""
        try:
            result = await self.collection.insert_one(safety_record)
            logger.debug(f"Safety assessment recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording safety assessment: {error}")
            raise error
    
    async def get_safety_analytics(
        self,
        agent_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get safety analytics for an agent."""
        try:
            # Default time range: last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            if options:
                if options.get("start_time"):
                    start_time = options["start_time"]
                if options.get("end_time"):
                    end_time = options["end_time"]
            
            # Aggregation pipeline for safety analytics
            pipeline = [
                {
                    "$match": {
                        "agentId": agent_id,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$riskLevel",
                        "assessmentCount": {"$sum": 1},
                        "avgRiskScore": {"$avg": "$riskScore"},
                        "maxRiskScore": {"$max": "$riskScore"},
                        "minRiskScore": {"$min": "$riskScore"},
                        "violationCount": {"$sum": {"$size": {"$ifNull": ["$violations", []]}}},
                        "actionRequiredCount": {"$sum": {"$cond": ["$actionRequired", 1, 0]}},
                        "mitigationStrategies": {"$push": "$mitigationStrategies"},
                        "recommendations": {"$push": "$recommendations"}
                    }
                },
                {
                    "$project": {
                        "riskLevel": "$_id",
                        "assessmentCount": 1,
                        "avgRiskScore": {"$round": ["$avgRiskScore", 3]},
                        "maxRiskScore": {"$round": ["$maxRiskScore", 3]},
                        "minRiskScore": {"$round": ["$minRiskScore", 3]},
                        "violationCount": 1,
                        "actionRequiredCount": 1,
                        "mitigationStrategies": 1,
                        "recommendations": 1,
                        "_id": 0
                    }
                }
            ]
            
            results = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Calculate overall analytics
            total_assessments = sum(r["assessmentCount"] for r in results)
            total_violations = sum(r["violationCount"] for r in results)
            total_actions_required = sum(r["actionRequiredCount"] for r in results)
            
            if total_assessments > 0:
                overall_risk_score = sum(r["avgRiskScore"] * r["assessmentCount"] for r in results) / total_assessments
                action_required_rate = total_actions_required / total_assessments
            else:
                overall_risk_score = action_required_rate = 0.0
            
            # Get top recommendations
            all_recommendations = []
            for result in results:
                for rec_list in result.get("recommendations", []):
                    if isinstance(rec_list, list):
                        all_recommendations.extend(rec_list)
            
            recommendation_counts = {}
            for rec in all_recommendations:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
            
            top_recommendations = sorted(
                recommendation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Analyze mitigation strategies
            all_strategies = []
            for result in results:
                for strategy_list in result.get("mitigationStrategies", []):
                    if isinstance(strategy_list, list):
                        all_strategies.extend(strategy_list)
            
            strategy_counts = {}
            for strategy in all_strategies:
                if isinstance(strategy, dict) and "action" in strategy:
                    action = strategy["action"]
                    strategy_counts[action] = strategy_counts.get(action, 0) + 1
            
            top_strategies = sorted(
                strategy_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "riskLevels": results,
                "totalAssessments": total_assessments,
                "totalViolations": total_violations,
                "totalActionsRequired": total_actions_required,
                "overallRiskScore": round(overall_risk_score, 3),
                "actionRequiredRate": round(action_required_rate, 3),
                "riskLevelsCount": len(results),
                "topRecommendations": [rec[0] for rec in top_recommendations],
                "topMitigationStrategies": [strat[0] for strat in top_strategies]
            }
                
        except Exception as error:
            logger.error(f"Error getting safety analytics: {error}")
            raise error
    
    async def get_safety_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get safety statistics."""
        try:
            match_filter = {}
            if agent_id:
                match_filter["agentId"] = agent_id
            
            # Get basic stats
            total_assessments = await self.collection.count_documents(match_filter)
            
            # Get recent stats (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_filter = {**match_filter, "timestamp": {"$gte": recent_time}}
            recent_assessments = await self.collection.count_documents(recent_filter)
            
            # Get high-risk assessments
            high_risk_filter = {**match_filter, "riskScore": {"$gte": 0.8}}
            high_risk_assessments = await self.collection.count_documents(high_risk_filter)
            
            # Get action required assessments
            action_required_filter = {**match_filter, "actionRequired": True}
            action_required_assessments = await self.collection.count_documents(action_required_filter)
            
            # Get safety performance stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "uniqueRiskLevels": {"$addToSet": "$riskLevel"},
                        "avgRiskScore": {"$avg": "$riskScore"},
                        "maxRiskScore": {"$max": "$riskScore"},
                        "totalViolations": {"$sum": {"$size": {"$ifNull": ["$violations", []]}}},
                        "actionRequiredCount": {"$sum": {"$cond": ["$actionRequired", 1, 0]}}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                action_required_rate = stats.get("actionRequiredCount", 0) / total_assessments if total_assessments > 0 else 0
                
                return {
                    "totalAssessments": total_assessments,
                    "recentAssessments": recent_assessments,
                    "highRiskAssessments": high_risk_assessments,
                    "actionRequiredAssessments": action_required_assessments,
                    "uniqueRiskLevels": len(stats.get("uniqueRiskLevels", [])),
                    "avgRiskScore": round(stats.get("avgRiskScore", 0.0), 3),
                    "maxRiskScore": round(stats.get("maxRiskScore", 0.0), 3),
                    "totalViolations": stats.get("totalViolations", 0),
                    "actionRequiredRate": round(action_required_rate, 3),
                    "riskLevels": stats.get("uniqueRiskLevels", [])
                }
            else:
                return {
                    "totalAssessments": total_assessments,
                    "recentAssessments": recent_assessments,
                    "highRiskAssessments": high_risk_assessments,
                    "actionRequiredAssessments": action_required_assessments,
                    "uniqueRiskLevels": 0,
                    "avgRiskScore": 0.0,
                    "maxRiskScore": 0.0,
                    "totalViolations": 0,
                    "actionRequiredRate": 0.0,
                    "riskLevels": []
                }
                
        except Exception as error:
            logger.error(f"Error getting safety stats: {error}")
            raise error
