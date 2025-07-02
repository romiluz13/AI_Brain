"""
Skill Capability Collection - MongoDB collection for skill assessment and capability management

Handles storage and retrieval of skill assessments, capability tracking, and performance analytics.
Provides comprehensive skill development insights and capability optimization.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class SkillCapabilityCollection:
    """
    Skill Capability Collection for MongoDB operations
    
    Manages skill assessments, capability tracking, and performance analytics.
    Provides insights for skill development and capability optimization.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.skill_capabilities
    
    async def create_indexes(self) -> None:
        """Create indexes for skill capability collection."""
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
            
            # Skill assessment indexes
            await self.collection.create_index([
                ("skillDomain", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="skill_domain_index", background=True)
            
            await self.collection.create_index([
                ("skillLevel", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="skill_level_index", background=True)
            
            await self.collection.create_index([
                ("proficiencyScore", DESCENDING)
            ], name="proficiency_score_index", background=True)
            
            # Capability indexes
            await self.collection.create_index([
                ("capabilities.capabilityType", ASCENDING),
                ("capabilities.currentLevel", DESCENDING)
            ], name="capability_type_level_index", background=True)
            
            await self.collection.create_index([
                ("capabilities.improvementPotential", DESCENDING)
            ], name="improvement_potential_index", background=True)
            
            # Performance indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("skillDomain", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_skill_performance_index", background=True)
            
            await self.collection.create_index([
                ("performanceMetrics.accuracy", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="performance_accuracy_index", background=True)
            
            # Analytics indexes
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("skillDomain", ASCENDING),
                ("proficiencyScore", DESCENDING)
            ], name="skill_analytics_index", background=True)
            
            # TTL index for automatic cleanup (180 days)
            await self.collection.create_index([
                ("timestamp", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=15552000, background=True)
            
            logger.info("✅ SkillCapabilityCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating SkillCapabilityCollection indexes: {error}")
            raise error
    
    async def record_skill_assessment(self, skill_record: Dict[str, Any]) -> ObjectId:
        """Record a skill assessment."""
        try:
            result = await self.collection.insert_one(skill_record)
            logger.debug(f"Skill assessment recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording skill assessment: {error}")
            raise error
    
    async def get_skill_analytics(
        self,
        agent_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get skill analytics for an agent."""
        try:
            # Default time range: last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            if options:
                if options.get("start_time"):
                    start_time = options["start_time"]
                if options.get("end_time"):
                    end_time = options["end_time"]
            
            # Aggregation pipeline for skill analytics
            pipeline = [
                {
                    "$match": {
                        "agentId": agent_id,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$skillDomain",
                        "assessmentCount": {"$sum": 1},
                        "avgProficiencyScore": {"$avg": "$proficiencyScore"},
                        "maxProficiencyScore": {"$max": "$proficiencyScore"},
                        "minProficiencyScore": {"$min": "$proficiencyScore"},
                        "avgSkillLevel": {"$avg": "$skillLevel"},
                        "latestAssessment": {"$last": "$$ROOT"},
                        "improvementRecommendations": {"$push": "$improvementRecommendations"}
                    }
                },
                {
                    "$project": {
                        "skillDomain": "$_id",
                        "assessmentCount": 1,
                        "avgProficiencyScore": {"$round": ["$avgProficiencyScore", 3]},
                        "maxProficiencyScore": 1,
                        "minProficiencyScore": 1,
                        "avgSkillLevel": {"$round": ["$avgSkillLevel", 2]},
                        "latestAssessment": 1,
                        "improvementRecommendations": 1,
                        "_id": 0
                    }
                }
            ]
            
            results = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Calculate overall analytics
            total_assessments = sum(r["assessmentCount"] for r in results)
            overall_proficiency = sum(r["avgProficiencyScore"] * r["assessmentCount"] for r in results) / total_assessments if total_assessments > 0 else 0
            
            # Get top improvement areas
            all_recommendations = []
            for result in results:
                for rec_list in result.get("improvementRecommendations", []):
                    if isinstance(rec_list, list):
                        all_recommendations.extend(rec_list)
            
            recommendation_counts = {}
            for rec in all_recommendations:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
            
            top_improvements = sorted(
                recommendation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "skillDomains": results,
                "totalAssessments": total_assessments,
                "overallProficiency": round(overall_proficiency, 3),
                "skillDomainsCount": len(results),
                "topImprovementAreas": [imp[0] for imp in top_improvements]
            }
                
        except Exception as error:
            logger.error(f"Error getting skill analytics: {error}")
            raise error
    
    async def get_skill_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get skill capability statistics."""
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
            
            # Get skill diversity and performance stats
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "uniqueSkillDomains": {"$addToSet": "$skillDomain"},
                        "avgProficiencyScore": {"$avg": "$proficiencyScore"},
                        "avgSkillLevel": {"$avg": "$skillLevel"},
                        "maxProficiencyScore": {"$max": "$proficiencyScore"},
                        "capabilities": {"$push": "$capabilities"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                
                # Count total capabilities
                all_capabilities = []
                for cap_list in stats.get("capabilities", []):
                    if isinstance(cap_list, list):
                        all_capabilities.extend(cap_list)
                
                return {
                    "totalAssessments": total_assessments,
                    "recentAssessments": recent_assessments,
                    "uniqueSkillDomains": len(stats.get("uniqueSkillDomains", [])),
                    "avgProficiencyScore": round(stats.get("avgProficiencyScore", 0.0), 3),
                    "avgSkillLevel": round(stats.get("avgSkillLevel", 0.0), 2),
                    "maxProficiencyScore": round(stats.get("maxProficiencyScore", 0.0), 3),
                    "totalCapabilities": len(all_capabilities),
                    "skillDomains": stats.get("uniqueSkillDomains", [])
                }
            else:
                return {
                    "totalAssessments": total_assessments,
                    "recentAssessments": recent_assessments,
                    "uniqueSkillDomains": 0,
                    "avgProficiencyScore": 0.0,
                    "avgSkillLevel": 0.0,
                    "maxProficiencyScore": 0.0,
                    "totalCapabilities": 0,
                    "skillDomains": []
                }
                
        except Exception as error:
            logger.error(f"Error getting skill stats: {error}")
            raise error
    
    async def find_skill_assessments(
        self,
        agent_id: str,
        skill_domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find skill assessments for an agent."""
        try:
            match_filter = {"agentId": agent_id}
            if skill_domain:
                match_filter["skillDomain"] = skill_domain
            
            cursor = self.collection.find(match_filter).sort("timestamp", DESCENDING).limit(limit)
            assessments = await cursor.to_list(length=limit)
            
            return assessments
            
        except Exception as error:
            logger.error(f"Error finding skill assessments: {error}")
            raise error
