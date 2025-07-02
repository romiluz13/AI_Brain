"""
WorkflowOrchestrationCollection - MongoDB collection for workflow orchestration tracking

Exact Python equivalent of JavaScript WorkflowOrchestrationCollection.ts with:
- Workflow execution records and routing decisions
- Parallel task execution tracking
- Performance analytics and optimization
- Workflow evaluation and learning
- Advanced indexing for optimal query performance

Features:
- Intelligent request routing tracking
- Parallel task execution coordination
- Workflow evaluation and optimization
- Real-time workflow monitoring
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class WorkflowOrchestrationCollection:
    """
    Workflow Orchestration Collection for MongoDB operations
    
    Manages workflow execution records, routing decisions, and performance
    analytics using MongoDB's advanced indexing and aggregation capabilities.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.workflow_states
        
    async def initialize_indexes(self):
        """Create indexes for optimal query performance."""
        try:
            # Primary queries
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_timestamp_idx", background=True)
            
            await self.collection.create_index([
                ("executionId", ASCENDING)
            ], name="execution_id_idx", unique=True, background=True)
            
            await self.collection.create_index([
                ("workflowType", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="workflow_type_idx", background=True)
            
            # Status and performance queries
            await self.collection.create_index([
                ("status", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="status_timestamp_idx", background=True)
            
            await self.collection.create_index([
                ("success", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="success_timestamp_idx", background=True)
            
            await self.collection.create_index([
                ("duration", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="duration_timestamp_idx", background=True)
            
            # Routing specific indexes
            await self.collection.create_index([
                ("routing.request.taskType", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="routing_task_type_idx", background=True)
            
            await self.collection.create_index([
                ("routing.path.confidence", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="routing_confidence_idx", background=True)
            
            await self.collection.create_index([
                ("routing.riskAssessment.level", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="routing_risk_idx", background=True)
            
            # Parallel execution indexes
            await self.collection.create_index([
                ("parallel.performance.parallelEfficiency", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="parallel_efficiency_idx", background=True)
            
            # Evaluation indexes
            await self.collection.create_index([
                ("evaluation.metrics.efficiency", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="evaluation_efficiency_idx", background=True)
            
            # Compound indexes for complex queries
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("workflowType", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_workflow_timestamp_idx", background=True)
            
            await self.collection.create_index([
                ("success", ASCENDING),
                ("duration", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="success_duration_timestamp_idx", background=True)
            
            # TTL index for automatic cleanup (180 days)
            await self.collection.create_index([
                ("createdAt", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=15552000, background=True)
            
            logger.info("✅ WorkflowOrchestrationCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating WorkflowOrchestrationCollection indexes: {error}")
            raise error
    
    async def record_execution(self, execution: Dict[str, Any]) -> ObjectId:
        """Record a workflow execution."""
        try:
            now = datetime.utcnow()
            execution_record = {
                **execution,
                "createdAt": now,
                "updatedAt": now
            }
            
            result = await self.collection.insert_one(execution_record)
            logger.debug(f"Workflow execution recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording workflow execution: {error}")
            raise error
    
    async def update_execution_status(
        self,
        execution_id: ObjectId,
        status: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update workflow execution status."""
        try:
            update_data = {
                "status": status,
                "updatedAt": datetime.utcnow()
            }
            
            if additional_data:
                update_data.update(additional_data)
            
            if status in ["completed", "failed", "cancelled"]:
                update_data["endTime"] = datetime.utcnow()
            
            result = await self.collection.update_one(
                {"executionId": execution_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
        except Exception as error:
            logger.error(f"Error updating workflow execution status: {error}")
            raise error
    
    async def get_workflow_performance_analytics(
        self,
        workflow_type: str,
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """Get performance analytics for a specific workflow type."""
        try:
            start_date = datetime.utcnow() - timedelta(days=timeframe_days)
            
            pipeline = [
                {
                    "$match": {
                        "workflowType": workflow_type,
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "totalExecutions": {"$sum": 1},
                        "successfulExecutions": {
                            "$sum": {"$cond": [{"$eq": ["$success", True]}, 1, 0]}
                        },
                        "avgDuration": {"$avg": "$duration"},
                        "maxDuration": {"$max": "$duration"},
                        "minDuration": {"$min": "$duration"},
                        "avgEfficiency": {"$avg": "$evaluation.metrics.efficiency"},
                        "avgAccuracy": {"$avg": "$evaluation.metrics.accuracy"},
                        "avgReliability": {"$avg": "$evaluation.metrics.reliability"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "totalExecutions": 1,
                        "successRate": {
                            "$divide": ["$successfulExecutions", "$totalExecutions"]
                        },
                        "avgDuration": {"$round": ["$avgDuration", 2]},
                        "maxDuration": 1,
                        "minDuration": 1,
                        "avgEfficiency": {"$round": ["$avgEfficiency", 3]},
                        "avgAccuracy": {"$round": ["$avgAccuracy", 3]},
                        "avgReliability": {"$round": ["$avgReliability", 3]}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if not result:
                return {
                    "totalExecutions": 0,
                    "successRate": 0,
                    "avgDuration": 0,
                    "maxDuration": 0,
                    "minDuration": 0,
                    "avgEfficiency": 0,
                    "avgAccuracy": 0,
                    "avgReliability": 0
                }
            
            return result[0]
        except Exception as error:
            logger.error(f"Error getting workflow performance analytics: {error}")
            raise error

    async def get_top_performing_patterns(
        self,
        workflow_type: str,
        timeframe_days: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top performing workflow patterns."""
        try:
            start_date = datetime.utcnow() - timedelta(days=timeframe_days)

            pipeline = [
                {
                    "$match": {
                        "workflowType": workflow_type,
                        "timestamp": {"$gte": start_date},
                        "success": True
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "route": "$routing.path.route",
                            "coordination": "$parallel.request.coordination.strategy"
                        },
                        "count": {"$sum": 1},
                        "avgDuration": {"$avg": "$duration"},
                        "avgEfficiency": {"$avg": "$evaluation.metrics.efficiency"},
                        "avgAccuracy": {"$avg": "$evaluation.metrics.accuracy"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "pattern": "$_id",
                        "count": 1,
                        "avgDuration": {"$round": ["$avgDuration", 2]},
                        "avgEfficiency": {"$round": ["$avgEfficiency", 3]},
                        "avgAccuracy": {"$round": ["$avgAccuracy", 3]},
                        "score": {
                            "$multiply": [
                                {"$divide": ["$avgEfficiency", 100]},
                                {"$divide": ["$avgAccuracy", 100]}
                            ]
                        }
                    }
                },
                {
                    "$sort": {"score": -1}
                },
                {
                    "$limit": limit
                }
            ]

            return await self.collection.aggregate(pipeline).to_list(length=limit)
        except Exception as error:
            logger.error(f"Error getting top performing patterns: {error}")
            raise error

    async def get_common_failures(
        self,
        workflow_type: str,
        timeframe_days: int = 30,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get common failure patterns."""
        try:
            start_date = datetime.utcnow() - timedelta(days=timeframe_days)

            pipeline = [
                {
                    "$match": {
                        "workflowType": workflow_type,
                        "timestamp": {"$gte": start_date},
                        "success": False
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "errorType": "$error.type",
                            "errorCode": "$error.code"
                        },
                        "count": {"$sum": 1},
                        "examples": {"$push": "$error.message"},
                        "avgDuration": {"$avg": "$duration"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "errorType": "$_id.errorType",
                        "errorCode": "$_id.errorCode",
                        "count": 1,
                        "examples": {"$slice": ["$examples", 3]},
                        "avgDuration": {"$round": ["$avgDuration", 2]}
                    }
                },
                {
                    "$sort": {"count": -1}
                },
                {
                    "$limit": limit
                }
            ]

            return await self.collection.aggregate(pipeline).to_list(length=limit)
        except Exception as error:
            logger.error(f"Error getting common failures: {error}")
            raise error

    async def get_active_workflows(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get currently active workflows."""
        try:
            match_filter = {
                "status": {"$in": ["pending", "in_progress"]}
            }

            if agent_id:
                match_filter["agentId"] = agent_id

            cursor = self.collection.find(match_filter).sort([
                ("timestamp", ASCENDING)
            ])

            return await cursor.to_list(length=None)
        except Exception as error:
            logger.error(f"Error getting active workflows: {error}")
            raise error

    async def get_workflow_recommendations(
        self,
        workflow_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get workflow optimization recommendations."""
        try:
            # Get performance analytics
            analytics = await self.get_workflow_performance_analytics(workflow_type, 30)

            # Get top performing patterns
            top_patterns = await self.get_top_performing_patterns(workflow_type, 30, 5)

            # Get common failures
            common_failures = await self.get_common_failures(workflow_type, 30, 5)

            # Generate recommendations based on context
            recommendations = []

            if analytics["successRate"] < 0.8:
                recommendations.append({
                    "type": "reliability",
                    "priority": "high",
                    "message": f"Success rate is {analytics['successRate']:.1%}. Consider reviewing error patterns.",
                    "actions": ["Review common failures", "Implement better error handling"]
                })

            if analytics["avgDuration"] > 30000:  # 30 seconds
                recommendations.append({
                    "type": "performance",
                    "priority": "medium",
                    "message": f"Average duration is {analytics['avgDuration']/1000:.1f}s. Consider optimization.",
                    "actions": ["Optimize slow operations", "Implement parallel processing"]
                })

            if top_patterns:
                best_pattern = top_patterns[0]
                recommendations.append({
                    "type": "optimization",
                    "priority": "low",
                    "message": f"Best performing pattern has {best_pattern['avgEfficiency']:.1%} efficiency.",
                    "actions": ["Adopt best practices from top patterns"]
                })

            return {
                "analytics": analytics,
                "topPatterns": top_patterns,
                "commonFailures": common_failures,
                "recommendations": recommendations
            }
        except Exception as error:
            logger.error(f"Error getting workflow recommendations: {error}")
            raise error
