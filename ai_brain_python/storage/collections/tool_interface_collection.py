"""
ToolInterfaceCollection - MongoDB collection for tool execution tracking

Exact Python equivalent of JavaScript ToolInterfaceCollection.ts with:
- Tool execution records and performance metrics
- Validation results and error tracking
- Human-in-loop approval workflows
- Tool capability discovery and documentation
- Advanced indexing for optimal query performance

Features:
- Tool execution tracking with retry and recovery
- Performance analytics and optimization
- Human approval workflow management
- Tool capability discovery and documentation
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from ai_brain_python.utils.logger import logger


class ToolInterfaceCollection:
    """
    Tool Interface Collection for MongoDB operations
    
    Manages tool execution records, performance metrics, and validation
    results using MongoDB's advanced indexing and aggregation capabilities.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.tool_validations
        
    async def initialize_indexes(self):
        """Create indexes for optimal query performance."""
        try:
            # Primary queries
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_timestamp_idx", background=True)
            
            await self.collection.create_index([
                ("toolName", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="tool_timestamp_idx", background=True)
            
            await self.collection.create_index([
                ("executionId", ASCENDING)
            ], name="execution_id_idx", unique=True, background=True)
            
            # Performance analytics
            await self.collection.create_index([
                ("toolName", ASCENDING),
                ("success", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="tool_success_analytics_idx", background=True)
            
            await self.collection.create_index([
                ("context.priority", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="priority_timestamp_idx", background=True)
            
            # Human interaction tracking
            await self.collection.create_index([
                ("humanInteraction.approvalStatus", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="approval_status_idx", background=True)
            
            # Compound indexes for complex queries
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("toolName", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_tool_timestamp_idx", background=True)
            
            await self.collection.create_index([
                ("success", ASCENDING),
                ("performance.executionTime", ASCENDING)
            ], name="success_performance_idx", background=True)
            
            # TTL index for automatic cleanup (180 days)
            await self.collection.create_index([
                ("createdAt", ASCENDING)
            ], name="ttl_index", expireAfterSeconds=15552000, background=True)
            
            logger.info("✅ ToolInterfaceCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating ToolInterfaceCollection indexes: {error}")
            raise error
    
    async def record_execution(self, execution: Dict[str, Any]) -> ObjectId:
        """Record a tool execution."""
        try:
            now = datetime.utcnow()
            execution_record = {
                **execution,
                "createdAt": now,
                "updatedAt": now
            }
            
            result = await self.collection.insert_one(execution_record)
            logger.debug(f"Tool execution recorded: {result.inserted_id}")
            return result.inserted_id
        except Exception as error:
            logger.error(f"Error recording tool execution: {error}")
            raise error
    
    async def update_execution_result(
        self,
        execution_id: ObjectId,
        result: Dict[str, Any]
    ) -> bool:
        """Update tool execution with result."""
        try:
            update_data = {
                **result,
                "updatedAt": datetime.utcnow()
            }
            
            result = await self.collection.update_one(
                {"executionId": execution_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
        except Exception as error:
            logger.error(f"Error updating tool execution result: {error}")
            raise error
    
    async def get_tool_performance_analytics(
        self,
        tool_name: str,
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """Get performance analytics for a specific tool."""
        try:
            start_date = datetime.utcnow() - timedelta(days=timeframe_days)
            
            pipeline = [
                {
                    "$match": {
                        "toolName": tool_name,
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
                        "avgExecutionTime": {"$avg": "$performance.executionTime"},
                        "maxExecutionTime": {"$max": "$performance.executionTime"},
                        "minExecutionTime": {"$min": "$performance.executionTime"},
                        "totalRetries": {"$sum": "$performance.retryCount"},
                        "avgValidationScore": {"$avg": "$validation.score"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "totalExecutions": 1,
                        "successRate": {
                            "$divide": ["$successfulExecutions", "$totalExecutions"]
                        },
                        "avgExecutionTime": {"$round": ["$avgExecutionTime", 2]},
                        "maxExecutionTime": 1,
                        "minExecutionTime": 1,
                        "avgRetryCount": {
                            "$divide": ["$totalRetries", "$totalExecutions"]
                        },
                        "avgValidationScore": {"$round": ["$avgValidationScore", 3]}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(length=1)
            
            if not result:
                return {
                    "totalExecutions": 0,
                    "successRate": 0,
                    "avgExecutionTime": 0,
                    "maxExecutionTime": 0,
                    "minExecutionTime": 0,
                    "avgRetryCount": 0,
                    "avgValidationScore": 0
                }
            
            return result[0]
        except Exception as error:
            logger.error(f"Error getting tool performance analytics: {error}")
            raise error

    async def get_pending_approvals(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get tool executions pending human approval."""
        try:
            match_filter = {
                "humanInteraction.approvalStatus": "pending"
            }

            if agent_id:
                match_filter["agentId"] = agent_id

            cursor = self.collection.find(match_filter).sort([
                ("context.priority", DESCENDING),
                ("timestamp", ASCENDING)
            ])

            return await cursor.to_list(length=None)
        except Exception as error:
            logger.error(f"Error getting pending approvals: {error}")
            raise error

    async def approve_execution(
        self,
        execution_id: ObjectId,
        approver: str,
        feedback: Optional[str] = None
    ) -> bool:
        """Approve a tool execution."""
        try:
            update_data = {
                "humanInteraction.approvalStatus": "approved",
                "humanInteraction.approver": approver,
                "humanInteraction.approvalTimestamp": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }

            if feedback:
                update_data["humanInteraction.feedback"] = feedback

            result = await self.collection.update_one(
                {"executionId": execution_id},
                {"$set": update_data}
            )

            return result.modified_count > 0
        except Exception as error:
            logger.error(f"Error approving execution: {error}")
            raise error

    async def reject_execution(
        self,
        execution_id: ObjectId,
        approver: str,
        reason: str
    ) -> bool:
        """Reject a tool execution."""
        try:
            update_data = {
                "humanInteraction.approvalStatus": "rejected",
                "humanInteraction.approver": approver,
                "humanInteraction.rejectionReason": reason,
                "humanInteraction.approvalTimestamp": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }

            result = await self.collection.update_one(
                {"executionId": execution_id},
                {"$set": update_data}
            )

            return result.modified_count > 0
        except Exception as error:
            logger.error(f"Error rejecting execution: {error}")
            raise error

    async def get_tool_capabilities(
        self,
        tool_name: str
    ) -> Dict[str, Any]:
        """Get tool capabilities and reliability metrics."""
        try:
            # Get recent performance data
            analytics = await self.get_tool_performance_analytics(tool_name, 30)

            # Get common error patterns
            error_pipeline = [
                {
                    "$match": {
                        "toolName": tool_name,
                        "success": False,
                        "timestamp": {
                            "$gte": datetime.utcnow() - timedelta(days=30)
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$error.type",
                        "count": {"$sum": 1},
                        "examples": {"$push": "$error.message"}
                    }
                },
                {
                    "$sort": {"count": -1}
                },
                {
                    "$limit": 5
                }
            ]

            error_patterns = await self.collection.aggregate(error_pipeline).to_list(length=5)

            return {
                "name": tool_name,
                "reliability": {
                    "successRate": analytics.get("successRate", 0),
                    "avgExecutionTime": analytics.get("avgExecutionTime", 0),
                    "errorPatterns": [
                        {
                            "type": pattern["_id"],
                            "frequency": pattern["count"],
                            "examples": pattern["examples"][:3]  # Limit examples
                        }
                        for pattern in error_patterns
                    ]
                },
                "performance": {
                    "totalExecutions": analytics.get("totalExecutions", 0),
                    "avgRetryCount": analytics.get("avgRetryCount", 0),
                    "avgValidationScore": analytics.get("avgValidationScore", 0)
                }
            }
        except Exception as error:
            logger.error(f"Error getting tool capabilities: {error}")
            raise error

    async def get_execution_history(
        self,
        agent_id: str,
        tool_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get execution history for an agent."""
        try:
            match_filter = {"agentId": agent_id}

            if tool_name:
                match_filter["toolName"] = tool_name

            cursor = self.collection.find(match_filter).sort([
                ("timestamp", DESCENDING)
            ]).limit(limit)

            return await cursor.to_list(length=limit)
        except Exception as error:
            logger.error(f"Error getting execution history: {error}")
            raise error
