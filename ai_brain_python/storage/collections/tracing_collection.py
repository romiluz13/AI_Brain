"""
TracingCollection - MongoDB collection for execution tracing and monitoring

Manages execution traces, performance metrics, and system monitoring
with real-time analytics and alerting capabilities.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...utils.logger import logger


class TracingCollection:
    """MongoDB collection for execution tracing and monitoring."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.execution_traces
    
    async def create_indexes(self) -> None:
        """Create indexes for tracing collection."""
        indexes = [
            # Primary indexes
            {"traceId": 1},
            {"agentId": 1, "timestamp": -1},
            {"sessionId": 1, "timestamp": -1},
            
            # Execution indexes
            {"execution.operation": 1, "timestamp": -1},
            {"execution.status": 1, "timestamp": -1},
            {"execution.duration": 1},
            
            # Performance indexes
            {"performance.cpuUsage": 1},
            {"performance.memoryUsage": 1},
            {"performance.responseTime": 1, "timestamp": -1},
            
            # Error indexes
            {"error.occurred": 1, "timestamp": -1},
            {"error.type": 1},
            {"error.severity": 1, "timestamp": -1},
            
            # Context indexes
            {"context.component": 1},
            {"context.environment": 1},
            
            # TTL index for automatic cleanup (30 days)
            {"timestamp": 1}
        ]
        
        for index in indexes:
            try:
                if "timestamp" in index:
                    await self.collection.create_index(
                        list(index.items()),
                        expireAfterSeconds=60 * 60 * 24 * 30
                    )
                else:
                    await self.collection.create_index(list(index.items()))
            except Exception as error:
                logger.warning(f"Could not create tracing index {index}: {error}")
    
    async def store_trace(self, trace: Dict[str, Any]) -> ObjectId:
        """Store execution trace."""
        result = await self.collection.insert_one(trace)
        return result.inserted_id
    
    async def get_trace(self, trace_id: ObjectId) -> Optional[Dict[str, Any]]:
        """Get trace by ID."""
        return await self.collection.find_one({"traceId": trace_id})
    
    async def get_agent_traces(
        self,
        agent_id: str,
        timeframe_hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get traces for an agent."""
        start_date = datetime.utcnow() - timedelta(hours=timeframe_hours)
        
        cursor = self.collection.find({
            "agentId": agent_id,
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_performance_metrics(
        self,
        agent_id: str,
        timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance metrics for an agent."""
        start_date = datetime.utcnow() - timedelta(hours=timeframe_hours)
        
        pipeline = [
            {"$match": {
                "agentId": agent_id,
                "timestamp": {"$gte": start_date}
            }},
            {"$group": {
                "_id": None,
                "avgCpuUsage": {"$avg": "$performance.cpuUsage"},
                "avgMemoryUsage": {"$avg": "$performance.memoryUsage"},
                "avgResponseTime": {"$avg": "$performance.responseTime"},
                "totalOperations": {"$sum": 1},
                "errorCount": {"$sum": {"$cond": ["$error.occurred", 1, 0]}}
            }},
            {"$project": {
                "avgCpuUsage": 1,
                "avgMemoryUsage": 1,
                "avgResponseTime": 1,
                "totalOperations": 1,
                "errorRate": {"$divide": ["$errorCount", "$totalOperations"]}
            }}
        ]
        
        result = await self.collection.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return {
                "avgCpuUsage": 0,
                "avgMemoryUsage": 0,
                "avgResponseTime": 0,
                "totalOperations": 0,
                "errorRate": 0
            }
        
        return result[0]
    
    async def get_error_traces(
        self,
        agent_id: str,
        timeframe_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get error traces for debugging."""
        start_date = datetime.utcnow() - timedelta(hours=timeframe_hours)
        
        cursor = self.collection.find({
            "agentId": agent_id,
            "error.occurred": True,
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1)
        
        return await cursor.to_list(length=None)
