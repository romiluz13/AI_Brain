"""
MetricsCollection - MongoDB collection for system metrics and analytics

Manages system performance metrics, KPIs, and analytics with
time-series data aggregation and real-time monitoring capabilities.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...utils.logger import logger


class MetricsCollection:
    """MongoDB collection for system metrics and analytics."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.system_metrics
    
    async def create_indexes(self) -> None:
        """Create indexes for metrics collection."""
        indexes = [
            # Primary indexes
            {"metricId": 1},
            {"agentId": 1, "timestamp": -1},
            {"sessionId": 1, "timestamp": -1},
            
            # Metric type indexes
            {"metric.type": 1, "timestamp": -1},
            {"metric.category": 1},
            {"metric.name": 1, "timestamp": -1},
            
            # Value indexes
            {"metric.value": 1, "timestamp": -1},
            {"metric.unit": 1},
            
            # Performance indexes
            {"performance.throughput": 1, "timestamp": -1},
            {"performance.latency": 1, "timestamp": -1},
            {"performance.errorRate": 1, "timestamp": -1},
            
            # System indexes
            {"system.component": 1},
            {"system.version": 1},
            {"system.environment": 1},
            
            # TTL index for automatic cleanup (90 days)
            {"timestamp": 1}
        ]
        
        for index in indexes:
            try:
                if "timestamp" in index:
                    await self.collection.create_index(
                        list(index.items()),
                        expireAfterSeconds=60 * 60 * 24 * 90
                    )
                else:
                    await self.collection.create_index(list(index.items()))
            except Exception as error:
                logger.warning(f"Could not create metrics index {index}: {error}")
    
    async def store_metric(self, metric: Dict[str, Any]) -> ObjectId:
        """Store system metric."""
        result = await self.collection.insert_one(metric)
        return result.inserted_id
    
    async def get_metric(self, metric_id: ObjectId) -> Optional[Dict[str, Any]]:
        """Get metric by ID."""
        return await self.collection.find_one({"metricId": metric_id})
    
    async def get_metrics_by_type(
        self,
        agent_id: str,
        metric_type: str,
        timeframe_hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get metrics by type for an agent."""
        start_date = datetime.utcnow() - timedelta(hours=timeframe_hours)
        
        cursor = self.collection.find({
            "agentId": agent_id,
            "metric.type": metric_type,
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_performance_summary(
        self,
        agent_id: str,
        timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance summary for an agent."""
        start_date = datetime.utcnow() - timedelta(hours=timeframe_hours)
        
        pipeline = [
            {"$match": {
                "agentId": agent_id,
                "timestamp": {"$gte": start_date}
            }},
            {"$group": {
                "_id": "$metric.type",
                "avgValue": {"$avg": "$metric.value"},
                "maxValue": {"$max": "$metric.value"},
                "minValue": {"$min": "$metric.value"},
                "count": {"$sum": 1}
            }}
        ]
        
        results = await self.collection.aggregate(pipeline).to_list(length=None)
        
        summary = {}
        for result in results:
            summary[result["_id"]] = {
                "average": result["avgValue"],
                "maximum": result["maxValue"],
                "minimum": result["minValue"],
                "count": result["count"]
            }
        
        return summary
    
    async def get_time_series_data(
        self,
        agent_id: str,
        metric_name: str,
        timeframe_hours: int = 24,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get time-series data for a specific metric."""
        start_date = datetime.utcnow() - timedelta(hours=timeframe_hours)
        
        pipeline = [
            {"$match": {
                "agentId": agent_id,
                "metric.name": metric_name,
                "timestamp": {"$gte": start_date}
            }},
            {"$group": {
                "_id": {
                    "$dateTrunc": {
                        "date": "$timestamp",
                        "unit": "minute",
                        "binSize": interval_minutes
                    }
                },
                "avgValue": {"$avg": "$metric.value"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        return await self.collection.aggregate(pipeline).to_list(length=None)
    
    async def get_system_health(self, agent_id: str) -> Dict[str, Any]:
        """Get overall system health metrics."""
        recent_time = datetime.utcnow() - timedelta(minutes=5)
        
        pipeline = [
            {"$match": {
                "agentId": agent_id,
                "timestamp": {"$gte": recent_time}
            }},
            {"$group": {
                "_id": None,
                "avgThroughput": {"$avg": "$performance.throughput"},
                "avgLatency": {"$avg": "$performance.latency"},
                "avgErrorRate": {"$avg": "$performance.errorRate"},
                "totalMetrics": {"$sum": 1}
            }}
        ]
        
        result = await self.collection.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return {
                "status": "unknown",
                "throughput": 0,
                "latency": 0,
                "errorRate": 0,
                "totalMetrics": 0
            }
        
        data = result[0]
        
        # Determine health status based on metrics
        status = "healthy"
        if data["avgErrorRate"] > 0.05:  # 5% error rate threshold
            status = "degraded"
        if data["avgLatency"] > 1000:  # 1 second latency threshold
            status = "slow"
        
        return {
            "status": status,
            "throughput": data["avgThroughput"],
            "latency": data["avgLatency"],
            "errorRate": data["avgErrorRate"],
            "totalMetrics": data["totalMetrics"]
        }
