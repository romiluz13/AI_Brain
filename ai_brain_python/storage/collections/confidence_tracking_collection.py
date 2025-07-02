"""
ConfidenceTrackingCollection - MongoDB collection for confidence tracking

Manages confidence records with time-series capabilities, confidence analytics,
and performance tracking using MongoDB's advanced indexing and aggregation.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...utils.logger import logger


class ConfidenceTrackingCollection:
    """MongoDB collection for confidence tracking and analytics."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.confidence_tracking
    
    async def create_indexes(self) -> None:
        """Create indexes for confidence tracking collection."""
        indexes = [
            # Primary indexes
            {"recordId": 1},
            {"agentId": 1, "timestamp": -1},
            {"sessionId": 1, "timestamp": -1},
            
            # Confidence indexes
            {"confidence.overall": 1, "timestamp": -1},
            {"confidence.domain": 1},
            {"confidence.task": 1},
            
            # Performance indexes
            {"performance.accuracy": 1, "timestamp": -1},
            {"performance.consistency": 1},
            
            # Context indexes
            {"context.domain": 1},
            {"context.complexity": 1},
            
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
                logger.warning(f"Could not create confidence index {index}: {error}")
    
    async def store_confidence_record(self, record: Dict[str, Any]) -> ObjectId:
        """Store confidence record."""
        result = await self.collection.insert_one(record)
        return result.inserted_id
    
    async def get_confidence_record(self, record_id: ObjectId) -> Optional[Dict[str, Any]]:
        """Get confidence record by ID."""
        return await self.collection.find_one({"recordId": record_id})
    
    async def get_agent_confidence_history(
        self,
        agent_id: str,
        timeframe_days: int = 30,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get confidence history for an agent."""
        start_date = datetime.utcnow() - timedelta(days=timeframe_days)
        
        cursor = self.collection.find({
            "agentId": agent_id,
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_confidence_analytics(
        self,
        agent_id: str,
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """Get confidence analytics for an agent."""
        start_date = datetime.utcnow() - timedelta(days=timeframe_days)
        
        pipeline = [
            {"$match": {
                "agentId": agent_id,
                "timestamp": {"$gte": start_date}
            }},
            {"$group": {
                "_id": None,
                "avgConfidence": {"$avg": "$confidence.overall"},
                "maxConfidence": {"$max": "$confidence.overall"},
                "minConfidence": {"$min": "$confidence.overall"},
                "totalRecords": {"$sum": 1},
                "avgAccuracy": {"$avg": "$performance.accuracy"}
            }}
        ]
        
        result = await self.collection.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return {
                "avgConfidence": 0,
                "maxConfidence": 0,
                "minConfidence": 0,
                "totalRecords": 0,
                "avgAccuracy": 0
            }
        
        return result[0]
