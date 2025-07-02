"""
AttentionStateCollection - MongoDB collection for real-time attention management

Exact Python equivalent of JavaScript AttentionStateCollection.ts with:
- Real-time attention allocation with change streams
- Cognitive load monitoring and balancing
- Priority-based attention management with queues
- Distraction filtering and attention protection
- Real-time attention analytics and optimization

Features:
- Change streams for real-time attention monitoring
- Complex indexing for attention priority queries
- Real-time updates for cognitive load balancing
- Priority queue management with MongoDB operations
- Advanced aggregation for attention analytics
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from ..base_collection import BaseCollection
from ...core.types import AttentionState, AttentionFilter, AttentionAnalyticsOptions
from ...utils.logger import logger


class AttentionStateCollection(BaseCollection[AttentionState]):
    """
    AttentionStateCollection - Manages real-time attention states with change streams
    
    Exact Python equivalent of JavaScript AttentionStateCollection with:
    - Change streams for real-time attention monitoring
    - Complex indexing for attention priority queries
    - Real-time updates for cognitive load balancing
    - Priority queue management with MongoDB operations
    - Advanced aggregation for attention analytics
    """
    
    @property
    def collection_name(self) -> str:
        return "agent_attention_states"
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)
        self.initialize_collection()
        self.change_stream = None
    
    async def create_indexes(self) -> None:
        """Create indexes optimized for real-time attention management."""
        try:
            # Agent and timestamp index for real-time queries
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_timestamp_realtime", background=True)
            
            # Cognitive load monitoring index
            await self.collection.create_index([
                ("cognitiveLoad.overload", ASCENDING),
                ("cognitiveLoad.utilization", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="cognitive_load_monitoring", background=True)
            
            # Priority queue index
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("attention.primary.priority", ASCENDING),
                ("attention.primary.startTime", DESCENDING)
            ], name="priority_queue_index", background=True)
            
            # Attention efficiency index
            await self.collection.create_index([
                ("attention.efficiency.focusQuality", DESCENDING),
                ("attention.efficiency.distractionLevel", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="attention_efficiency_index", background=True)
            
            # Real-time alerts index
            await self.collection.create_index([
                ("monitoring.alertsEnabled", ASCENDING),
                ("cognitiveLoad.overload", ASCENDING),
                ("attention.efficiency.focusQuality", ASCENDING)
            ], name="realtime_alerts_index", background=True)
            
            # Session analytics index
            await self.collection.create_index([
                ("sessionId", ASCENDING),
                ("analytics.session.attentionEfficiency", DESCENDING),
                ("timestamp", DESCENDING)
            ], name="session_analytics_index", background=True, sparse=True)
            
            logger.info("✅ AttentionStateCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating AttentionStateCollection indexes: {error}")
            raise error
    
    async def record_attention_state(self, state: Dict[str, Any]) -> ObjectId:
        """Record a new attention state."""
        state_with_timestamp = {
            **state,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        
        result = await self.collection.insert_one(state_with_timestamp)
        return result.inserted_id
    
    async def get_current_attention_state(
        self, 
        agent_id: str, 
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get current attention state for an agent (most recent)."""
        filter_dict = {"agentId": agent_id}
        if session_id:
            filter_dict["sessionId"] = session_id
        
        return await self.collection.find_one(
            filter_dict,
            sort=[("timestamp", DESCENDING)]
        )
    
    async def update_attention_allocation(
        self,
        agent_id: str,
        primary_task: Dict[str, Any],
        secondary_tasks: List[Dict[str, Any]] = None
    ) -> None:
        """Update attention allocation in real-time."""
        if secondary_tasks is None:
            secondary_tasks = []
        
        total_allocation = primary_task["focus"] + sum(task["focus"] for task in secondary_tasks)
        
        if total_allocation > 1.0:
            raise ValueError("Total attention allocation cannot exceed 1.0")
        
        current_state = await self.get_current_attention_state(agent_id)
        if not current_state:
            raise ValueError("No current attention state found for agent")
        
        await self.collection.update_one(
            {"_id": current_state["_id"]},
            {
                "$set": {
                    "attention.primary": primary_task,
                    "attention.secondary": secondary_tasks,
                    "attention.totalAllocation": total_allocation,
                    "updatedAt": datetime.utcnow(),
                    "metadata.updateTrigger": "manual"
                }
            }
        )
    
    async def update_cognitive_load(
        self,
        agent_id: str,
        cognitive_load: Dict[str, Any]
    ) -> None:
        """Update cognitive load in real-time."""
        current_state = await self.get_current_attention_state(agent_id)
        if not current_state:
            raise ValueError("No current attention state found for agent")
        
        await self.collection.update_one(
            {"_id": current_state["_id"]},
            {
                "$set": {
                    "cognitiveLoad": cognitive_load,
                    "updatedAt": datetime.utcnow(),
                    "metadata.updateTrigger": "automatic"
                }
            }
        )
    
    async def add_to_priority_queue(
        self,
        agent_id: str,
        priority: str,
        task: Dict[str, Any]
    ) -> None:
        """Add task to priority queue."""
        current_state = await self.get_current_attention_state(agent_id)
        if not current_state:
            raise ValueError("No current attention state found for agent")
        
        queue_task = {
            **task,
            "arrivalTime": datetime.utcnow(),
            "dependencies": task.get("dependencies", [])
        }
        
        await self.collection.update_one(
            {"_id": current_state["_id"]},
            {
                "$push": {
                    f"priorityQueue.{priority}": queue_task
                },
                "$set": {
                    "updatedAt": datetime.utcnow()
                }
            }
        )
    
    async def get_attention_timeline(
        self,
        agent_id: str,
        options: Optional[AttentionAnalyticsOptions] = None
    ) -> List[Dict[str, Any]]:
        """Get attention timeline for an agent."""
        if options is None:
            options = AttentionAnalyticsOptions()
        
        filter_dict = {"agentId": agent_id}
        
        if options.time_range:
            filter_dict["timestamp"] = {
                "$gte": options.time_range["start"],
                "$lte": options.time_range["end"]
            }
        
        if options.min_focus_quality:
            filter_dict["attention.efficiency.focusQuality"] = {"$gte": options.min_focus_quality}
        
        cursor = self.collection.find(filter_dict).sort([("timestamp", ASCENDING)])
        return await cursor.to_list(length=None)
    
    async def analyze_attention_patterns(self, agent_id: str, days: int = 7) -> Dict[str, Any]:
        """Analyze attention patterns using MongoDB aggregation."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Focus patterns analysis
        focus_patterns_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": "$attention.primary.taskType",
                    "frequency": {"$sum": 1},
                    "avgFocus": {"$avg": "$attention.primary.focus"}
                }
            },
            {
                "$project": {
                    "taskType": "$_id",
                    "frequency": 1,
                    "avgFocus": {"$round": ["$avgFocus", 3]},
                    "_id": 0
                }
            },
            {"$sort": {"frequency": -1}}
        ]
        
        focus_patterns = await self.collection.aggregate(focus_patterns_pipeline).to_list(length=None)
        
        # Cognitive load trends by hour
        cognitive_load_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": {"$hour": "$timestamp"},
                    "avgLoad": {"$avg": "$cognitiveLoad.utilization"},
                    "overloadCount": {
                        "$sum": {"$cond": ["$cognitiveLoad.overload", 1, 0]}
                    },
                    "totalCount": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "hour": "$_id",
                    "avgLoad": {"$round": ["$avgLoad", 3]},
                    "overloadFrequency": {
                        "$round": [{"$divide": ["$overloadCount", "$totalCount"]}, 3]
                    },
                    "_id": 0
                }
            },
            {"$sort": {"hour": 1}}
        ]
        
        cognitive_load_trends = await self.collection.aggregate(cognitive_load_pipeline).to_list(length=None)
        
        # Distraction analysis
        distraction_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avgDistractionLevel": {"$avg": "$attention.efficiency.distractionLevel"},
                    "totalDistractions": {"$sum": {"$size": {"$ifNull": ["$distractions.active", []]}}},
                    "filteredCount": {
                        "$sum": {
                            "$size": {
                                "$filter": {
                                    "input": {"$ifNull": ["$distractions.active", []]},
                                    "cond": {"$eq": ["$$this.filtered", True]}
                                }
                            }
                        }
                    }
                }
            }
        ]
        
        distraction_result = await self.collection.aggregate(distraction_pipeline).to_list(length=1)
        distraction_stats = distraction_result[0] if distraction_result else {}
        
        # Efficiency metrics
        efficiency_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avgFocusQuality": {"$avg": "$attention.efficiency.focusQuality"},
                    "avgTaskSwitchingCost": {"$avg": "$attention.efficiency.taskSwitchingCost"},
                    "avgAttentionStability": {"$avg": "$attention.efficiency.attentionStability"}
                }
            }
        ]
        
        efficiency_result = await self.collection.aggregate(efficiency_pipeline).to_list(length=1)
        efficiency_stats = efficiency_result[0] if efficiency_result else {}
        
        # Generate recommendations
        recommendations = self._generate_attention_recommendations(
            focus_patterns,
            cognitive_load_trends,
            distraction_stats,
            efficiency_stats
        )
        
        return {
            "focusPatterns": focus_patterns,
            "cognitiveLoadTrends": cognitive_load_trends,
            "distractionAnalysis": {
                "avgDistractionLevel": distraction_stats.get("avgDistractionLevel", 0),
                "topSources": [],  # Would extract from distractions data
                "filteringEffectiveness": (
                    distraction_stats.get("filteredCount", 0) / distraction_stats.get("totalDistractions", 1)
                    if distraction_stats.get("totalDistractions", 0) > 0 else 0
                )
            },
            "efficiencyMetrics": {
                "avgFocusQuality": efficiency_stats.get("avgFocusQuality", 0),
                "taskSwitchingCost": efficiency_stats.get("avgTaskSwitchingCost", 0),
                "attentionStability": efficiency_stats.get("avgAttentionStability", 0)
            },
            "recommendations": recommendations
        }

    async def get_attention_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get attention statistics."""
        filter_dict = {"agentId": agent_id} if agent_id else {}

        stats_pipeline = [
            {"$match": filter_dict},
            {
                "$group": {
                    "_id": None,
                    "totalStates": {"$sum": 1},
                    "avgFocusQuality": {"$avg": "$attention.efficiency.focusQuality"},
                    "avgCognitiveLoad": {"$avg": "$cognitiveLoad.utilization"},
                    "overloadCount": {
                        "$sum": {"$cond": ["$cognitiveLoad.overload", 1, 0]}
                    },
                    "avgDistractionLevel": {"$avg": "$attention.efficiency.distractionLevel"}
                }
            }
        ]

        stats_result = await self.collection.aggregate(stats_pipeline).to_list(length=1)

        result = stats_result[0] if stats_result else {
            "totalStates": 0,
            "avgFocusQuality": 0,
            "avgCognitiveLoad": 0,
            "overloadCount": 0,
            "avgDistractionLevel": 0
        }

        return {
            "totalStates": result["totalStates"],
            "avgFocusQuality": result.get("avgFocusQuality", 0),
            "avgCognitiveLoad": result.get("avgCognitiveLoad", 0),
            "overloadFrequency": (
                result["overloadCount"] / result["totalStates"]
                if result["totalStates"] > 0 else 0
            ),
            "avgDistractionLevel": result.get("avgDistractionLevel", 0)
        }

    def _generate_attention_recommendations(
        self,
        focus_patterns: List[Dict[str, Any]],
        cognitive_load_trends: List[Dict[str, Any]],
        distraction_stats: Dict[str, Any],
        efficiency_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate attention recommendations."""
        recommendations = []

        if efficiency_stats.get("avgFocusQuality", 0) < 0.6:
            recommendations.append("Focus quality is below optimal - consider reducing task switching")

        if distraction_stats.get("avgDistractionLevel", 0) > 0.5:
            recommendations.append("High distraction levels detected - enable stronger filtering")

        high_load_hours = [
            trend for trend in cognitive_load_trends
            if trend.get("avgLoad", 0) > 0.8
        ]
        if high_load_hours:
            hours = [str(h["hour"]) for h in high_load_hours]
            recommendations.append(f"High cognitive load during hours: {', '.join(hours)}")

        if efficiency_stats.get("avgTaskSwitchingCost", 0) > 0.7:
            recommendations.append("High task switching costs - consider batching similar tasks")

        if efficiency_stats.get("avgAttentionStability", 0) < 0.5:
            recommendations.append("Low attention stability - implement focus protection periods")

        # Analyze focus patterns for recommendations
        if focus_patterns:
            low_focus_tasks = [
                pattern for pattern in focus_patterns
                if pattern.get("avgFocus", 0) < 0.5
            ]
            if low_focus_tasks:
                task_types = [task["taskType"] for task in low_focus_tasks]
                recommendations.append(f"Low focus on task types: {', '.join(task_types)} - consider restructuring")

        return recommendations

    async def enable_real_time_monitoring(self, agent_id: str) -> None:
        """Enable real-time monitoring for an agent."""
        current_state = await self.get_current_attention_state(agent_id)
        if not current_state:
            raise ValueError("No current attention state found for agent")

        await self.collection.update_one(
            {"_id": current_state["_id"]},
            {
                "$set": {
                    "monitoring.alertsEnabled": True,
                    "monitoring.thresholds": {
                        "overloadWarning": 0.8,
                        "focusDegradation": 0.4,
                        "distractionAlert": 0.6
                    },
                    "updatedAt": datetime.utcnow()
                }
            }
        )

    async def check_attention_alerts(self, agent_id: str) -> List[Dict[str, Any]]:
        """Check for attention alerts based on thresholds."""
        current_state = await self.get_current_attention_state(agent_id)
        if not current_state or not current_state.get("monitoring", {}).get("alertsEnabled"):
            return []

        alerts = []
        thresholds = current_state["monitoring"]["thresholds"]

        # Check cognitive overload
        if current_state["cognitiveLoad"]["overload"]:
            alerts.append({
                "type": "overload",
                "timestamp": datetime.utcnow(),
                "severity": "critical",
                "message": "Cognitive overload detected",
                "resolved": False
            })

        # Check focus degradation
        focus_quality = current_state["attention"]["efficiency"]["focusQuality"]
        if focus_quality < thresholds["focusDegradation"]:
            alerts.append({
                "type": "focus_degradation",
                "timestamp": datetime.utcnow(),
                "severity": "high" if focus_quality < 0.3 else "medium",
                "message": f"Focus quality degraded to {focus_quality:.2f}",
                "resolved": False
            })

        # Check high distraction
        distraction_level = current_state["attention"]["efficiency"]["distractionLevel"]
        if distraction_level > thresholds["distractionAlert"]:
            alerts.append({
                "type": "high_distraction",
                "timestamp": datetime.utcnow(),
                "severity": "high" if distraction_level > 0.8 else "medium",
                "message": f"High distraction level: {distraction_level:.2f}",
                "resolved": False
            })

        # Update alert history if there are new alerts
        if alerts:
            await self.collection.update_one(
                {"_id": current_state["_id"]},
                {
                    "$push": {
                        "monitoring.alertHistory": {"$each": alerts}
                    },
                    "$set": {
                        "monitoring.lastAlert": datetime.utcnow(),
                        "updatedAt": datetime.utcnow()
                    }
                }
            )

        return alerts
