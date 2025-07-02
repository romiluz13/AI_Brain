"""
GoalHierarchyCollection - MongoDB collection for hierarchical goal management

Exact Python equivalent of JavaScript GoalHierarchyCollection.ts with:
- Materialized paths for efficient tree operations
- Goal decomposition and sub-goal management
- Progress tracking with aggregation pipelines
- Dependency management and constraint satisfaction
- Real-time goal status updates

Features:
- Materialized paths pattern for hierarchical data
- Complex aggregation pipelines for goal analytics
- Dependency tracking with graph-like operations
- Real-time progress monitoring and propagation
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from ..base_collection import BaseCollection
from ...core.types import Goal, GoalFilter, GoalUpdateData, GoalAnalyticsOptions
from ...utils.logger import logger


class GoalHierarchyCollection(BaseCollection[Goal]):
    """
    GoalHierarchyCollection - Manages hierarchical goal structures with materialized paths
    
    Exact Python equivalent of JavaScript GoalHierarchyCollection with:
    - Materialized paths for efficient tree operations
    - Complex aggregation pipelines for goal analytics
    - Dependency tracking with graph-like operations
    - Real-time progress monitoring
    """
    
    @property
    def collection_name(self) -> str:
        return "agent_goal_hierarchies"
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)
        self.initialize_collection()
    
    async def create_indexes(self) -> None:
        """Create indexes optimized for hierarchical queries and goal management."""
        try:
            # Materialized path index for tree operations
            await self.collection.create_index([
                ("path", ASCENDING)
            ], name="materialized_path_index", background=True)
            
            # Agent and level index for efficient filtering
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("level", ASCENDING),
                ("status", ASCENDING)
            ], name="agent_level_status_index", background=True)
            
            # Priority and deadline index for urgent goals
            await self.collection.create_index([
                ("goal.priority", ASCENDING),
                ("timeline.deadline", ASCENDING),
                ("status", ASCENDING)
            ], name="priority_deadline_index", background=True)
            
            # Progress tracking index
            await self.collection.create_index([
                ("agentId", ASCENDING),
                ("progress.percentage", DESCENDING),
                ("progress.lastUpdated", DESCENDING)
            ], name="progress_tracking_index", background=True)
            
            # Dependency tracking index
            await self.collection.create_index([
                ("dependencies.requiredGoals", ASCENDING)
            ], name="dependency_tracking_index", background=True, sparse=True)
            
            # Session-based goal tracking
            await self.collection.create_index([
                ("sessionId", ASCENDING),
                ("timeline.startTime", DESCENDING)
            ], name="session_timeline_index", background=True, sparse=True)
            
            logger.info("✅ GoalHierarchyCollection indexes created successfully")
        except Exception as error:
            logger.error(f"❌ Error creating GoalHierarchyCollection indexes: {error}")
            raise error
    
    async def create_goal(self, goal: Dict[str, Any]) -> ObjectId:
        """Create a new goal with automatic path generation."""
        # Generate materialized path
        path = "/root"
        level = 0
        
        if goal.get("parentId"):
            parent = await self.collection.find_one({"_id": ObjectId(goal["parentId"])})
            if parent:
                path = f"{parent['path']}/{goal['parentId']}"
                level = parent["level"] + 1
        
        goal_with_path = {
            **goal,
            "path": path,
            "level": level,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        
        result = await self.collection.insert_one(goal_with_path)
        return result.inserted_id
    
    async def get_goal_hierarchy(
        self, 
        agent_id: str, 
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get goal hierarchy tree for an agent."""
        filter_dict = {"agentId": agent_id}
        if session_id:
            filter_dict["sessionId"] = session_id
        
        cursor = self.collection.find(filter_dict).sort([("path", ASCENDING), ("level", ASCENDING)])
        return await cursor.to_list(length=None)
    
    async def get_sub_goals(self, goal_id: Union[str, ObjectId]) -> List[Dict[str, Any]]:
        """Get all sub-goals of a specific goal using materialized paths."""
        goal_object_id = ObjectId(goal_id) if isinstance(goal_id, str) else goal_id
        parent_goal = await self.collection.find_one({"_id": goal_object_id})
        
        if not parent_goal:
            raise ValueError("Parent goal not found")
        
        # Use materialized path to find all descendants
        path_regex = f"^{re.escape(parent_goal['path'])}/{goal_object_id}"
        
        cursor = self.collection.find({
            "path": {"$regex": path_regex}
        }).sort([("level", ASCENDING), ("path", ASCENDING)])
        
        return await cursor.to_list(length=None)
    
    async def get_goal_dependencies(self, goal_id: Union[str, ObjectId]) -> Dict[str, Any]:
        """Get goal dependencies and check for conflicts."""
        goal_object_id = ObjectId(goal_id) if isinstance(goal_id, str) else goal_id
        goal = await self.collection.find_one({"_id": goal_object_id})
        
        if not goal:
            raise ValueError("Goal not found")
        
        # Get all dependency goals in parallel
        required_cursor = self.collection.find({"_id": {"$in": goal["dependencies"]["requiredGoals"]}})
        blockers_cursor = self.collection.find({"_id": {"$in": goal["dependencies"]["blockedBy"]}})
        enabled_cursor = self.collection.find({"_id": {"$in": goal["dependencies"]["enables"]}})
        conflicts_cursor = self.collection.find({"_id": {"$in": goal["dependencies"]["conflicts"]}})
        
        required = await required_cursor.to_list(length=None)
        blockers = await blockers_cursor.to_list(length=None)
        enabled = await enabled_cursor.to_list(length=None)
        conflicts = await conflicts_cursor.to_list(length=None)
        
        # Check if goal can start (all required goals completed, no blockers)
        can_start = (
            all(req["status"] == "completed" for req in required) and 
            len(blockers) == 0
        )
        
        return {
            "required": required,
            "blockers": blockers,
            "enabled": enabled,
            "conflicts": conflicts,
            "canStart": can_start
        }
    
    async def update_goal_progress(self, goal_id: Union[str, ObjectId], progress: float) -> None:
        """Update goal progress and propagate to parent goals."""
        goal_object_id = ObjectId(goal_id) if isinstance(goal_id, str) else goal_id
        goal = await self.collection.find_one({"_id": goal_object_id})
        
        if not goal:
            raise ValueError("Goal not found")
        
        # Update current goal
        await self.collection.update_one(
            {"_id": goal_object_id},
            {
                "$set": {
                    "progress.percentage": progress,
                    "progress.lastUpdated": datetime.utcnow(),
                    "status": "completed" if progress == 100 else "in_progress",
                    "updatedAt": datetime.utcnow()
                }
            }
        )
        
        # Propagate progress to parent goals
        if goal.get("parentId"):
            await self._propagate_progress_to_parent(ObjectId(goal["parentId"]))
    
    async def _propagate_progress_to_parent(self, parent_id: ObjectId) -> None:
        """Propagate progress changes to parent goals."""
        parent = await self.collection.find_one({"_id": parent_id})
        if not parent:
            return
        
        # Get all direct children
        children_cursor = self.collection.find({"parentId": parent_id})
        children = await children_cursor.to_list(length=None)
        
        if len(children) == 0:
            return
        
        # Calculate average progress of children
        total_progress = sum(child["progress"]["percentage"] for child in children)
        avg_progress = round(total_progress / len(children))
        completed_children = len([child for child in children if child["status"] == "completed"])
        
        # Update parent progress
        await self.collection.update_one(
            {"_id": parent_id},
            {
                "$set": {
                    "progress.percentage": avg_progress,
                    "progress.completedSubGoals": completed_children,
                    "progress.totalSubGoals": len(children),
                    "progress.lastUpdated": datetime.utcnow(),
                    "status": "completed" if avg_progress == 100 else "in_progress",
                    "updatedAt": datetime.utcnow()
                }
            }
        )
        
        # Continue propagation up the hierarchy
        if parent.get("parentId"):
            await self._propagate_progress_to_parent(ObjectId(parent["parentId"]))
    
    async def analyze_goal_patterns(self, agent_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze goal completion patterns using MongoDB aggregation."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Completion rate analysis
        completion_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "createdAt": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "totalGoals": {"$sum": 1},
                    "completedGoals": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    },
                    "avgDuration": {"$avg": "$timeline.actualDuration"}
                }
            }
        ]
        
        completion_result = await self.collection.aggregate(completion_pipeline).to_list(length=1)
        completion_rate = 0
        avg_duration = 0
        
        if completion_result:
            total = completion_result[0]["totalGoals"]
            completed = completion_result[0]["completedGoals"]
            completion_rate = completed / total if total > 0 else 0
            avg_duration = completion_result[0]["avgDuration"] or 0
        
        # Priority distribution analysis
        priority_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "createdAt": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": "$goal.priority",
                    "count": {"$sum": 1},
                    "avgCompletion": {"$avg": "$progress.percentage"}
                }
            },
            {
                "$project": {
                    "priority": "$_id",
                    "count": 1,
                    "avgCompletion": {"$round": ["$avgCompletion", 1]},
                    "_id": 0
                }
            },
            {"$sort": {"count": -1}}
        ]
        
        priority_distribution = await self.collection.aggregate(priority_pipeline).to_list(length=None)
        
        # Type analysis
        type_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "createdAt": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": "$goal.type",
                    "count": {"$sum": 1},
                    "successCount": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    }
                }
            },
            {
                "$project": {
                    "type": "$_id",
                    "count": 1,
                    "successRate": {
                        "$cond": [
                            {"$eq": ["$count", 0]},
                            0,
                            {"$divide": ["$successCount", "$count"]}
                        ]
                    },
                    "_id": 0
                }
            }
        ]
        
        type_analysis = await self.collection.aggregate(type_pipeline).to_list(length=None)
        
        # Difficulty analysis
        difficulty_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "createdAt": {"$gte": start_date},
                    "learning.difficulty": {"$exists": True},
                    "learning.satisfaction": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avgDifficulty": {"$avg": "$learning.difficulty"},
                    "difficulties": {"$push": "$learning.difficulty"},
                    "satisfactions": {"$push": "$learning.satisfaction"}
                }
            }
        ]

        difficulty_result = await self.collection.aggregate(difficulty_pipeline).to_list(length=1)
        avg_difficulty = difficulty_result[0]["avgDifficulty"] if difficulty_result else 0
        satisfaction_correlation = 0  # Simplified for now

        # Timeline accuracy analysis
        timeline_pipeline = [
            {
                "$match": {
                    "agentId": agent_id,
                    "createdAt": {"$gte": start_date},
                    "timeline.deadline": {"$exists": True},
                    "timeline.endTime": {"$exists": True}
                }
            },
            {
                "$project": {
                    "onTime": {
                        "$cond": [
                            {"$lte": ["$timeline.endTime", "$timeline.deadline"]},
                            1,
                            0
                        ]
                    },
                    "delay": {
                        "$cond": [
                            {"$gt": ["$timeline.endTime", "$timeline.deadline"]},
                            {
                                "$divide": [
                                    {"$subtract": ["$timeline.endTime", "$timeline.deadline"]},
                                    1000 * 60  # Convert to minutes
                                ]
                            },
                            0
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "totalWithDeadlines": {"$sum": 1},
                    "onTimeCount": {"$sum": "$onTime"},
                    "avgDelay": {"$avg": "$delay"}
                }
            }
        ]

        timeline_result = await self.collection.aggregate(timeline_pipeline).to_list(length=1)
        on_time_rate = 0
        avg_delay = 0

        if timeline_result:
            total_with_deadlines = timeline_result[0]["totalWithDeadlines"]
            on_time_count = timeline_result[0]["onTimeCount"]
            on_time_rate = on_time_count / total_with_deadlines if total_with_deadlines > 0 else 0
            avg_delay = timeline_result[0]["avgDelay"] or 0

        return {
            "completionRate": completion_rate,
            "avgDuration": avg_duration,
            "priorityDistribution": priority_distribution,
            "typeAnalysis": type_analysis,
            "difficultyAnalysis": {
                "avgDifficulty": avg_difficulty,
                "satisfactionCorrelation": satisfaction_correlation
            },
            "timelineAccuracy": {
                "onTimeRate": on_time_rate,
                "avgDelay": avg_delay
            }
        }

    async def get_goal_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get goal statistics."""
        filter_dict = {"agentId": agent_id} if agent_id else {}

        # Main statistics
        stats_pipeline = [
            {"$match": filter_dict},
            {
                "$group": {
                    "_id": None,
                    "totalGoals": {"$sum": 1},
                    "activeGoals": {
                        "$sum": {
                            "$cond": [
                                {"$in": ["$status", ["in_progress", "not_started"]]},
                                1,
                                0
                            ]
                        }
                    },
                    "completedGoals": {
                        "$sum": {
                            "$cond": [{"$eq": ["$status", "completed"]}, 1, 0]
                        }
                    },
                    "avgProgress": {"$avg": "$progress.percentage"}
                }
            }
        ]

        stats_result = await self.collection.aggregate(stats_pipeline).to_list(length=1)

        # Goals by level
        level_pipeline = [
            {"$match": filter_dict},
            {
                "$group": {
                    "_id": "$level",
                    "count": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "level": "$_id",
                    "count": 1,
                    "_id": 0
                }
            },
            {"$sort": {"level": 1}}
        ]

        goals_by_level = await self.collection.aggregate(level_pipeline).to_list(length=None)

        return {
            "totalGoals": stats_result[0]["totalGoals"] if stats_result else 0,
            "activeGoals": stats_result[0]["activeGoals"] if stats_result else 0,
            "completedGoals": stats_result[0]["completedGoals"] if stats_result else 0,
            "avgProgress": stats_result[0]["avgProgress"] if stats_result else 0,
            "goalsByLevel": goals_by_level
        }

    async def cleanup_old_goals(self, days: int = 90) -> int:
        """Clean up completed goals older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        result = await self.collection.delete_many({
            "status": "completed",
            "timeline.endTime": {"$lt": cutoff_date}
        })

        return result.deleted_count
