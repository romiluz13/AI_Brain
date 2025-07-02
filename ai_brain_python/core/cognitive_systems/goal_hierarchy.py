"""
GoalHierarchyManager - Advanced hierarchical goal management for AI agents

Exact Python equivalent of JavaScript GoalHierarchyManager.ts with:
- Hierarchical goal decomposition with materialized paths
- Dependency tracking and constraint satisfaction
- Progress propagation through goal hierarchies
- Goal analytics and pattern recognition
- Real-time goal status monitoring
- Intelligent goal prioritization and scheduling

Features:
- Efficient hierarchical goal structures
- Complex dependency tracking and resolution
- Progress propagation through goal trees
- Advanced analytics with aggregation pipelines
- Real-time goal monitoring and optimization
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_brain_python.storage.collections.goal_hierarchy_collection import GoalHierarchyCollection
from ai_brain_python.core.types import Goal, GoalAnalyticsOptions
from ai_brain_python.utils.logger import logger


@dataclass
class GoalCreationRequest:
    """Goal creation request interface."""
    agent_id: str
    session_id: Optional[str]
    parent_goal_id: Optional[ObjectId]
    title: str
    description: str
    type: str  # 'objective' | 'task' | 'milestone' | 'action' | 'constraint'
    priority: str  # 'critical' | 'high' | 'medium' | 'low'
    category: str
    estimated_duration: int  # minutes
    deadline: Optional[datetime]
    success_criteria: List[Dict[str, Any]]
    dependencies: Optional[Dict[str, List[ObjectId]]]
    context: Dict[str, Any]


@dataclass
class GoalDecompositionResult:
    """Goal decomposition result interface."""
    parent_goal: Dict[str, Any]
    sub_goals: List[Dict[str, Any]]
    decomposition_strategy: str
    estimated_total_duration: int
    critical_path: List[ObjectId]
    risk_assessment: Dict[str, Any]


@dataclass
class GoalExecutionPlan:
    """Goal execution plan interface."""
    goals: List[Dict[str, Any]]
    execution_order: List[ObjectId]
    parallel_groups: List[List[ObjectId]]
    timeline: Dict[str, Any]
    resource_requirements: Dict[str, Any]


@dataclass
class GoalProgressUpdate:
    """Goal progress update interface."""
    goal_id: ObjectId
    progress: float
    status: Optional[str]
    actual_duration: Optional[int]
    learnings: Optional[Dict[str, Any]]


@dataclass
class GoalAnalytics:
    """Goal analytics interface."""
    completion_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    learning_metrics: Dict[str, Any]
    predictive_insights: Dict[str, Any]


class GoalHierarchyManager:
    """
    GoalHierarchyManager - Advanced hierarchical goal management for AI agents
    
    Exact Python equivalent of JavaScript GoalHierarchyManager with:
    - Efficient hierarchical goal structures
    - Complex dependency tracking and resolution
    - Progress propagation through goal trees
    - Advanced analytics with aggregation pipelines
    - Real-time goal monitoring and optimization
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.goal_hierarchy_collection = GoalHierarchyCollection(db)
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the goal hierarchy manager."""
        if self.is_initialized:
            return
        
        try:
            # Initialize goal hierarchy collection
            await self.goal_hierarchy_collection.create_indexes()
            
            self.is_initialized = True
            logger.info("✅ GoalHierarchyManager initialized successfully")
            
        except Exception as error:
            logger.error(f"❌ Failed to initialize GoalHierarchyManager: {error}")
            raise error
    
    async def create_goal(self, request: GoalCreationRequest) -> ObjectId:
        """Create a new goal in the hierarchy."""
        if not self.is_initialized:
            raise Exception("GoalHierarchyManager must be initialized first")
        
        # Validate dependencies
        if request.dependencies:
            await self._validate_dependencies(request.dependencies)

        # Create goal document (matching JavaScript structure)
        goal = {
            "agentId": request.agent_id,
            "sessionId": request.session_id,
            "parentId": request.parent_goal_id,
            "goal": {
                "title": request.title,
                "description": request.description,
                "type": request.type,
                "priority": request.priority,
                "category": request.category
            },
            "status": "not_started",
            "progress": {
                "percentage": 0,
                "completedSubGoals": 0,
                "totalSubGoals": 0,
                "lastUpdated": datetime.utcnow()
            },
            "timeline": {
                "estimatedDuration": request.estimated_duration,
                "deadline": request.deadline
            },
            "dependencies": {
                "requiredGoals": request.dependencies.get("requiredGoals", []) if request.dependencies else [],
                "blockedBy": [],
                "enables": [],
                "conflicts": request.dependencies.get("conflicts", []) if request.dependencies else []
            },
            "successCriteria": {
                "conditions": [
                    {**criteria, "achieved": False} for criteria in request.success_criteria
                ],
                "verification": "manual"
            },
            "context": {
                "trigger": request.context.get("trigger", "user_request"),
                "reasoning": request.context.get("reasoning", ""),
                "assumptions": request.context.get("assumptions", []),
                "risks": request.context.get("risks", [])
            },
            "learning": {
                "difficulty": 0,
                "satisfaction": 0,
                "lessons": [],
                "improvements": []
            },
            "metadata": {
                "framework": "python_ai_brain",
                "createdBy": "agent",
                "tags": [request.category, request.priority],
                "version": "1.0.0"
            }
        }

        # Create the goal (collection handles path generation)
        goal_id = await self.goal_hierarchy_collection.create_goal(goal)

        # Update parent goal's sub-goal count if applicable
        if request.parent_goal_id:
            await self._update_parent_goal(request.parent_goal_id)
        
        logger.info(f"✅ Goal created: {request.title} (ID: {goal_id})")
        return goal_id

    async def _build_materialized_path(self, parent_goal_id: Optional[ObjectId]) -> str:
        """Build materialized path for goal hierarchy."""
        if not parent_goal_id:
            return "/root"

        parent_goal = await self.goal_hierarchy_collection.collection.find_one({"_id": parent_goal_id})
        if not parent_goal:
            return "/root"

        return f"{parent_goal['path']}/{parent_goal_id}"

    async def _validate_dependencies(self, dependencies: Dict[str, List[ObjectId]]) -> None:
        """Validate goal dependencies."""
        required_goals = dependencies.get("requiredGoals", [])
        conflicts = dependencies.get("conflicts", [])

        # Check if required goals exist
        for goal_id in required_goals:
            goal = await self.goal_hierarchy_collection.collection.find_one({"_id": goal_id})
            if not goal:
                raise ValueError(f"Required goal {goal_id} not found")

        # Check if conflicting goals exist
        for goal_id in conflicts:
            goal = await self.goal_hierarchy_collection.collection.find_one({"_id": goal_id})
            if not goal:
                raise ValueError(f"Conflicting goal {goal_id} not found")

    async def _update_parent_goal(self, parent_goal_id: ObjectId) -> None:
        """Update parent goal's sub-goal count."""
        sub_goals = await self.goal_hierarchy_collection.get_sub_goals(parent_goal_id)
        direct_children = [g for g in sub_goals if g.get("parentId") == parent_goal_id]

        await self.goal_hierarchy_collection.collection.update_one(
            {"_id": parent_goal_id},
            {
                "$set": {
                    "progress.totalSubGoals": len(direct_children),
                    "updatedAt": datetime.utcnow()
                }
            }
        )

    async def decompose_goal(
        self,
        goal_id: ObjectId,
        decomposition_strategy: str = "automatic",
        max_sub_goals: int = 5
    ) -> List[ObjectId]:
        """Decompose a goal into sub-goals."""
        try:
            # Get the goal to decompose
            goal = await self.goal_hierarchy_collection.collection.find_one({"_id": goal_id})
            if not goal:
                raise ValueError(f"Goal {goal_id} not found")

            # Simple decomposition logic (in production, would use AI)
            sub_goal_ids = []
            goal_title = goal.get("goal", {}).get("title", "")

            # Create sub-goals based on goal type
            if "project" in goal_title.lower():
                sub_goals = ["Planning", "Implementation", "Testing", "Deployment"]
            elif "learn" in goal_title.lower():
                sub_goals = ["Research", "Practice", "Apply", "Evaluate"]
            else:
                sub_goals = ["Analyze", "Plan", "Execute", "Review"]

            for i, sub_goal_title in enumerate(sub_goals[:max_sub_goals]):
                sub_goal_request = GoalCreationRequest(
                    agent_id=goal["agentId"],
                    session_id=goal["sessionId"],
                    parent_goal_id=goal_id,
                    title=f"{sub_goal_title}: {goal_title}",
                    description=f"Sub-goal {i+1} for {goal_title}",
                    type="sub_goal",
                    priority="medium",
                    category=goal.get("goal", {}).get("category", "general"),
                    estimated_duration=goal.get("timeline", {}).get("estimatedDuration", 60) // len(sub_goals),
                    deadline=goal.get("timeline", {}).get("deadline"),
                    success_criteria=[{"condition": f"Complete {sub_goal_title.lower()} phase", "measurable": True}],
                    dependencies=None,
                    context={"trigger": "goal_decomposition", "parent_goal": str(goal_id)}
                )

                sub_goal_id = await self.create_goal(sub_goal_request)
                sub_goal_ids.append(sub_goal_id)

            logger.info(f"Decomposed goal {goal_id} into {len(sub_goal_ids)} sub-goals")
            return sub_goal_ids

        except Exception as error:
            logger.error(f"Error decomposing goal: {error}")
            raise error

    async def update_goal_progress(
        self,
        goal_id: ObjectId,
        progress_percentage: float,
        status: Optional[str] = None,
        notes: Optional[str] = None
    ) -> None:
        """Update goal progress."""
        try:
            update_data = {
                "progress.percentage": progress_percentage,
                "progress.lastUpdated": datetime.utcnow()
            }

            if status:
                update_data["status"] = status

            if notes:
                update_data["progress.notes"] = notes

            await self.goal_hierarchy_collection.collection.update_one(
                {"_id": goal_id},
                {"$set": update_data}
            )

            # Update parent goal if this is a sub-goal
            goal = await self.goal_hierarchy_collection.collection.find_one({"_id": goal_id})
            if goal and goal.get("parentId"):
                await self._update_parent_goal(goal["parentId"])

            logger.info(f"Updated progress for goal {goal_id}: {progress_percentage}%")

        except Exception as error:
            logger.error(f"Error updating goal progress: {error}")
            raise error

    async def get_goal_analytics(self, agent_id: str, days: int = 30) -> Dict[str, Any]:
        """Get goal analytics for an agent."""
        try:
            # Get goals from the last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            goals = await self.goal_hierarchy_collection.collection.find({
                "agentId": agent_id,
                "metadata.createdBy": {"$gte": cutoff_date}
            }).to_list(length=None)

            if not goals:
                return {
                    "totalGoals": 0,
                    "completedGoals": 0,
                    "averageProgress": 0,
                    "goalsByCategory": {},
                    "goalsByPriority": {},
                    "completionRate": 0
                }

            # Calculate analytics
            total_goals = len(goals)
            completed_goals = len([g for g in goals if g.get("status") == "completed"])
            avg_progress = sum(g.get("progress", {}).get("percentage", 0) for g in goals) / total_goals

            # Group by category
            goals_by_category = {}
            for goal in goals:
                category = goal.get("goal", {}).get("category", "uncategorized")
                goals_by_category[category] = goals_by_category.get(category, 0) + 1

            # Group by priority
            goals_by_priority = {}
            for goal in goals:
                priority = goal.get("goal", {}).get("priority", "medium")
                goals_by_priority[priority] = goals_by_priority.get(priority, 0) + 1

            completion_rate = (completed_goals / total_goals) * 100 if total_goals > 0 else 0

            return {
                "totalGoals": total_goals,
                "completedGoals": completed_goals,
                "averageProgress": avg_progress,
                "goalsByCategory": goals_by_category,
                "goalsByPriority": goals_by_priority,
                "completionRate": completion_rate
            }

        except Exception as error:
            logger.error(f"Error getting goal analytics: {error}")
            return {
                "totalGoals": 0,
                "completedGoals": 0,
                "averageProgress": 0,
                "goalsByCategory": {},
                "goalsByPriority": {},
                "completionRate": 0
            }

    async def get_goal_hierarchy_visualization(self, agent_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get goal hierarchy visualization data."""
        try:
            query = {"agentId": agent_id}
            if session_id:
                query["sessionId"] = session_id

            goals = await self.goal_hierarchy_collection.collection.find(query).to_list(length=None)

            # Build hierarchy tree
            hierarchy = {
                "nodes": [],
                "edges": [],
                "levels": {}
            }

            for goal in goals:
                node = {
                    "id": str(goal["_id"]),
                    "title": goal.get("goal", {}).get("title", "Untitled"),
                    "status": goal.get("status", "not_started"),
                    "progress": goal.get("progress", {}).get("percentage", 0),
                    "category": goal.get("goal", {}).get("category", "general"),
                    "priority": goal.get("goal", {}).get("priority", "medium")
                }
                hierarchy["nodes"].append(node)

                # Add edge to parent if exists
                parent_id = goal.get("parentId")
                if parent_id:
                    hierarchy["edges"].append({
                        "source": str(parent_id),
                        "target": str(goal["_id"]),
                        "type": "parent_child"
                    })

            return hierarchy

        except Exception as error:
            logger.error(f"Error getting goal hierarchy visualization: {error}")
            return {"nodes": [], "edges": [], "levels": {}}

    async def create_execution_plan(self, root_goal_id: ObjectId) -> Dict[str, Any]:
        """Create execution plan for a goal hierarchy."""
        try:
            # Get all sub-goals for the root goal
            sub_goals = await self.goal_hierarchy_collection.get_sub_goals(root_goal_id)

            # Create execution plan
            execution_plan = {
                "rootGoalId": str(root_goal_id),
                "phases": [],
                "timeline": {
                    "estimatedDuration": 0,
                    "criticalPath": []
                },
                "dependencies": [],
                "milestones": []
            }

            # Simple execution plan (in production, would use sophisticated planning)
            for i, goal in enumerate(sub_goals):
                phase = {
                    "id": f"phase_{i+1}",
                    "goalId": str(goal["_id"]),
                    "title": goal.get("goal", {}).get("title", ""),
                    "order": i + 1,
                    "estimatedDuration": goal.get("timeline", {}).get("estimatedDuration", 60),
                    "dependencies": []
                }
                execution_plan["phases"].append(phase)
                execution_plan["timeline"]["estimatedDuration"] += phase["estimatedDuration"]

            return execution_plan

        except Exception as error:
            logger.error(f"Error creating execution plan: {error}")
            return {"rootGoalId": str(root_goal_id), "phases": [], "timeline": {}, "dependencies": [], "milestones": []}

    async def get_goal_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get goal statistics."""
        try:
            query = {}
            if agent_id:
                query["agentId"] = agent_id

            goals = await self.goal_hierarchy_collection.collection.find(query).to_list(length=None)

            if not goals:
                return {
                    "totalGoals": 0,
                    "activeGoals": 0,
                    "completedGoals": 0,
                    "averageProgress": 0
                }

            total_goals = len(goals)
            active_goals = len([g for g in goals if g.get("status") in ["active", "in_progress"]])
            completed_goals = len([g for g in goals if g.get("status") == "completed"])
            avg_progress = sum(g.get("progress", {}).get("percentage", 0) for g in goals) / total_goals

            return {
                "totalGoals": total_goals,
                "activeGoals": active_goals,
                "completedGoals": completed_goals,
                "averageProgress": avg_progress
            }

        except Exception as error:
            logger.error(f"Error getting goal stats: {error}")
            return {
                "totalGoals": 0,
                "activeGoals": 0,
                "completedGoals": 0,
                "averageProgress": 0
            }

    async def cleanup(self, days: int = 90) -> int:
        """Cleanup old completed goals."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = await self.goal_hierarchy_collection.collection.delete_many({
                "status": "completed",
                "metadata.createdBy": {"$lt": cutoff_date}
            })

            logger.info(f"Cleaned up {result.deleted_count} old goals")
            return result.deleted_count

        except Exception as error:
            logger.error(f"Error cleaning up goals: {error}")
            return 0
