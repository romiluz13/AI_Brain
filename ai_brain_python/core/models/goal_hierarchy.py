"""
Goal Hierarchy Models for AI Brain Python

Models for hierarchical goal planning and achievement tracking:
- Goal definition and relationships
- Priority management
- Progress tracking
- Achievement metrics
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, validator, model_validator, root_validator

from ai_brain_python.core.models.base_models import BaseAIBrainModel


class GoalStatus(str, Enum):
    """Goal status enumeration."""
    PENDING = "pending"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class GoalPriority(str, Enum):
    """Goal priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"


class GoalType(str, Enum):
    """Types of goals."""
    IMMEDIATE = "immediate"      # Short-term, tactical goals
    SHORT_TERM = "short_term"    # Goals within days/weeks
    MEDIUM_TERM = "medium_term"  # Goals within months
    LONG_TERM = "long_term"      # Goals within years
    STRATEGIC = "strategic"      # High-level strategic goals
    MAINTENANCE = "maintenance"  # Ongoing maintenance goals


class GoalRelationshipType(str, Enum):
    """Types of relationships between goals."""
    PARENT_CHILD = "parent_child"
    DEPENDENCY = "dependency"
    CONFLICT = "conflict"
    SYNERGY = "synergy"
    ALTERNATIVE = "alternative"
    SEQUENCE = "sequence"


class Goal(BaseAIBrainModel):
    """Individual goal model."""
    
    # Basic goal information
    title: str = Field(description="Goal title")
    description: str = Field(description="Detailed goal description")
    goal_type: GoalType = Field(description="Type of goal")
    
    # Status and priority
    status: GoalStatus = Field(default=GoalStatus.PENDING, description="Current goal status")
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM, description="Goal priority")
    
    # Ownership and context
    user_id: Optional[str] = Field(default=None, description="User who owns this goal")
    context: Dict[str, Any] = Field(default_factory=dict, description="Goal context information")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Goal creation time")
    target_date: Optional[datetime] = Field(default=None, description="Target completion date")
    started_at: Optional[datetime] = Field(default=None, description="Goal start time")
    completed_at: Optional[datetime] = Field(default=None, description="Goal completion time")
    
    # Progress tracking
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Goal milestones")
    completed_milestones: Set[str] = Field(default_factory=set, description="Completed milestone IDs")
    
    # Success criteria
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Acceptance criteria")
    
    # Resources and constraints
    required_resources: List[str] = Field(default_factory=list, description="Required resources")
    constraints: List[str] = Field(default_factory=list, description="Goal constraints")
    estimated_effort: Optional[float] = Field(default=None, ge=0.0, description="Estimated effort in hours")
    
    # Relationships
    parent_goal_id: Optional[str] = Field(default=None, description="Parent goal ID")
    child_goal_ids: Set[str] = Field(default_factory=set, description="Child goal IDs")
    dependency_goal_ids: Set[str] = Field(default_factory=set, description="Dependency goal IDs")
    
    # Performance metrics
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Goal importance score")
    urgency_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Goal urgency score")
    feasibility_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Goal feasibility score")
    
    # Learning and adaptation
    lessons_learned: List[str] = Field(default_factory=list, description="Lessons learned")
    adaptation_notes: List[str] = Field(default_factory=list, description="Goal adaptation notes")
    
    @validator('progress_percentage')
    def validate_progress(cls, v, values):
        """Validate progress percentage consistency with status."""
        status = values.get('status')
        if status == GoalStatus.COMPLETED and v < 100.0:
            raise ValueError("Completed goals must have 100% progress")
        if status == GoalStatus.PENDING and v > 0.0:
            raise ValueError("Pending goals should have 0% progress")
        return v
    
    @validator('completed_at')
    def validate_completion_time(cls, v, values):
        """Validate completion time consistency with status."""
        status = values.get('status')
        if status == GoalStatus.COMPLETED and v is None:
            raise ValueError("Completed goals must have completion time")
        if status != GoalStatus.COMPLETED and v is not None:
            raise ValueError("Only completed goals should have completion time")
        return v
    
    @model_validator(mode='after')
    def validate_goal_consistency(self):
        """Validate overall goal consistency."""
        if self.target_date and self.created_at and self.target_date <= self.created_at:
            raise ValueError("Target date must be after creation date")

        return self
    
    def calculate_priority_score(self) -> float:
        """Calculate numerical priority score."""
        priority_scores = {
            GoalPriority.CRITICAL: 1.0,
            GoalPriority.HIGH: 0.8,
            GoalPriority.MEDIUM: 0.6,
            GoalPriority.LOW: 0.4,
            GoalPriority.DEFERRED: 0.2,
        }
        return priority_scores.get(self.priority, 0.6)
    
    def is_overdue(self) -> bool:
        """Check if goal is overdue."""
        if not self.target_date:
            return False
        return (
            self.status not in [GoalStatus.COMPLETED, GoalStatus.CANCELLED] and
            datetime.utcnow() > self.target_date
        )
    
    def time_remaining(self) -> Optional[timedelta]:
        """Calculate time remaining until target date."""
        if not self.target_date:
            return None
        return self.target_date - datetime.utcnow()
    
    def add_milestone(self, milestone_id: str, title: str, description: str, target_date: Optional[datetime] = None) -> None:
        """Add a milestone to the goal."""
        milestone = {
            "id": milestone_id,
            "title": title,
            "description": description,
            "target_date": target_date.isoformat() if target_date else None,
            "created_at": datetime.utcnow().isoformat(),
            "completed": False
        }
        self.milestones.append(milestone)
    
    def complete_milestone(self, milestone_id: str) -> bool:
        """Mark a milestone as completed."""
        for milestone in self.milestones:
            if milestone["id"] == milestone_id:
                milestone["completed"] = True
                milestone["completed_at"] = datetime.utcnow().isoformat()
                self.completed_milestones.add(milestone_id)
                
                # Update progress based on completed milestones
                if self.milestones:
                    completed_count = len(self.completed_milestones)
                    total_count = len(self.milestones)
                    self.progress_percentage = (completed_count / total_count) * 100.0
                
                return True
        return False


class GoalRelationship(BaseAIBrainModel):
    """Relationship between goals."""
    
    source_goal_id: str = Field(description="Source goal ID")
    target_goal_id: str = Field(description="Target goal ID")
    relationship_type: GoalRelationshipType = Field(description="Type of relationship")
    
    # Relationship metadata
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength")
    description: Optional[str] = Field(default=None, description="Relationship description")
    
    # Constraints and conditions
    conditions: List[str] = Field(default_factory=list, description="Conditions for relationship")
    is_active: bool = Field(default=True, description="Whether relationship is active")
    
    @validator('source_goal_id', 'target_goal_id')
    def validate_goal_ids(cls, v):
        """Validate goal IDs are not empty."""
        if not v or not v.strip():
            raise ValueError("Goal IDs cannot be empty")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_different_goals(cls, values):
        """Validate source and target are different goals."""
        source = values.get('source_goal_id')
        target = values.get('target_goal_id')

        if source == target:
            raise ValueError("Source and target goals must be different")

        return values


class GoalHierarchy(BaseAIBrainModel):
    """Goal hierarchy management model."""
    
    # Hierarchy identification
    user_id: Optional[str] = Field(default=None, description="User who owns this hierarchy")
    hierarchy_name: str = Field(default="Default", description="Name of the goal hierarchy")
    
    # Goals and relationships
    goals: Dict[str, Goal] = Field(default_factory=dict, description="Goals in the hierarchy")
    relationships: List[GoalRelationship] = Field(default_factory=list, description="Goal relationships")
    
    # Hierarchy structure
    root_goal_ids: Set[str] = Field(default_factory=set, description="Root goal IDs")
    active_goal_ids: Set[str] = Field(default_factory=set, description="Currently active goal IDs")
    
    # Priority management
    priority_queue: List[str] = Field(default_factory=list, description="Priority-ordered goal IDs")
    focus_goal_id: Optional[str] = Field(default=None, description="Currently focused goal ID")
    
    # Performance metrics
    total_goals: int = Field(default=0, ge=0, description="Total number of goals")
    completed_goals: int = Field(default=0, ge=0, description="Number of completed goals")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Goal completion success rate")
    
    # Time management
    average_completion_time: Optional[float] = Field(default=None, ge=0.0, description="Average goal completion time in days")
    overdue_goals: int = Field(default=0, ge=0, description="Number of overdue goals")
    
    def add_goal(self, goal: Goal) -> str:
        """Add a goal to the hierarchy."""
        goal_id = goal.id
        self.goals[goal_id] = goal
        self.total_goals = len(self.goals)
        
        # Update root goals if no parent
        if not goal.parent_goal_id:
            self.root_goal_ids.add(goal_id)
        
        # Update active goals if status is active
        if goal.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]:
            self.active_goal_ids.add(goal_id)
        
        self._update_priority_queue()
        return goal_id
    
    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal from the hierarchy."""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        
        # Remove from child goals of parent
        if goal.parent_goal_id and goal.parent_goal_id in self.goals:
            parent = self.goals[goal.parent_goal_id]
            parent.child_goal_ids.discard(goal_id)
        
        # Update children to remove parent reference
        for child_id in goal.child_goal_ids:
            if child_id in self.goals:
                self.goals[child_id].parent_goal_id = None
                self.root_goal_ids.add(child_id)
        
        # Remove from sets
        self.root_goal_ids.discard(goal_id)
        self.active_goal_ids.discard(goal_id)
        
        # Remove relationships
        self.relationships = [
            rel for rel in self.relationships 
            if rel.source_goal_id != goal_id and rel.target_goal_id != goal_id
        ]
        
        # Remove goal
        del self.goals[goal_id]
        self.total_goals = len(self.goals)
        
        self._update_priority_queue()
        return True
    
    def update_goal_status(self, goal_id: str, new_status: GoalStatus) -> bool:
        """Update goal status and related metrics."""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        old_status = goal.status
        goal.status = new_status
        
        # Update active goals
        if new_status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]:
            self.active_goal_ids.add(goal_id)
        else:
            self.active_goal_ids.discard(goal_id)
        
        # Update completion metrics
        if new_status == GoalStatus.COMPLETED and old_status != GoalStatus.COMPLETED:
            self.completed_goals += 1
            goal.completed_at = datetime.utcnow()
            goal.progress_percentage = 100.0
        elif old_status == GoalStatus.COMPLETED and new_status != GoalStatus.COMPLETED:
            self.completed_goals = max(0, self.completed_goals - 1)
            goal.completed_at = None
        
        # Update success rate
        if self.total_goals > 0:
            self.success_rate = self.completed_goals / self.total_goals
        
        self._update_priority_queue()
        return True
    
    def add_relationship(self, relationship: GoalRelationship) -> bool:
        """Add a relationship between goals."""
        # Validate goals exist
        if (relationship.source_goal_id not in self.goals or 
            relationship.target_goal_id not in self.goals):
            return False
        
        # Check for duplicate relationships
        for existing in self.relationships:
            if (existing.source_goal_id == relationship.source_goal_id and
                existing.target_goal_id == relationship.target_goal_id and
                existing.relationship_type == relationship.relationship_type):
                return False
        
        self.relationships.append(relationship)
        
        # Update goal references for parent-child relationships
        if relationship.relationship_type == GoalRelationshipType.PARENT_CHILD:
            parent = self.goals[relationship.source_goal_id]
            child = self.goals[relationship.target_goal_id]
            
            parent.child_goal_ids.add(relationship.target_goal_id)
            child.parent_goal_id = relationship.source_goal_id
            
            # Remove child from root goals
            self.root_goal_ids.discard(relationship.target_goal_id)
        
        return True
    
    def get_goal_dependencies(self, goal_id: str) -> List[str]:
        """Get all dependencies for a goal."""
        dependencies = []
        for rel in self.relationships:
            if (rel.target_goal_id == goal_id and 
                rel.relationship_type == GoalRelationshipType.DEPENDENCY):
                dependencies.append(rel.source_goal_id)
        return dependencies
    
    def get_goal_children(self, goal_id: str) -> List[str]:
        """Get all child goals for a goal."""
        if goal_id not in self.goals:
            return []
        return list(self.goals[goal_id].child_goal_ids)
    
    def get_goal_path(self, goal_id: str) -> List[str]:
        """Get the path from root to goal."""
        if goal_id not in self.goals:
            return []
        
        path = []
        current_id = goal_id
        
        while current_id:
            path.insert(0, current_id)
            goal = self.goals[current_id]
            current_id = goal.parent_goal_id
        
        return path
    
    def _update_priority_queue(self) -> None:
        """Update the priority queue based on current goals."""
        active_goals = [
            (goal_id, self.goals[goal_id]) 
            for goal_id in self.active_goal_ids 
            if goal_id in self.goals
        ]
        
        # Sort by priority score (descending) and urgency (descending)
        active_goals.sort(
            key=lambda x: (
                x[1].calculate_priority_score(),
                x[1].urgency_score,
                -x[1].progress_percentage  # Prefer goals with less progress
            ),
            reverse=True
        )
        
        self.priority_queue = [goal_id for goal_id, _ in active_goals]
    
    def get_next_goal(self) -> Optional[str]:
        """Get the next goal to focus on."""
        if not self.priority_queue:
            self._update_priority_queue()
        
        return self.priority_queue[0] if self.priority_queue else None
    
    def get_overdue_goals(self) -> List[str]:
        """Get list of overdue goal IDs."""
        overdue = []
        for goal_id, goal in self.goals.items():
            if goal.is_overdue():
                overdue.append(goal_id)
        
        self.overdue_goals = len(overdue)
        return overdue
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy statistics."""
        stats = {
            "total_goals": self.total_goals,
            "completed_goals": self.completed_goals,
            "active_goals": len(self.active_goal_ids),
            "success_rate": self.success_rate,
            "overdue_goals": len(self.get_overdue_goals()),
        }
        
        # Goal distribution by status
        status_counts = {}
        for goal in self.goals.values():
            status = goal.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        stats["status_distribution"] = status_counts
        
        # Goal distribution by priority
        priority_counts = {}
        for goal in self.goals.values():
            priority = goal.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        stats["priority_distribution"] = priority_counts
        
        # Average progress
        if self.goals:
            total_progress = sum(goal.progress_percentage for goal in self.goals.values())
            stats["average_progress"] = total_progress / len(self.goals)
        else:
            stats["average_progress"] = 0.0
        
        return stats
