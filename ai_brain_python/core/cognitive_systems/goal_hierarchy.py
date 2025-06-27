"""
Goal Hierarchy Manager

Hierarchical goal planning, tracking, and achievement system.
Manages complex goal relationships and provides intelligent prioritization.

Features:
- Hierarchical goal decomposition and management
- Dynamic priority adjustment based on context
- Goal relationship tracking (dependencies, conflicts, synergies)
- Progress monitoring and milestone tracking
- Intelligent goal recommendation and planning
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, CognitiveSystemType
from ai_brain_python.core.models.goal_hierarchy import (
    Goal, GoalHierarchy, GoalStatus, GoalPriority, GoalType, GoalRelationship, GoalRelationshipType
)

logger = logging.getLogger(__name__)


class GoalHierarchyManager(CognitiveSystemInterface):
    """
    Goal Hierarchy Manager - System 2 of 16
    
    Manages hierarchical goal planning, tracking, and achievement
    with intelligent prioritization and relationship management.
    """
    
    def __init__(self, system_id: str = "goal_hierarchy", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Goal hierarchies by user
        self._user_hierarchies: Dict[str, GoalHierarchy] = {}
        
        # Goal extraction patterns
        self._goal_patterns = [
            r'\b(?:i want to|i need to|i plan to|i will|i should|i must|goal is to|objective is to)\s+(.+)',
            r'\b(?:my goal|my objective|my plan|my intention)\s+(?:is to|is)\s+(.+)',
            r'\b(?:planning to|hoping to|aiming to|trying to|working to)\s+(.+)',
            r'\b(?:complete|finish|achieve|accomplish|reach|attain)\s+(.+)',
        ]
        
        # Priority keywords
        self._priority_keywords = {
            GoalPriority.CRITICAL: ['urgent', 'critical', 'emergency', 'asap', 'immediately'],
            GoalPriority.HIGH: ['important', 'high priority', 'soon', 'quickly', 'priority'],
            GoalPriority.MEDIUM: ['moderate', 'normal', 'regular', 'standard'],
            GoalPriority.LOW: ['low priority', 'eventually', 'someday', 'when possible'],
            GoalPriority.DEFERRED: ['later', 'postpone', 'defer', 'maybe', 'if time']
        }
        
        # Time horizon keywords
        self._time_horizon_keywords = {
            GoalType.IMMEDIATE: ['now', 'today', 'immediately', 'right now'],
            GoalType.SHORT_TERM: ['this week', 'next week', 'soon', 'quickly'],
            GoalType.MEDIUM_TERM: ['this month', 'next month', 'in a few weeks'],
            GoalType.LONG_TERM: ['this year', 'next year', 'long term', 'eventually'],
            GoalType.STRATEGIC: ['strategic', 'vision', 'mission', 'long-term vision']
        }
    
    @property
    def system_name(self) -> str:
        return "Goal Hierarchy Manager"
    
    @property
    def system_description(self) -> str:
        return "Hierarchical goal planning, tracking, and achievement system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.GOAL_PLANNING}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {
            SystemCapability.GOAL_PLANNING,
            SystemCapability.GOAL_TRACKING
        }
    
    async def initialize(self) -> None:
        """Initialize the Goal Hierarchy Manager."""
        try:
            logger.info("Initializing Goal Hierarchy Manager...")
            
            # Load user goal hierarchies
            await self._load_goal_hierarchies()
            
            # Initialize goal analysis models
            await self._initialize_goal_models()
            
            self._is_initialized = True
            logger.info("Goal Hierarchy Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Goal Hierarchy Manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Goal Hierarchy Manager."""
        try:
            logger.info("Shutting down Goal Hierarchy Manager...")
            
            # Save goal hierarchies
            await self._save_goal_hierarchies()
            
            self._is_initialized = False
            logger.info("Goal Hierarchy Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Goal Hierarchy Manager shutdown: {e}")
    
    async def process(
        self, 
        input_data: CognitiveInputData,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input through goal hierarchy analysis."""
        if not self._is_initialized:
            raise RuntimeError("Goal Hierarchy Manager not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Ensure user has a goal hierarchy
            if user_id not in self._user_hierarchies:
                await self._create_user_hierarchy(user_id)
            
            hierarchy = self._user_hierarchies[user_id]
            
            # Extract goals from input
            extracted_goals = await self._extract_goals_from_text(input_data.text or "")
            
            # Update existing goals based on context
            goal_updates = await self._analyze_goal_updates(input_data, hierarchy)
            
            # Get current goal recommendations
            recommendations = await self._generate_goal_recommendations(hierarchy, input_data)
            
            # Get next priority goal
            next_goal = hierarchy.get_next_goal()
            next_goal_info = None
            if next_goal and next_goal in hierarchy.goals:
                goal = hierarchy.goals[next_goal]
                next_goal_info = {
                    "id": goal.id,
                    "title": goal.title,
                    "priority": goal.priority.value,
                    "progress": goal.progress_percentage,
                    "due_date": goal.target_date.isoformat() if goal.target_date else None
                }
            
            # Calculate processing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.85,
                "goal_hierarchy": {
                    "total_goals": hierarchy.total_goals,
                    "active_goals": len(hierarchy.active_goal_ids),
                    "completed_goals": hierarchy.completed_goals,
                    "success_rate": hierarchy.success_rate,
                    "overdue_goals": len(hierarchy.get_overdue_goals())
                },
                "extracted_goals": extracted_goals,
                "goal_updates": goal_updates,
                "next_priority_goal": next_goal_info,
                "recommendations": recommendations,
                "hierarchy_statistics": hierarchy.get_hierarchy_statistics()
            }
            
            logger.debug(f"Goal Hierarchy processing completed for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in Goal Hierarchy processing: {e}")
            return {
                "system": self.system_id,
                "status": "error",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current goal hierarchy state."""
        state_data = {
            "total_users": len(self._user_hierarchies),
            "goal_patterns_loaded": len(self._goal_patterns)
        }
        
        if user_id and user_id in self._user_hierarchies:
            hierarchy = self._user_hierarchies[user_id]
            state_data.update({
                "user_total_goals": hierarchy.total_goals,
                "user_active_goals": len(hierarchy.active_goal_ids),
                "user_completed_goals": hierarchy.completed_goals,
                "user_success_rate": hierarchy.success_rate
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.GOAL_HIERARCHY,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.9,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update goal hierarchy state."""
        try:
            # Update internal state based on provided state
            if "goal_hierarchy" in state.state_data and user_id:
                # This would update the user's goal hierarchy
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating Goal Hierarchy state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for goal hierarchy processing."""
        violations = []
        warnings = []
        
        # Check if we have text to analyze
        if not input_data.text:
            warnings.append("No text provided for goal analysis")
        
        # Check for user context
        if not input_data.context.user_id:
            warnings.append("No user ID provided - using anonymous goal tracking")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            confidence=1.0 if len(violations) == 0 else 0.0,
            violations=violations,
            warnings=warnings
        )
    
    # Public goal management methods
    
    async def create_goal(
        self, 
        user_id: str, 
        title: str, 
        description: str,
        goal_type: GoalType = GoalType.SHORT_TERM,
        priority: GoalPriority = GoalPriority.MEDIUM,
        target_date: Optional[datetime] = None
    ) -> str:
        """Create a new goal for a user."""
        if user_id not in self._user_hierarchies:
            await self._create_user_hierarchy(user_id)
        
        goal = Goal(
            title=title,
            description=description,
            goal_type=goal_type,
            priority=priority,
            target_date=target_date,
            user_id=user_id
        )
        
        hierarchy = self._user_hierarchies[user_id]
        goal_id = hierarchy.add_goal(goal)
        
        logger.info(f"Created goal '{title}' for user {user_id}")
        return goal_id
    
    async def update_goal_progress(self, user_id: str, goal_id: str, progress: float) -> bool:
        """Update progress for a specific goal."""
        if user_id not in self._user_hierarchies:
            return False
        
        hierarchy = self._user_hierarchies[user_id]
        if goal_id not in hierarchy.goals:
            return False
        
        goal = hierarchy.goals[goal_id]
        goal.progress_percentage = max(0.0, min(100.0, progress))
        
        # Auto-complete if progress reaches 100%
        if progress >= 100.0 and goal.status != GoalStatus.COMPLETED:
            hierarchy.update_goal_status(goal_id, GoalStatus.COMPLETED)
        
        return True
    
    async def get_user_goals(self, user_id: str, status: Optional[GoalStatus] = None) -> List[Dict[str, Any]]:
        """Get goals for a user, optionally filtered by status."""
        if user_id not in self._user_hierarchies:
            return []
        
        hierarchy = self._user_hierarchies[user_id]
        goals = []
        
        for goal_id, goal in hierarchy.goals.items():
            if status is None or goal.status == status:
                goals.append({
                    "id": goal.id,
                    "title": goal.title,
                    "description": goal.description,
                    "status": goal.status.value,
                    "priority": goal.priority.value,
                    "progress": goal.progress_percentage,
                    "created_at": goal.created_at.isoformat(),
                    "target_date": goal.target_date.isoformat() if goal.target_date else None,
                    "is_overdue": goal.is_overdue()
                })
        
        return goals
    
    # Private methods
    
    async def _load_goal_hierarchies(self) -> None:
        """Load user goal hierarchies from storage."""
        # In production, this would load from MongoDB
        logger.debug("Goal hierarchies loaded")
    
    async def _save_goal_hierarchies(self) -> None:
        """Save user goal hierarchies to storage."""
        # In production, this would save to MongoDB
        logger.debug("Goal hierarchies saved")
    
    async def _initialize_goal_models(self) -> None:
        """Initialize goal analysis models."""
        # In production, this would load ML models for goal extraction
        logger.debug("Goal analysis models initialized")
    
    async def _create_user_hierarchy(self, user_id: str) -> None:
        """Create a new goal hierarchy for a user."""
        hierarchy = GoalHierarchy(
            user_id=user_id,
            hierarchy_name=f"{user_id}_goals"
        )
        self._user_hierarchies[user_id] = hierarchy
        logger.debug(f"Created goal hierarchy for user {user_id}")
    
    async def _extract_goals_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract potential goals from text input."""
        if not text:
            return []
        
        import re
        extracted_goals = []
        text_lower = text.lower()
        
        for pattern in self._goal_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                goal_text = match.group(1).strip()
                if len(goal_text) > 5:  # Filter out very short matches
                    
                    # Determine priority
                    priority = self._determine_priority(text_lower)
                    
                    # Determine time horizon
                    goal_type = self._determine_goal_type(text_lower)
                    
                    extracted_goals.append({
                        "text": goal_text,
                        "priority": priority.value,
                        "type": goal_type.value,
                        "confidence": 0.7
                    })
        
        return extracted_goals
    
    async def _analyze_goal_updates(self, input_data: CognitiveInputData, hierarchy: GoalHierarchy) -> List[Dict[str, Any]]:
        """Analyze input for potential goal updates."""
        updates = []
        text = input_data.text or ""
        text_lower = text.lower()
        
        # Look for completion indicators
        completion_patterns = [
            r'\b(?:completed|finished|done|accomplished|achieved)\s+(.+)',
            r'\b(.+)\s+(?:is complete|is done|is finished)',
            r'\b(?:i completed|i finished|i did)\s+(.+)'
        ]
        
        import re
        for pattern in completion_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                goal_text = match.group(1).strip()
                
                # Try to match with existing goals
                matching_goals = self._find_matching_goals(goal_text, hierarchy)
                for goal_id in matching_goals:
                    updates.append({
                        "goal_id": goal_id,
                        "update_type": "completion",
                        "confidence": 0.8
                    })
        
        return updates
    
    async def _generate_goal_recommendations(self, hierarchy: GoalHierarchy, input_data: CognitiveInputData) -> List[str]:
        """Generate goal recommendations based on current state."""
        recommendations = []
        
        # Check for overdue goals
        overdue_goals = hierarchy.get_overdue_goals()
        if overdue_goals:
            recommendations.append(f"You have {len(overdue_goals)} overdue goals that need attention")
        
        # Check for stalled goals (no progress in a while)
        stalled_goals = 0
        for goal in hierarchy.goals.values():
            if (goal.status == GoalStatus.IN_PROGRESS and 
                goal.progress_percentage < 50 and
                (datetime.utcnow() - goal.created_at).days > 7):
                stalled_goals += 1
        
        if stalled_goals > 0:
            recommendations.append(f"Consider breaking down {stalled_goals} stalled goals into smaller milestones")
        
        # Suggest goal creation if user has few goals
        if hierarchy.total_goals < 3:
            recommendations.append("Consider setting more specific goals to improve focus and productivity")
        
        # Suggest priority review if too many high-priority goals
        high_priority_count = sum(
            1 for goal in hierarchy.goals.values() 
            if goal.priority in [GoalPriority.CRITICAL, GoalPriority.HIGH] and goal.status == GoalStatus.ACTIVE
        )
        
        if high_priority_count > 5:
            recommendations.append("Consider reviewing goal priorities - too many high-priority goals can reduce focus")
        
        return recommendations
    
    def _determine_priority(self, text: str) -> GoalPriority:
        """Determine goal priority from text."""
        for priority, keywords in self._priority_keywords.items():
            if any(keyword in text for keyword in keywords):
                return priority
        return GoalPriority.MEDIUM
    
    def _determine_goal_type(self, text: str) -> GoalType:
        """Determine goal type from text."""
        for goal_type, keywords in self._time_horizon_keywords.items():
            if any(keyword in text for keyword in keywords):
                return goal_type
        return GoalType.SHORT_TERM
    
    def _find_matching_goals(self, goal_text: str, hierarchy: GoalHierarchy) -> List[str]:
        """Find goals that match the given text."""
        matching_goals = []
        goal_text_lower = goal_text.lower()
        
        for goal_id, goal in hierarchy.goals.items():
            # Simple text matching - in production would use semantic similarity
            if (goal_text_lower in goal.title.lower() or 
                goal_text_lower in goal.description.lower() or
                goal.title.lower() in goal_text_lower):
                matching_goals.append(goal_id)
        
        return matching_goals
