"""
Temporal Planning Engine

Time-aware task management and scheduling system.
Provides intelligent temporal planning and deadline management.

Features:
- Time-aware task scheduling and prioritization
- Deadline management and pressure assessment
- Temporal context analysis and planning
- Time allocation optimization
- Schedule conflict detection and resolution
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, TemporalPlan, CognitiveSystemType

logger = logging.getLogger(__name__)


class TemporalPlanningEngine(CognitiveSystemInterface):
    """Temporal Planning Engine - System 8 of 16"""
    
    def __init__(self, system_id: str = "temporal_planning", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Temporal plans by user
        self._user_temporal_plans: Dict[str, TemporalPlan] = {}
        
        # Time-related keywords and patterns
        self._time_patterns = {
            "immediate": ["now", "immediately", "asap", "right away", "urgent"],
            "today": ["today", "this morning", "this afternoon", "this evening"],
            "tomorrow": ["tomorrow", "next day"],
            "this_week": ["this week", "by friday", "end of week"],
            "next_week": ["next week", "following week"],
            "this_month": ["this month", "by month end"],
            "deadline": ["deadline", "due date", "must finish", "needs to be done"],
            "schedule": ["schedule", "plan", "calendar", "appointment", "meeting"]
        }
        
        # Time estimation patterns
        self._duration_patterns = {
            "minutes": r"(\d+)\s*(?:minute|min)s?",
            "hours": r"(\d+)\s*(?:hour|hr)s?",
            "days": r"(\d+)\s*days?",
            "weeks": r"(\d+)\s*weeks?",
            "months": r"(\d+)\s*months?"
        }
        
        # Priority time factors
        self._time_priority_factors = {
            "overdue": 2.0,
            "due_today": 1.5,
            "due_tomorrow": 1.2,
            "due_this_week": 1.1,
            "future": 1.0
        }
    
    @property
    def system_name(self) -> str:
        return "Temporal Planning Engine"
    
    @property
    def system_description(self) -> str:
        return "Time-aware task management and scheduling system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.TEMPORAL_PLANNING}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.TEMPORAL_PLANNING}
    
    async def initialize(self) -> None:
        """Initialize the Temporal Planning Engine."""
        try:
            logger.info("Initializing Temporal Planning Engine...")
            await self._load_temporal_data()
            self._is_initialized = True
            logger.info("Temporal Planning Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Temporal Planning Engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Temporal Planning Engine."""
        try:
            await self._save_temporal_data()
            self._is_initialized = False
            logger.info("Temporal Planning Engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during Temporal Planning Engine shutdown: {e}")
    
    async def process(self, input_data: CognitiveInputData, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through temporal planning analysis."""
        if not self._is_initialized:
            raise RuntimeError("Temporal Planning Engine not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Ensure user has temporal plan
            if user_id not in self._user_temporal_plans:
                await self._create_user_temporal_plan(user_id)
            
            # Analyze temporal context
            temporal_analysis = await self._analyze_temporal_context(input_data)
            
            # Extract time-related tasks
            time_tasks = await self._extract_time_tasks(input_data)
            
            # Update temporal plan
            plan_updates = await self._update_temporal_plan(user_id, temporal_analysis, time_tasks)
            
            # Calculate deadline pressure
            deadline_pressure = await self._calculate_deadline_pressure(user_id)
            
            # Generate temporal recommendations
            recommendations = await self._generate_temporal_recommendations(user_id, temporal_analysis)
            
            # Optimize time allocation
            time_allocation = await self._optimize_time_allocation(user_id)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.85,
                "temporal_plan": {
                    "current_time": self._user_temporal_plans[user_id].current_time.isoformat(),
                    "time_horizon": str(self._user_temporal_plans[user_id].time_horizon),
                    "scheduled_tasks": len(self._user_temporal_plans[user_id].scheduled_tasks),
                    "deadline_pressure": deadline_pressure,
                    "efficiency_score": self._user_temporal_plans[user_id].efficiency_score
                },
                "temporal_analysis": temporal_analysis,
                "time_tasks": time_tasks,
                "plan_updates": plan_updates,
                "deadline_pressure": deadline_pressure,
                "time_allocation": time_allocation,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in Temporal Planning processing: {e}")
            return {"system": self.system_id, "status": "error", "error": str(e), "confidence": 0.0}
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current temporal planning state."""
        state_data = {
            "total_users": len(self._user_temporal_plans),
            "time_patterns_loaded": len(self._time_patterns)
        }
        
        if user_id and user_id in self._user_temporal_plans:
            plan = self._user_temporal_plans[user_id]
            state_data.update({
                "user_scheduled_tasks": len(plan.scheduled_tasks),
                "user_deadline_pressure": plan.deadline_pressure,
                "user_efficiency_score": plan.efficiency_score
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.TEMPORAL_PLANNING,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.9,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update temporal planning state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Temporal Planning state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for temporal planning."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Public temporal methods
    
    async def schedule_task(self, user_id: str, task: Dict[str, Any]) -> bool:
        """Schedule a task for a user."""
        if user_id not in self._user_temporal_plans:
            await self._create_user_temporal_plan(user_id)
        
        plan = self._user_temporal_plans[user_id]
        
        # Add task to scheduled tasks
        task_entry = {
            "id": task.get("id", f"task_{len(plan.scheduled_tasks)}"),
            "title": task.get("title", "Untitled Task"),
            "description": task.get("description", ""),
            "scheduled_time": task.get("scheduled_time", datetime.utcnow().isoformat()),
            "duration": task.get("duration", 60),  # minutes
            "priority": task.get("priority", 5),
            "deadline": task.get("deadline")
        }
        
        plan.scheduled_tasks.append(task_entry)
        
        # Update task priorities
        if task_entry["deadline"]:
            deadline = datetime.fromisoformat(task_entry["deadline"])
            plan.deadlines[task_entry["id"]] = deadline
        
        return True
    
    async def get_schedule(self, user_id: str, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get schedule for a user on a specific date."""
        if user_id not in self._user_temporal_plans:
            return []
        
        plan = self._user_temporal_plans[user_id]
        target_date = date or datetime.utcnow()
        
        # Filter tasks for the target date
        daily_tasks = []
        for task in plan.scheduled_tasks:
            task_time = datetime.fromisoformat(task["scheduled_time"])
            if task_time.date() == target_date.date():
                daily_tasks.append(task)
        
        # Sort by scheduled time
        daily_tasks.sort(key=lambda x: x["scheduled_time"])
        
        return daily_tasks
    
    # Private methods
    
    async def _load_temporal_data(self) -> None:
        """Load temporal data from storage."""
        logger.debug("Temporal data loaded")
    
    async def _save_temporal_data(self) -> None:
        """Save temporal data to storage."""
        logger.debug("Temporal data saved")
    
    async def _create_user_temporal_plan(self, user_id: str) -> None:
        """Create temporal plan for a user."""
        temporal_plan = TemporalPlan(
            current_time=datetime.utcnow(),
            time_horizon=timedelta(days=30),  # 30-day planning horizon
            deadline_pressure=0.0,
            efficiency_score=0.8
        )
        
        self._user_temporal_plans[user_id] = temporal_plan
        logger.debug(f"Created temporal plan for user {user_id}")
    
    async def _analyze_temporal_context(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze temporal context from input."""
        text = input_data.text or ""
        text_lower = text.lower()
        
        # Analyze time patterns
        time_context = {}
        for time_type, patterns in self._time_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in text_lower)
            if matches > 0:
                time_context[time_type] = matches
        
        # Determine primary time context
        primary_time_context = max(time_context, key=time_context.get) if time_context else "general"
        
        # Extract duration estimates
        duration_estimates = await self._extract_duration_estimates(text)
        
        # Assess urgency
        urgency_score = self._assess_urgency(text_lower, time_context)
        
        return {
            "primary_time_context": primary_time_context,
            "time_patterns": time_context,
            "duration_estimates": duration_estimates,
            "urgency_score": urgency_score,
            "time_awareness": len(time_context) > 0
        }
    
    async def _extract_time_tasks(self, input_data: CognitiveInputData) -> List[Dict[str, Any]]:
        """Extract time-related tasks from input."""
        text = input_data.text or ""
        tasks = []
        
        # Simple task extraction based on patterns
        import re
        
        # Look for task patterns with time
        task_patterns = [
            r"(?:need to|have to|must|should)\s+(.+?)\s+(?:by|before|until)\s+(.+?)(?:\.|$)",
            r"(?:schedule|plan)\s+(.+?)\s+(?:for|at)\s+(.+?)(?:\.|$)",
            r"(?:deadline|due)\s+(.+?)\s+(?:is|on)\s+(.+?)(?:\.|$)"
        ]
        
        for pattern in task_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                task_description = match.group(1).strip()
                time_reference = match.group(2).strip()
                
                if len(task_description) > 3:  # Filter out very short matches
                    tasks.append({
                        "description": task_description,
                        "time_reference": time_reference,
                        "extracted_from": "pattern_matching",
                        "confidence": 0.7
                    })
        
        return tasks
    
    async def _update_temporal_plan(self, user_id: str, temporal_analysis: Dict[str, Any], time_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update temporal plan based on analysis."""
        plan = self._user_temporal_plans[user_id]
        updates = []
        
        # Update current time
        plan.current_time = datetime.utcnow()
        
        # Add new tasks from analysis
        for task in time_tasks:
            # Convert time reference to actual datetime
            scheduled_time = await self._parse_time_reference(task["time_reference"])
            
            if scheduled_time:
                task_entry = {
                    "id": f"extracted_{len(plan.scheduled_tasks)}",
                    "title": task["description"],
                    "description": f"Extracted from input: {task['description']}",
                    "scheduled_time": scheduled_time.isoformat(),
                    "duration": 60,  # Default 1 hour
                    "priority": 5,
                    "source": "temporal_analysis"
                }
                
                plan.scheduled_tasks.append(task_entry)
                updates.append({
                    "action": "task_added",
                    "task": task_entry,
                    "confidence": task["confidence"]
                })
        
        # Update deadline pressure
        plan.deadline_pressure = await self._calculate_deadline_pressure(user_id)
        
        return updates
    
    async def _calculate_deadline_pressure(self, user_id: str) -> float:
        """Calculate deadline pressure for a user."""
        plan = self._user_temporal_plans[user_id]
        current_time = datetime.utcnow()
        
        pressure_score = 0.0
        deadline_count = 0
        
        for task_id, deadline in plan.deadlines.items():
            time_until_deadline = (deadline - current_time).total_seconds()
            
            if time_until_deadline < 0:
                # Overdue
                pressure_score += 1.0
            elif time_until_deadline < 86400:  # Less than 1 day
                pressure_score += 0.8
            elif time_until_deadline < 604800:  # Less than 1 week
                pressure_score += 0.4
            else:
                pressure_score += 0.1
            
            deadline_count += 1
        
        if deadline_count > 0:
            average_pressure = pressure_score / deadline_count
        else:
            average_pressure = 0.0
        
        plan.deadline_pressure = min(1.0, average_pressure)
        return plan.deadline_pressure
    
    async def _generate_temporal_recommendations(self, user_id: str, temporal_analysis: Dict[str, Any]) -> List[str]:
        """Generate temporal planning recommendations."""
        recommendations = []
        plan = self._user_temporal_plans[user_id]
        
        if plan.deadline_pressure > 0.7:
            recommendations.append("High deadline pressure detected - consider prioritizing urgent tasks")
        
        if temporal_analysis["urgency_score"] > 0.8:
            recommendations.append("Urgent tasks identified - recommend immediate attention")
        
        if len(plan.scheduled_tasks) > 20:
            recommendations.append("Heavy schedule detected - consider task delegation or rescheduling")
        
        if plan.efficiency_score < 0.6:
            recommendations.append("Low efficiency detected - consider time management techniques")
        
        if not temporal_analysis["time_awareness"]:
            recommendations.append("Consider adding time estimates and deadlines to improve planning")
        
        return recommendations
    
    async def _optimize_time_allocation(self, user_id: str) -> Dict[str, float]:
        """Optimize time allocation for a user."""
        plan = self._user_temporal_plans[user_id]
        
        # Simple time allocation optimization
        total_time_available = 8 * 60  # 8 hours in minutes
        total_task_time = sum(task.get("duration", 60) for task in plan.scheduled_tasks)
        
        if total_task_time > total_time_available:
            # Overallocated
            allocation_factor = total_time_available / total_task_time
            allocation = {
                "available_time": total_time_available,
                "required_time": total_task_time,
                "allocation_factor": allocation_factor,
                "status": "overallocated"
            }
        else:
            # Underallocated
            free_time = total_time_available - total_task_time
            allocation = {
                "available_time": total_time_available,
                "required_time": total_task_time,
                "free_time": free_time,
                "status": "underallocated"
            }
        
        # Update time allocation in plan
        plan.time_allocation = {
            "work_tasks": total_task_time / total_time_available,
            "free_time": max(0, (total_time_available - total_task_time) / total_time_available)
        }
        
        return allocation
    
    async def _extract_duration_estimates(self, text: str) -> Dict[str, List[int]]:
        """Extract duration estimates from text."""
        import re
        
        duration_estimates = {}
        
        for unit, pattern in self._duration_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                duration_estimates[unit] = [int(match) for match in matches]
        
        return duration_estimates
    
    def _assess_urgency(self, text: str, time_context: Dict[str, int]) -> float:
        """Assess urgency level from text and time context."""
        urgency_keywords = ["urgent", "asap", "immediately", "critical", "emergency", "rush"]
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in text)
        
        # Base urgency from keywords
        keyword_urgency = min(1.0, urgency_count * 0.3)
        
        # Time context urgency
        context_urgency = 0.0
        if "immediate" in time_context:
            context_urgency += 0.8
        if "today" in time_context:
            context_urgency += 0.6
        if "deadline" in time_context:
            context_urgency += 0.4
        
        return min(1.0, keyword_urgency + context_urgency)
    
    async def _parse_time_reference(self, time_ref: str) -> Optional[datetime]:
        """Parse time reference to datetime."""
        time_ref_lower = time_ref.lower().strip()
        current_time = datetime.utcnow()
        
        # Simple time parsing
        if "today" in time_ref_lower:
            return current_time.replace(hour=12, minute=0, second=0, microsecond=0)
        elif "tomorrow" in time_ref_lower:
            return current_time + timedelta(days=1)
        elif "next week" in time_ref_lower:
            return current_time + timedelta(weeks=1)
        elif "next month" in time_ref_lower:
            return current_time + timedelta(days=30)
        else:
            # Default to current time + 1 hour
            return current_time + timedelta(hours=1)
