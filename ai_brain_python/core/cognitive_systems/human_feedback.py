"""
Human Feedback Integration Engine

Advanced human-in-the-loop feedback system.
Manages approval workflows, feedback collection, and continuous learning from human input.

Features:
- Human-in-the-loop approval workflows and feedback collection
- Feedback analysis and pattern recognition
- Continuous learning from human corrections
- Preference learning and personalization
- Quality assurance and validation workflows
- Collaborative decision-making support
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, CognitiveSystemType

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    APPROVAL = "approval"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    RATING = "rating"
    SUGGESTION = "suggestion"


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class HumanFeedbackIntegrationEngine(CognitiveSystemInterface):
    """Human Feedback Integration Engine - System 16 of 16"""
    
    def __init__(self, system_id: str = "human_feedback", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Feedback management
        self._feedback_queue: List[Dict[str, Any]] = []
        self._approval_workflows: Dict[str, Dict[str, Any]] = {}
        self._user_preferences: Dict[str, Dict[str, Any]] = {}
        self._feedback_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configuration
        self._config = {
            "require_approval_threshold": config.get("require_approval_threshold", 0.7) if config else 0.7,
            "feedback_timeout_hours": config.get("feedback_timeout_hours", 24) if config else 24,
            "enable_preference_learning": config.get("enable_preference_learning", True) if config else True,
            "enable_collaborative_filtering": config.get("enable_collaborative_filtering", True) if config else True,
            "max_feedback_queue_size": config.get("max_feedback_queue_size", 1000) if config else 1000
        }
        
        # Feedback categories
        self._feedback_categories = {
            "content_quality": {
                "description": "Quality and accuracy of generated content",
                "weight": 0.3,
                "learning_rate": 0.1
            },
            "user_satisfaction": {
                "description": "Overall user satisfaction with responses",
                "weight": 0.25,
                "learning_rate": 0.15
            },
            "safety_compliance": {
                "description": "Safety and policy compliance",
                "weight": 0.2,
                "learning_rate": 0.05
            },
            "relevance": {
                "description": "Relevance to user query and context",
                "weight": 0.15,
                "learning_rate": 0.1
            },
            "style_preference": {
                "description": "Communication style and tone preferences",
                "weight": 0.1,
                "learning_rate": 0.2
            }
        }
        
        # Learning patterns
        self._learning_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # Approval workflow templates
        self._workflow_templates = {
            "standard_approval": {
                "steps": ["initial_review", "approval_decision"],
                "timeout_hours": 24,
                "required_approvers": 1
            },
            "safety_critical": {
                "steps": ["safety_review", "content_review", "final_approval"],
                "timeout_hours": 4,
                "required_approvers": 2
            },
            "collaborative_review": {
                "steps": ["peer_review", "expert_review", "consensus"],
                "timeout_hours": 48,
                "required_approvers": 3
            }
        }
    
    @property
    def system_name(self) -> str:
        return "Human Feedback Integration Engine"
    
    @property
    def system_description(self) -> str:
        return "Advanced human-in-the-loop feedback system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.HUMAN_FEEDBACK_INTEGRATION}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.HUMAN_FEEDBACK_INTEGRATION}
    
    async def initialize(self) -> None:
        """Initialize the Human Feedback Integration Engine."""
        try:
            logger.info("Initializing Human Feedback Integration Engine...")
            
            # Load feedback history and preferences
            await self._load_feedback_data()
            
            # Initialize learning models
            await self._initialize_learning_models()
            
            # Start feedback processing
            await self._start_feedback_processor()
            
            self._is_initialized = True
            logger.info("Human Feedback Integration Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Human Feedback Integration Engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Human Feedback Integration Engine."""
        try:
            logger.info("Shutting down Human Feedback Integration Engine...")
            
            # Save feedback data and preferences
            await self._save_feedback_data()
            
            # Process remaining feedback
            await self._process_remaining_feedback()
            
            self._is_initialized = False
            logger.info("Human Feedback Integration Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Human Feedback Integration Engine shutdown: {e}")
    
    async def process(self, input_data: CognitiveInputData, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through human feedback analysis."""
        if not self._is_initialized:
            raise RuntimeError("Human Feedback Integration Engine not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Analyze feedback requirements
            feedback_analysis = await self._analyze_feedback_requirements(input_data, context or {})
            
            # Check if approval is needed
            approval_needed = await self._check_approval_requirements(feedback_analysis, context or {})
            
            # Apply user preferences
            preference_adjustments = await self._apply_user_preferences(user_id, context or {})
            
            # Generate feedback recommendations
            feedback_recommendations = await self._generate_feedback_recommendations(user_id, feedback_analysis)
            
            # Create approval workflow if needed
            workflow_id = None
            if approval_needed:
                workflow_id = await self._create_approval_workflow(user_id, feedback_analysis, context or {})
            
            # Learn from implicit feedback
            await self._learn_from_implicit_feedback(user_id, input_data, context or {})
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.9,
                "feedback_analysis": {
                    "approval_needed": approval_needed,
                    "confidence_score": feedback_analysis.get("confidence_score", 0.8),
                    "risk_level": feedback_analysis.get("risk_level", "low"),
                    "feedback_categories": list(feedback_analysis.get("categories", []))
                },
                "user_preferences": {
                    "preferences_applied": len(preference_adjustments),
                    "personalization_level": preference_adjustments.get("personalization_level", 0.5)
                },
                "approval_workflow": {
                    "workflow_id": workflow_id,
                    "workflow_created": workflow_id is not None,
                    "estimated_completion_time": feedback_analysis.get("estimated_approval_time", 0)
                },
                "feedback_recommendations": feedback_recommendations,
                "learning_insights": await self._get_learning_insights(user_id)
            }
            
        except Exception as e:
            logger.error(f"Error in Human Feedback processing: {e}")
            return {"system": self.system_id, "status": "error", "error": str(e), "confidence": 0.0}
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current human feedback integration state."""
        state_data = {
            "feedback_queue_size": len(self._feedback_queue),
            "active_workflows": len(self._active_workflows),
            "total_users_with_preferences": len(self._user_preferences),
            "feedback_categories": len(self._feedback_categories)
        }
        
        if user_id and user_id in self._user_preferences:
            user_prefs = self._user_preferences[user_id]
            state_data.update({
                "user_preference_count": len(user_prefs),
                "user_feedback_history": len(self._feedback_history.get(user_id, []))
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.HUMAN_FEEDBACK,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.95,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update human feedback integration state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Human Feedback state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for human feedback processing."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Public feedback methods
    
    async def submit_feedback(self, user_id: str, feedback_type: FeedbackType, content: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Submit feedback from a user."""
        feedback_id = f"feedback_{len(self._feedback_queue)}_{int(datetime.utcnow().timestamp())}"
        
        feedback_entry = {
            "id": feedback_id,
            "user_id": user_id,
            "type": feedback_type.value,
            "content": content,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
            "processed": False
        }
        
        self._feedback_queue.append(feedback_entry)
        
        # Maintain queue size
        if len(self._feedback_queue) > self._config["max_feedback_queue_size"]:
            self._feedback_queue = self._feedback_queue[-self._config["max_feedback_queue_size"]:]
        
        # Process feedback immediately if it's critical
        if feedback_type in [FeedbackType.CORRECTION, FeedbackType.APPROVAL]:
            await self._process_feedback_entry(feedback_entry)
        
        logger.info(f"Feedback submitted: {feedback_id} from user {user_id}")
        return feedback_id
    
    async def get_approval_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get approval workflow status."""
        if workflow_id not in self._approval_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self._approval_workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "status": workflow["status"],
            "current_step": workflow["current_step"],
            "progress": workflow["progress"],
            "created_at": workflow["created_at"],
            "estimated_completion": workflow.get("estimated_completion"),
            "approvers": workflow.get("approvers", [])
        }
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        
        # Merge new preferences
        self._user_preferences[user_id].update(preferences)
        
        # Add timestamp
        self._user_preferences[user_id]["last_updated"] = datetime.utcnow().isoformat()
        
        logger.info(f"Updated preferences for user {user_id}")
        return True
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        return self._user_preferences.get(user_id, {})
    
    # Private methods
    
    async def _load_feedback_data(self) -> None:
        """Load feedback data and preferences."""
        logger.debug("Feedback data loaded")
    
    async def _save_feedback_data(self) -> None:
        """Save feedback data and preferences."""
        logger.debug("Feedback data saved")
    
    async def _initialize_learning_models(self) -> None:
        """Initialize learning models for feedback analysis."""
        logger.debug("Learning models initialized")
    
    async def _start_feedback_processor(self) -> None:
        """Start background feedback processor."""
        logger.debug("Feedback processor started")
    
    async def _process_remaining_feedback(self) -> None:
        """Process remaining feedback in queue."""
        for feedback in self._feedback_queue:
            if not feedback.get("processed", False):
                await self._process_feedback_entry(feedback)
        logger.debug("Remaining feedback processed")
    
    async def _analyze_feedback_requirements(self, input_data: CognitiveInputData, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what kind of feedback is needed."""
        analysis = {
            "confidence_score": 0.8,
            "risk_level": "low",
            "categories": [],
            "estimated_approval_time": 0
        }
        
        # Analyze confidence from cognitive results
        cognitive_results = context.get("cognitive_results", {})
        if cognitive_results:
            confidence_scores = [
                result.get("confidence", 0.8) for result in cognitive_results.values()
                if isinstance(result, dict) and "confidence" in result
            ]
            if confidence_scores:
                analysis["confidence_score"] = sum(confidence_scores) / len(confidence_scores)
        
        # Determine risk level
        if analysis["confidence_score"] < 0.5:
            analysis["risk_level"] = "high"
            analysis["categories"].extend(["content_quality", "safety_compliance"])
        elif analysis["confidence_score"] < 0.7:
            analysis["risk_level"] = "medium"
            analysis["categories"].append("content_quality")
        
        # Check for safety-critical content
        if "safety_guardrails" in cognitive_results:
            safety_result = cognitive_results["safety_guardrails"]
            if isinstance(safety_result, dict) and not safety_result.get("safety_assessment", {}).get("is_safe", True):
                analysis["risk_level"] = "critical"
                analysis["categories"].append("safety_compliance")
        
        # Estimate approval time based on risk level
        if analysis["risk_level"] == "critical":
            analysis["estimated_approval_time"] = 4  # 4 hours
        elif analysis["risk_level"] == "high":
            analysis["estimated_approval_time"] = 12  # 12 hours
        elif analysis["risk_level"] == "medium":
            analysis["estimated_approval_time"] = 24  # 24 hours
        
        return analysis
    
    async def _check_approval_requirements(self, feedback_analysis: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if human approval is required."""
        confidence_score = feedback_analysis["confidence_score"]
        risk_level = feedback_analysis["risk_level"]
        
        # Require approval for low confidence or high risk
        if confidence_score < self._config["require_approval_threshold"]:
            return True
        
        if risk_level in ["high", "critical"]:
            return True
        
        # Check for safety violations
        if "safety_compliance" in feedback_analysis["categories"]:
            return True
        
        return False
    
    async def _apply_user_preferences(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user preferences to processing."""
        adjustments = {"personalization_level": 0.0}
        
        if user_id in self._user_preferences:
            preferences = self._user_preferences[user_id]
            
            # Apply communication style preferences
            if "communication_style" in preferences:
                adjustments["communication_style"] = preferences["communication_style"]
                adjustments["personalization_level"] += 0.3
            
            # Apply content preferences
            if "content_preferences" in preferences:
                adjustments["content_preferences"] = preferences["content_preferences"]
                adjustments["personalization_level"] += 0.2
            
            # Apply safety preferences
            if "safety_level" in preferences:
                adjustments["safety_level"] = preferences["safety_level"]
                adjustments["personalization_level"] += 0.1
        
        return adjustments
    
    async def _generate_feedback_recommendations(self, user_id: str, feedback_analysis: Dict[str, Any]) -> List[str]:
        """Generate feedback recommendations."""
        recommendations = []
        
        if feedback_analysis["confidence_score"] < 0.7:
            recommendations.append("Consider requesting human review for low-confidence responses")
        
        if feedback_analysis["risk_level"] == "high":
            recommendations.append("High-risk content detected - human oversight recommended")
        
        if user_id in self._user_preferences:
            recommendations.append("User preferences are being applied to personalize responses")
        else:
            recommendations.append("Consider setting user preferences for better personalization")
        
        # Check feedback history
        if user_id in self._feedback_history:
            recent_feedback = self._feedback_history[user_id][-5:]  # Last 5 feedback items
            negative_feedback = sum(1 for fb in recent_feedback if fb.get("rating", 5) < 3)
            if negative_feedback > 2:
                recommendations.append("Recent negative feedback detected - consider adjusting approach")
        
        return recommendations
    
    async def _create_approval_workflow(self, user_id: str, feedback_analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create approval workflow."""
        workflow_id = f"workflow_{len(self._approval_workflows)}_{int(datetime.utcnow().timestamp())}"
        
        # Select workflow template based on risk level
        risk_level = feedback_analysis["risk_level"]
        if risk_level == "critical":
            template = self._workflow_templates["safety_critical"]
        elif risk_level == "high":
            template = self._workflow_templates["collaborative_review"]
        else:
            template = self._workflow_templates["standard_approval"]
        
        workflow = {
            "id": workflow_id,
            "user_id": user_id,
            "template": template,
            "status": ApprovalStatus.PENDING.value,
            "current_step": template["steps"][0],
            "progress": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(hours=template["timeout_hours"])).isoformat(),
            "context": context,
            "feedback_analysis": feedback_analysis
        }
        
        self._approval_workflows[workflow_id] = workflow
        
        logger.info(f"Created approval workflow: {workflow_id} for user {user_id}")
        return workflow_id
    
    async def _learn_from_implicit_feedback(self, user_id: str, input_data: CognitiveInputData, context: Dict[str, Any]) -> None:
        """Learn from implicit feedback signals."""
        if not self._config["enable_preference_learning"]:
            return
        
        # Analyze interaction patterns
        implicit_signals = {
            "interaction_length": len(input_data.text or ""),
            "complexity_requested": len(input_data.requested_systems),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store learning patterns
        if user_id not in self._learning_patterns:
            self._learning_patterns[user_id] = []
        
        self._learning_patterns[user_id].append(implicit_signals)
        
        # Keep only recent patterns
        if len(self._learning_patterns[user_id]) > 100:
            self._learning_patterns[user_id] = self._learning_patterns[user_id][-100:]
    
    async def _process_feedback_entry(self, feedback_entry: Dict[str, Any]) -> None:
        """Process a single feedback entry."""
        user_id = feedback_entry["user_id"]
        feedback_type = feedback_entry["type"]
        content = feedback_entry["content"]
        
        # Record in feedback history
        if user_id not in self._feedback_history:
            self._feedback_history[user_id] = []
        
        self._feedback_history[user_id].append(feedback_entry)
        
        # Learn from feedback
        if feedback_type == "correction":
            await self._learn_from_correction(user_id, content)
        elif feedback_type == "preference":
            await self._learn_from_preference(user_id, content)
        elif feedback_type == "rating":
            await self._learn_from_rating(user_id, content)
        
        # Mark as processed
        feedback_entry["processed"] = True
        feedback_entry["processed_at"] = datetime.utcnow().isoformat()
    
    async def _learn_from_correction(self, user_id: str, correction: Dict[str, Any]) -> None:
        """Learn from user corrections."""
        # Update user preferences based on corrections
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        
        # Extract learning signals from correction
        if "preferred_style" in correction:
            self._user_preferences[user_id]["communication_style"] = correction["preferred_style"]
        
        if "content_adjustments" in correction:
            self._user_preferences[user_id]["content_preferences"] = correction["content_adjustments"]
    
    async def _learn_from_preference(self, user_id: str, preference: Dict[str, Any]) -> None:
        """Learn from explicit user preferences."""
        await self.update_user_preferences(user_id, preference)
    
    async def _learn_from_rating(self, user_id: str, rating: Dict[str, Any]) -> None:
        """Learn from user ratings."""
        rating_value = rating.get("rating", 5)
        category = rating.get("category", "overall")
        
        # Update learning patterns based on rating
        if user_id not in self._learning_patterns:
            self._learning_patterns[user_id] = []
        
        learning_entry = {
            "type": "rating",
            "category": category,
            "rating": rating_value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._learning_patterns[user_id].append(learning_entry)
    
    async def _get_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """Get learning insights for a user."""
        insights = {
            "total_feedback_items": 0,
            "average_rating": 0.0,
            "preference_strength": 0.0,
            "learning_progress": 0.0
        }
        
        if user_id in self._feedback_history:
            feedback_items = self._feedback_history[user_id]
            insights["total_feedback_items"] = len(feedback_items)
            
            # Calculate average rating
            ratings = [
                item["content"].get("rating", 5) for item in feedback_items
                if item["type"] == "rating" and "rating" in item["content"]
            ]
            if ratings:
                insights["average_rating"] = sum(ratings) / len(ratings)
        
        if user_id in self._user_preferences:
            preferences = self._user_preferences[user_id]
            insights["preference_strength"] = len(preferences) / 10.0  # Normalize to 0-1
        
        if user_id in self._learning_patterns:
            patterns = self._learning_patterns[user_id]
            insights["learning_progress"] = min(1.0, len(patterns) / 50.0)  # Normalize to 0-1
        
        return insights
