"""
Attention Management System

Dynamic attention allocation and focus control system.
Manages cognitive resources and optimizes attention distribution across tasks.

Features:
- Dynamic attention allocation across multiple tasks
- Focus level monitoring and optimization
- Distraction detection and management
- Cognitive load assessment and balancing
- Attention span tracking and prediction
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, AttentionState, AttentionType, CognitiveSystemType

logger = logging.getLogger(__name__)


class AttentionManagementSystem(CognitiveSystemInterface):
    """
    Attention Management System - System 4 of 16
    
    Manages dynamic attention allocation and focus control
    with intelligent resource optimization.
    """
    
    def __init__(self, system_id: str = "attention_management", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Attention states by user
        self._user_attention_states: Dict[str, AttentionState] = {}
        
        # Attention tracking
        self._attention_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Focus indicators in text
        self._focus_indicators = {
            "high_focus": [
                "concentrate", "focus", "attention", "carefully", "precisely",
                "detailed", "thorough", "specific", "exact", "important"
            ],
            "low_focus": [
                "distracted", "confused", "unclear", "maybe", "perhaps",
                "not sure", "kind of", "sort of", "whatever", "anyway"
            ],
            "task_switching": [
                "also", "additionally", "meanwhile", "by the way", "speaking of",
                "another thing", "switching to", "moving on", "next"
            ]
        }
        
        # Cognitive load indicators
        self._cognitive_load_indicators = {
            "high_load": [
                "overwhelmed", "too much", "complex", "complicated", "difficult",
                "struggling", "hard to understand", "confusing", "overloaded"
            ],
            "low_load": [
                "simple", "easy", "straightforward", "clear", "obvious",
                "basic", "elementary", "trivial", "no problem"
            ]
        }
        
        # Attention allocation strategies
        self._allocation_strategies = {
            "focused": {"primary_task": 0.8, "secondary_tasks": 0.2},
            "divided": {"primary_task": 0.5, "secondary_tasks": 0.5},
            "selective": {"primary_task": 0.9, "secondary_tasks": 0.1},
            "sustained": {"primary_task": 0.7, "monitoring": 0.3},
            "alternating": {"task_a": 0.5, "task_b": 0.5}
        }
    
    @property
    def system_name(self) -> str:
        return "Attention Management System"
    
    @property
    def system_description(self) -> str:
        return "Dynamic attention allocation and focus control system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.ATTENTION_ALLOCATION}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.ATTENTION_ALLOCATION}
    
    async def initialize(self) -> None:
        """Initialize the Attention Management System."""
        try:
            logger.info("Initializing Attention Management System...")
            
            # Load attention states and history
            await self._load_attention_data()
            
            # Initialize attention models
            await self._initialize_attention_models()
            
            self._is_initialized = True
            logger.info("Attention Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Attention Management System: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Attention Management System."""
        try:
            logger.info("Shutting down Attention Management System...")
            
            # Save attention data
            await self._save_attention_data()
            
            self._is_initialized = False
            logger.info("Attention Management System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Attention Management System shutdown: {e}")
    
    async def process(
        self, 
        input_data: CognitiveInputData,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input through attention management analysis."""
        if not self._is_initialized:
            raise RuntimeError("Attention Management System not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Ensure user has attention state
            if user_id not in self._user_attention_states:
                await self._create_user_attention_state(user_id)
            
            # Analyze attention requirements
            attention_analysis = await self._analyze_attention_requirements(input_data)
            
            # Update attention allocation
            attention_allocation = await self._update_attention_allocation(
                user_id, attention_analysis, context or {}
            )
            
            # Detect distractions
            distraction_analysis = await self._analyze_distractions(input_data)
            
            # Calculate cognitive load
            cognitive_load = await self._calculate_cognitive_load(input_data, context or {})
            
            # Generate attention recommendations
            recommendations = await self._generate_attention_recommendations(
                user_id, attention_analysis, cognitive_load
            )
            
            # Update attention state
            attention_state = await self._update_attention_state(
                user_id, attention_analysis, attention_allocation, cognitive_load
            )
            
            # Calculate processing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.85,
                "attention_state": {
                    "attention_type": attention_state.attention_type.value,
                    "focus_level": attention_state.focus_level,
                    "attention_span": attention_state.attention_span,
                    "cognitive_load": attention_state.cognitive_load,
                    "distraction_level": attention_state.distraction_level
                },
                "attention_allocation": attention_allocation,
                "distraction_analysis": distraction_analysis,
                "cognitive_load": cognitive_load,
                "recommendations": recommendations,
                "attention_metrics": await self._get_attention_metrics(user_id)
            }
            
            # Record attention data for analysis
            await self._record_attention_data(user_id, attention_analysis, cognitive_load)
            
            logger.debug(f"Attention Management processing completed for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in Attention Management processing: {e}")
            return {
                "system": self.system_id,
                "status": "error",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current attention management state."""
        state_data = {
            "total_users_tracked": len(self._user_attention_states),
            "focus_indicators_loaded": len(self._focus_indicators),
            "allocation_strategies": len(self._allocation_strategies)
        }
        
        if user_id and user_id in self._user_attention_states:
            attention_state = self._user_attention_states[user_id]
            state_data.update({
                "user_attention_type": attention_state.attention_type.value,
                "user_focus_level": attention_state.focus_level,
                "user_cognitive_load": attention_state.cognitive_load,
                "user_distraction_level": attention_state.distraction_level
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.ATTENTION_MANAGEMENT,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.9,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update attention management state."""
        try:
            if "attention_state" in state.state_data and user_id:
                # Update user attention state
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating Attention Management state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for attention management."""
        violations = []
        warnings = []
        
        # Attention management can work with any input
        if not input_data.text and not input_data.context.user_id:
            warnings.append("Limited context for attention analysis")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            confidence=1.0 if len(violations) == 0 else 0.0,
            violations=violations,
            warnings=warnings
        )
    
    # Public attention methods
    
    async def allocate_attention(self, user_id: str, tasks: Dict[str, float]) -> Dict[str, float]:
        """Allocate attention across multiple tasks."""
        if user_id not in self._user_attention_states:
            await self._create_user_attention_state(user_id)
        
        # Normalize allocation to sum to 1.0
        total_weight = sum(tasks.values())
        if total_weight > 0:
            normalized_allocation = {task: weight / total_weight for task, weight in tasks.items()}
        else:
            normalized_allocation = {task: 1.0 / len(tasks) for task in tasks}
        
        # Update attention state
        attention_state = self._user_attention_states[user_id]
        attention_state.attention_allocation = normalized_allocation
        
        return normalized_allocation
    
    async def get_focus_recommendations(self, user_id: str) -> List[str]:
        """Get focus improvement recommendations for a user."""
        if user_id not in self._user_attention_states:
            return ["No attention data available for recommendations"]
        
        attention_state = self._user_attention_states[user_id]
        recommendations = []
        
        if attention_state.focus_level < 0.5:
            recommendations.extend([
                "Consider taking a short break to refresh focus",
                "Try eliminating distractions from your environment",
                "Use the Pomodoro technique for better focus management"
            ])
        
        if attention_state.cognitive_load > 0.8:
            recommendations.extend([
                "Break down complex tasks into smaller, manageable parts",
                "Consider delegating or postponing non-essential tasks",
                "Take regular breaks to prevent cognitive overload"
            ])
        
        if attention_state.distraction_level > 0.6:
            recommendations.extend([
                "Identify and minimize sources of distraction",
                "Use focus-enhancing techniques like mindfulness",
                "Create a dedicated workspace for important tasks"
            ])
        
        return recommendations
    
    # Private methods
    
    async def _load_attention_data(self) -> None:
        """Load attention data from storage."""
        # In production, this would load from MongoDB
        logger.debug("Attention data loaded")
    
    async def _save_attention_data(self) -> None:
        """Save attention data to storage."""
        # In production, this would save to MongoDB
        logger.debug("Attention data saved")
    
    async def _initialize_attention_models(self) -> None:
        """Initialize attention analysis models."""
        # In production, this would load ML models for attention analysis
        logger.debug("Attention models initialized")
    
    async def _create_user_attention_state(self, user_id: str) -> None:
        """Create initial attention state for a user."""
        attention_state = AttentionState(
            attention_type=AttentionType.FOCUSED,
            focus_level=0.7,
            attention_span=1800.0,  # 30 minutes default
            cognitive_load=0.5,
            distraction_level=0.3
        )
        
        self._user_attention_states[user_id] = attention_state
        logger.debug(f"Created attention state for user {user_id}")
    
    async def _analyze_attention_requirements(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze attention requirements from input."""
        text = input_data.text or ""
        text_lower = text.lower()
        
        # Analyze focus indicators
        focus_score = 0.5  # Default
        for indicator in self._focus_indicators["high_focus"]:
            if indicator in text_lower:
                focus_score += 0.1
        
        for indicator in self._focus_indicators["low_focus"]:
            if indicator in text_lower:
                focus_score -= 0.1
        
        focus_score = max(0.0, min(1.0, focus_score))
        
        # Detect task switching
        task_switching = any(indicator in text_lower for indicator in self._focus_indicators["task_switching"])
        
        # Determine attention type needed
        if task_switching:
            attention_type = AttentionType.ALTERNATING
        elif focus_score > 0.8:
            attention_type = AttentionType.FOCUSED
        elif len(input_data.requested_systems) > 3:
            attention_type = AttentionType.DIVIDED
        else:
            attention_type = AttentionType.SELECTIVE
        
        # Estimate required attention span
        text_length = len(text)
        if text_length > 1000:
            required_span = 1800  # 30 minutes for long text
        elif text_length > 500:
            required_span = 900   # 15 minutes for medium text
        else:
            required_span = 300   # 5 minutes for short text
        
        return {
            "focus_score": focus_score,
            "attention_type": attention_type,
            "task_switching_detected": task_switching,
            "required_attention_span": required_span,
            "complexity_level": self._assess_complexity(text)
        }
    
    async def _update_attention_allocation(
        self, 
        user_id: str, 
        attention_analysis: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Update attention allocation based on analysis."""
        attention_type = attention_analysis["attention_type"]
        
        # Get base allocation strategy
        if attention_type.value in self._allocation_strategies:
            base_allocation = self._allocation_strategies[attention_type.value].copy()
        else:
            base_allocation = {"primary_task": 0.7, "secondary_tasks": 0.3}
        
        # Adjust based on cognitive systems engaged
        cognitive_results = context.get("cognitive_results", {})
        if cognitive_results:
            system_count = len(cognitive_results)
            if system_count > 5:
                # Many systems engaged, need divided attention
                base_allocation = {"primary_task": 0.4, "secondary_tasks": 0.6}
            elif system_count < 3:
                # Few systems, can focus more
                base_allocation = {"primary_task": 0.8, "secondary_tasks": 0.2}
        
        return base_allocation
    
    async def _analyze_distractions(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze potential distractions in input."""
        text = input_data.text or ""
        text_lower = text.lower()
        
        distraction_indicators = [
            "distracted", "interrupted", "can't focus", "lost track",
            "forgot what", "where was i", "anyway", "off topic"
        ]
        
        distraction_count = sum(1 for indicator in distraction_indicators if indicator in text_lower)
        distraction_level = min(1.0, distraction_count * 0.3)
        
        # Identify distraction sources
        distraction_sources = []
        if "notification" in text_lower or "phone" in text_lower:
            distraction_sources.append("notifications")
        if "noise" in text_lower or "loud" in text_lower:
            distraction_sources.append("environmental_noise")
        if "tired" in text_lower or "sleepy" in text_lower:
            distraction_sources.append("fatigue")
        
        return {
            "distraction_level": distraction_level,
            "distraction_sources": distraction_sources,
            "distraction_indicators_found": distraction_count
        }
    
    async def _calculate_cognitive_load(self, input_data: CognitiveInputData, context: Dict[str, Any]) -> float:
        """Calculate current cognitive load."""
        text = input_data.text or ""
        text_lower = text.lower()
        
        # Base load from text complexity
        complexity_load = self._assess_complexity(text) * 0.3
        
        # Load from cognitive systems engaged
        cognitive_results = context.get("cognitive_results", {})
        system_load = min(0.5, len(cognitive_results) * 0.05)
        
        # Load indicators in text
        load_adjustment = 0.0
        for indicator in self._cognitive_load_indicators["high_load"]:
            if indicator in text_lower:
                load_adjustment += 0.1
        
        for indicator in self._cognitive_load_indicators["low_load"]:
            if indicator in text_lower:
                load_adjustment -= 0.1
        
        total_load = complexity_load + system_load + load_adjustment
        return max(0.0, min(1.0, total_load))
    
    async def _generate_attention_recommendations(
        self, 
        user_id: str, 
        attention_analysis: Dict[str, Any], 
        cognitive_load: float
    ) -> List[str]:
        """Generate attention management recommendations."""
        recommendations = []
        
        if attention_analysis["focus_score"] < 0.5:
            recommendations.append("Consider improving focus through mindfulness or concentration exercises")
        
        if attention_analysis["task_switching_detected"]:
            recommendations.append("Task switching detected - consider completing one task before moving to another")
        
        if cognitive_load > 0.8:
            recommendations.append("High cognitive load detected - consider breaking tasks into smaller parts")
        
        if attention_analysis["complexity_level"] > 0.7:
            recommendations.append("Complex task detected - allocate sufficient time and minimize distractions")
        
        # Historical recommendations
        if user_id in self._attention_history:
            history = self._attention_history[user_id]
            if len(history) > 5:
                recent_loads = [entry["cognitive_load"] for entry in history[-5:]]
                if statistics.mean(recent_loads) > 0.7:
                    recommendations.append("Consistently high cognitive load - consider workload management")
        
        return recommendations
    
    async def _update_attention_state(
        self, 
        user_id: str, 
        attention_analysis: Dict[str, Any], 
        attention_allocation: Dict[str, float], 
        cognitive_load: float
    ) -> AttentionState:
        """Update the attention state for a user."""
        attention_state = self._user_attention_states[user_id]
        
        # Update state properties
        attention_state.attention_type = attention_analysis["attention_type"]
        attention_state.focus_level = attention_analysis["focus_score"]
        attention_state.attention_span = attention_analysis["required_attention_span"]
        attention_state.cognitive_load = cognitive_load
        attention_state.attention_allocation = attention_allocation
        
        # Update distraction level (moving average)
        current_distraction = attention_analysis.get("distraction_level", 0.3)
        attention_state.distraction_level = (attention_state.distraction_level * 0.7 + current_distraction * 0.3)
        
        return attention_state
    
    async def _get_attention_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get attention metrics for a user."""
        if user_id not in self._attention_history:
            return {"sessions": 0, "average_focus": 0.5, "average_load": 0.5}
        
        history = self._attention_history[user_id]
        
        if not history:
            return {"sessions": 0, "average_focus": 0.5, "average_load": 0.5}
        
        focus_levels = [entry["focus_score"] for entry in history]
        cognitive_loads = [entry["cognitive_load"] for entry in history]
        
        return {
            "sessions": len(history),
            "average_focus": statistics.mean(focus_levels),
            "average_load": statistics.mean(cognitive_loads),
            "focus_trend": self._calculate_trend(focus_levels),
            "load_trend": self._calculate_trend(cognitive_loads)
        }
    
    async def _record_attention_data(self, user_id: str, attention_analysis: Dict[str, Any], cognitive_load: float) -> None:
        """Record attention data for analysis."""
        if user_id not in self._attention_history:
            self._attention_history[user_id] = []
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "focus_score": attention_analysis["focus_score"],
            "cognitive_load": cognitive_load,
            "attention_type": attention_analysis["attention_type"].value,
            "complexity_level": attention_analysis["complexity_level"]
        }
        
        self._attention_history[user_id].append(entry)
        
        # Keep only recent history
        if len(self._attention_history[user_id]) > 100:
            self._attention_history[user_id] = self._attention_history[user_id][-100:]
    
    def _assess_complexity(self, text: str) -> float:
        """Assess text complexity for cognitive load calculation."""
        if not text:
            return 0.0
        
        # Simple complexity metrics
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        if sentence_count == 0:
            sentence_count = 1
        
        avg_words_per_sentence = word_count / sentence_count
        
        # Complexity based on sentence length and vocabulary
        complexity = 0.0
        
        if avg_words_per_sentence > 20:
            complexity += 0.3
        elif avg_words_per_sentence > 15:
            complexity += 0.2
        elif avg_words_per_sentence > 10:
            complexity += 0.1
        
        # Add complexity for technical terms (simplified)
        technical_indicators = ["algorithm", "implementation", "configuration", "optimization", "analysis"]
        technical_count = sum(1 for term in technical_indicators if term in text.lower())
        complexity += min(0.4, technical_count * 0.1)
        
        return min(1.0, complexity)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 3:
            return "insufficient_data"
        
        recent_avg = statistics.mean(values[-3:])
        earlier_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"
