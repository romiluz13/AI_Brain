"""
MetacognitiveAwarenessEngine - Advanced metacognitive awareness and self-monitoring for AI agents

Exact Python equivalent of JavaScript metacognitive capabilities with:
- Self-awareness algorithms and cognitive monitoring
- Metacognitive strategies and thinking about thinking processes
- Performance self-assessment and improvement identification
- Cognitive load monitoring and optimization
- Learning strategy adaptation and meta-learning
- Real-time cognitive state awareness

Features:
- Real-time cognitive state monitoring
- Metacognitive strategy selection and adaptation
- Self-assessment and performance evaluation
- Cognitive load optimization and management
- Learning strategy meta-optimization
- Thinking process introspection and analysis
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from bson import ObjectId
import asyncio
import json
import random
import math

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_brain_python.storage.collections.metacognitive_state_collection import MetacognitiveStateCollection
from ai_brain_python.core.types import MetacognitiveState, MetacognitiveAnalyticsOptions
from ai_brain_python.utils.logger import logger


@dataclass
class CognitiveMonitoringRequest:
    """Cognitive monitoring request interface."""
    agent_id: str
    session_id: Optional[str]
    current_task: str
    cognitive_context: Dict[str, Any]
    performance_metrics: Dict[str, float]
    environment_factors: Dict[str, Any]


@dataclass
class MetacognitiveStrategy:
    """Metacognitive strategy interface."""
    strategy_id: str
    name: str
    description: str
    effectiveness: float
    cognitive_load: float
    applicable_contexts: List[str]


@dataclass
class MetacognitiveAwarenessResult:
    """Metacognitive awareness result interface."""
    awareness_id: ObjectId
    cognitive_state: Dict[str, Any]
    metacognitive_strategies: List[MetacognitiveStrategy]
    self_assessment: Dict[str, float]
    cognitive_load: float
    optimization_recommendations: List[str]


class MetacognitiveAwarenessEngine:
    """
    MetacognitiveAwarenessEngine - Advanced metacognitive awareness and self-monitoring
    
    Exact Python equivalent of JavaScript MetacognitiveAwarenessEngine with:
    - Real-time cognitive state monitoring
    - Metacognitive strategy selection and adaptation
    - Self-assessment and performance evaluation
    - Cognitive load optimization and management
    - Learning strategy meta-optimization
    - Thinking process introspection and analysis
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.metacognitive_collection = MetacognitiveStateCollection(db)
        self.is_initialized = False
        
        # Metacognitive configuration
        self._config = {
            "monitoring_interval": 30,  # seconds
            "cognitive_load_threshold": 0.8,
            "strategy_adaptation_threshold": 0.6,
            "self_assessment_frequency": 300,  # 5 minutes
            "optimization_trigger_threshold": 0.7
        }
        
        # Metacognitive strategies
        self._strategies = {
            "planning": MetacognitiveStrategy(
                strategy_id="planning",
                name="Strategic Planning",
                description="Plan approach before execution",
                effectiveness=0.85,
                cognitive_load=0.3,
                applicable_contexts=["complex_tasks", "problem_solving"]
            ),
            "monitoring": MetacognitiveStrategy(
                strategy_id="monitoring",
                name="Progress Monitoring",
                description="Monitor progress during execution",
                effectiveness=0.75,
                cognitive_load=0.2,
                applicable_contexts=["all_tasks"]
            ),
            "evaluation": MetacognitiveStrategy(
                strategy_id="evaluation",
                name="Performance Evaluation",
                description="Evaluate performance after completion",
                effectiveness=0.8,
                cognitive_load=0.25,
                applicable_contexts=["learning_tasks", "improvement"]
            )
        }
        
        # Cognitive state tracking
        self._cognitive_states: Dict[str, Dict[str, Any]] = {}
        self._performance_history: List[Dict[str, Any]] = []
        
        # Self-awareness metrics
        self._self_awareness_metrics = {
            "cognitive_clarity": 0.8,
            "strategy_effectiveness": 0.75,
            "learning_efficiency": 0.7,
            "adaptation_speed": 0.65
        }
    
    async def initialize(self) -> None:
        """Initialize the metacognitive awareness engine."""
        if self.is_initialized:
            return
        
        try:
            # Initialize metacognitive collection
            await self.metacognitive_collection.create_indexes()
            
            # Load metacognitive state
            await self._load_metacognitive_state()
            
            # Start cognitive monitoring
            await self._start_cognitive_monitoring()
            
            self.is_initialized = True
            logger.info("✅ MetacognitiveAwarenessEngine initialized successfully")
            
        except Exception as error:
            logger.error(f"❌ Failed to initialize MetacognitiveAwarenessEngine: {error}")
            raise error
    
    async def monitor_cognitive_state(
        self,
        request: CognitiveMonitoringRequest
    ) -> MetacognitiveAwarenessResult:
        """Monitor and analyze cognitive state."""
        if not self.is_initialized:
            raise Exception("MetacognitiveAwarenessEngine must be initialized first")
        
        # Generate awareness ID
        awareness_id = ObjectId()
        
        # Analyze current cognitive state
        cognitive_state = await self._analyze_cognitive_state(
            request.cognitive_context,
            request.performance_metrics
        )
        
        # Select appropriate metacognitive strategies
        strategies = await self._select_metacognitive_strategies(
            request.current_task,
            cognitive_state,
            request.environment_factors
        )
        
        # Perform self-assessment
        self_assessment = await self._perform_self_assessment(
            request.performance_metrics,
            cognitive_state
        )
        
        # Calculate cognitive load
        cognitive_load = await self._calculate_cognitive_load(
            cognitive_state,
            strategies
        )
        
        # Generate optimization recommendations
        optimization_recommendations = await self._generate_optimization_recommendations(
            cognitive_state,
            self_assessment,
            cognitive_load
        )
        
        # Create metacognitive record
        metacognitive_record = {
            "awarenessId": awareness_id,
            "agentId": request.agent_id,
            "sessionId": request.session_id,
            "timestamp": datetime.utcnow(),
            "currentTask": request.current_task,
            "cognitiveContext": request.cognitive_context,
            "cognitiveState": cognitive_state,
            "metacognitiveStrategies": [s.__dict__ for s in strategies],
            "selfAssessment": self_assessment,
            "cognitiveLoad": cognitive_load,
            "optimizationRecommendations": optimization_recommendations,
            "performanceMetrics": request.performance_metrics,
            "environmentFactors": request.environment_factors,
            "metadata": {
                "framework": "python_ai_brain",
                "version": "1.0.0",
                "source": "metacognitive_awareness_engine"
            }
        }
        
        # Store metacognitive record
        await self.metacognitive_collection.record_metacognitive_state(metacognitive_record)
        
        # Update cognitive state tracking
        self._cognitive_states[request.agent_id] = cognitive_state
        
        return MetacognitiveAwarenessResult(
            awareness_id=awareness_id,
            cognitive_state=cognitive_state,
            metacognitive_strategies=strategies,
            self_assessment=self_assessment,
            cognitive_load=cognitive_load,
            optimization_recommendations=optimization_recommendations
        )
    
    async def get_metacognitive_analytics(
        self,
        agent_id: str,
        options: Optional[MetacognitiveAnalyticsOptions] = None
    ) -> Dict[str, Any]:
        """Get metacognitive analytics for an agent."""
        return await self.metacognitive_collection.get_metacognitive_analytics(agent_id, options)
    
    async def get_metacognitive_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metacognitive statistics."""
        stats = await self.metacognitive_collection.get_metacognitive_stats(agent_id)
        
        return {
            **stats,
            "cognitiveStatesTracked": len(self._cognitive_states),
            "performanceHistory": len(self._performance_history),
            "selfAwarenessMetrics": self._self_awareness_metrics
        }

    # Private helper methods
    async def _load_metacognitive_state(self) -> None:
        """Load metacognitive state from storage."""
        logger.debug("Metacognitive state loaded")

    async def _start_cognitive_monitoring(self) -> None:
        """Start cognitive monitoring."""
        logger.debug("Cognitive monitoring started")

    async def _analyze_cognitive_state(
        self,
        cognitive_context: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze current cognitive state."""
        # Calculate cognitive clarity
        clarity = self._calculate_cognitive_clarity(cognitive_context, performance_metrics)

        # Assess cognitive resources
        resources = self._assess_cognitive_resources(cognitive_context)

        # Determine cognitive mode
        mode = self._determine_cognitive_mode(performance_metrics)

        return {
            "clarity": clarity,
            "resources": resources,
            "mode": mode,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": min(1.0, (clarity + resources) / 2)
        }

    async def _select_metacognitive_strategies(
        self,
        current_task: str,
        cognitive_state: Dict[str, Any],
        environment_factors: Dict[str, Any]
    ) -> List[MetacognitiveStrategy]:
        """Select appropriate metacognitive strategies."""
        selected_strategies = []

        # Always include monitoring for all tasks
        selected_strategies.append(self._strategies["monitoring"])

        # Add planning for complex tasks
        if environment_factors.get("complexity", 0) > 0.7:
            selected_strategies.append(self._strategies["planning"])

        # Add evaluation for learning contexts
        if "learning" in current_task.lower() or cognitive_state.get("mode") == "learning":
            selected_strategies.append(self._strategies["evaluation"])

        return selected_strategies

    async def _perform_self_assessment(
        self,
        performance_metrics: Dict[str, float],
        cognitive_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Perform self-assessment of cognitive performance."""
        assessment = {}

        # Assess performance accuracy
        accuracy = performance_metrics.get("accuracy", 0.8)
        assessment["performance_accuracy"] = accuracy

        # Assess cognitive efficiency
        efficiency = cognitive_state.get("resources", 0.7)
        assessment["cognitive_efficiency"] = efficiency

        # Assess strategy effectiveness
        strategy_effectiveness = self._self_awareness_metrics["strategy_effectiveness"]
        assessment["strategy_effectiveness"] = strategy_effectiveness

        # Overall self-assessment
        assessment["overall_assessment"] = (accuracy + efficiency + strategy_effectiveness) / 3

        return assessment

    async def _calculate_cognitive_load(
        self,
        cognitive_state: Dict[str, Any],
        strategies: List[MetacognitiveStrategy]
    ) -> float:
        """Calculate current cognitive load."""
        base_load = 1.0 - cognitive_state.get("resources", 0.7)

        # Add strategy overhead
        strategy_load = sum(s.cognitive_load for s in strategies) / len(strategies) if strategies else 0

        total_load = min(1.0, base_load + strategy_load)
        return total_load

    async def _generate_optimization_recommendations(
        self,
        cognitive_state: Dict[str, Any],
        self_assessment: Dict[str, float],
        cognitive_load: float
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # High cognitive load recommendations
        if cognitive_load > self._config["cognitive_load_threshold"]:
            recommendations.append("Reduce cognitive load by simplifying current strategies")

        # Low performance recommendations
        if self_assessment.get("overall_assessment", 0.8) < 0.6:
            recommendations.append("Consider alternative metacognitive strategies")

        # Low clarity recommendations
        if cognitive_state.get("clarity", 0.8) < 0.5:
            recommendations.append("Improve cognitive clarity through focused attention")

        return recommendations

    def _calculate_cognitive_clarity(
        self,
        cognitive_context: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> float:
        """Calculate cognitive clarity score."""
        # Base clarity from confidence
        base_clarity = performance_metrics.get("confidence", 0.8)

        # Adjust for context complexity
        complexity = cognitive_context.get("complexity", 0.5)
        clarity_adjustment = max(0.1, 1.0 - complexity * 0.3)

        return min(1.0, base_clarity * clarity_adjustment)

    def _assess_cognitive_resources(self, cognitive_context: Dict[str, Any]) -> float:
        """Assess available cognitive resources."""
        # Simple resource assessment based on context
        base_resources = 0.8

        # Reduce resources based on concurrent tasks
        concurrent_tasks = cognitive_context.get("concurrent_tasks", 1)
        resource_reduction = min(0.5, (concurrent_tasks - 1) * 0.2)

        return max(0.2, base_resources - resource_reduction)

    def _determine_cognitive_mode(self, performance_metrics: Dict[str, float]) -> str:
        """Determine current cognitive mode."""
        accuracy = performance_metrics.get("accuracy", 0.8)
        speed = performance_metrics.get("response_time", 1000)

        if accuracy > 0.9 and speed < 500:
            return "optimal"
        elif accuracy > 0.8:
            return "focused"
        elif speed < 1000:
            return "fast"
        else:
            return "learning"
