"""
Base Framework Adapter

Abstract base class for all AI framework adapters.
Provides common functionality and interface for framework integration.

Features:
- Common adapter interface and patterns
- Framework detection and validation
- Configuration management
- Error handling and logging
- Performance monitoring
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext

logger = logging.getLogger(__name__)


class BaseFrameworkAdapter(ABC):
    """
    Abstract base class for all framework adapters.
    
    Provides common functionality and interface that all framework
    adapters must implement.
    """
    
    def __init__(self, framework_name: str, ai_brain_config: UniversalAIBrainConfig):
        self.framework_name = framework_name
        self.ai_brain_config = ai_brain_config
        self.ai_brain: Optional[UniversalAIBrain] = None
        
        # Adapter state
        self.is_initialized = False
        self.initialization_time: Optional[datetime] = None
        self.usage_stats: Dict[str, int] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
        
        # Framework-specific configuration
        self.framework_config: Dict[str, Any] = {}
        
        logger.info(f"Initialized {framework_name} adapter")
    
    async def initialize(self) -> None:
        """Initialize the adapter and AI Brain."""
        try:
            if not self.ai_brain:
                self.ai_brain = UniversalAIBrain(self.ai_brain_config)
                await self.ai_brain.initialize()
            
            await self._framework_specific_initialization()
            
            self.is_initialized = True
            self.initialization_time = datetime.utcnow()
            
            logger.info(f"{self.framework_name} adapter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.framework_name} adapter: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the adapter and cleanup resources."""
        try:
            if self.ai_brain:
                await self.ai_brain.shutdown()
            
            await self._framework_specific_shutdown()
            
            self.is_initialized = False
            
            logger.info(f"{self.framework_name} adapter shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during {self.framework_name} adapter shutdown: {e}")
    
    async def process_with_ai_brain(
        self,
        input_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        requested_systems: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through AI Brain with framework-specific enhancements.
        
        This is the main method that framework adapters use to leverage
        AI Brain capabilities.
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.ai_brain:
            raise RuntimeError(f"{self.framework_name} adapter not properly initialized")
        
        try:
            self.usage_stats["total_requests"] += 1
            
            # Create cognitive input
            cognitive_context = CognitiveContext(
                user_id=user_id or f"{self.framework_name}_user",
                session_id=session_id or f"{self.framework_name}_session_{datetime.utcnow().timestamp()}"
            )
            
            input_data = CognitiveInputData(
                text=input_text,
                input_type="framework_request",
                context=cognitive_context,
                requested_systems=requested_systems or self._get_default_systems(),
                processing_priority=7
            )
            
            # Process through AI Brain
            response = await self.ai_brain.process_input(input_data)
            
            # Apply framework-specific enhancements
            enhanced_response = await self._enhance_response(response, context or {})
            
            self.usage_stats["successful_requests"] += 1
            
            return enhanced_response
            
        except Exception as e:
            self.usage_stats["failed_requests"] += 1
            logger.error(f"Error processing request in {self.framework_name} adapter: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get adapter usage statistics."""
        success_rate = 0.0
        if self.usage_stats["total_requests"] > 0:
            success_rate = self.usage_stats["successful_requests"] / self.usage_stats["total_requests"]
        
        return {
            "framework": self.framework_name,
            "is_initialized": self.is_initialized,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "usage_stats": self.usage_stats.copy(),
            "success_rate": success_rate,
            "framework_config": self.framework_config.copy()
        }
    
    def configure_framework(self, config: Dict[str, Any]) -> None:
        """Configure framework-specific settings."""
        self.framework_config.update(config)
        logger.info(f"Updated {self.framework_name} configuration")
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _framework_specific_initialization(self) -> None:
        """Framework-specific initialization logic."""
        pass
    
    @abstractmethod
    async def _framework_specific_shutdown(self) -> None:
        """Framework-specific shutdown logic."""
        pass
    
    @abstractmethod
    def _get_default_systems(self) -> List[str]:
        """Get default cognitive systems for this framework."""
        pass
    
    @abstractmethod
    async def _enhance_response(self, response, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply framework-specific enhancements to AI Brain response."""
        pass
    
    @abstractmethod
    def get_framework_info(self) -> Dict[str, Any]:
        """Get framework-specific information and capabilities."""
        pass
    
    # Helper methods for common functionality
    
    def _extract_cognitive_insights(self, cognitive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from cognitive results."""
        insights = {}
        
        for system, result in cognitive_results.items():
            if isinstance(result, dict) and result.get("status") == "completed":
                insights[system] = {
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": result.get("processing_time_ms", 0),
                    "status": result.get("status", "unknown")
                }
        
        return insights
    
    def _calculate_overall_confidence(self, cognitive_results: Dict[str, Any]) -> float:
        """Calculate overall confidence from cognitive results."""
        confidence_scores = []
        
        for result in cognitive_results.values():
            if isinstance(result, dict) and "confidence" in result:
                confidence_scores.append(result["confidence"])
        
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        
        return 0.5  # Default confidence
    
    def _get_emotional_context(self, cognitive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotional context from cognitive results."""
        emotional_context = {
            "primary_emotion": "neutral",
            "emotion_intensity": 0.5,
            "empathy_response": "neutral"
        }
        
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            if "emotional_state" in emotional_result:
                emotional_state = emotional_result["emotional_state"]
                emotional_context.update({
                    "primary_emotion": emotional_state.get("primary_emotion", "neutral"),
                    "emotion_intensity": emotional_state.get("emotion_intensity", 0.5)
                })
            
            if "empathy_response" in emotional_result:
                empathy = emotional_result["empathy_response"]
                emotional_context["empathy_response"] = empathy.get("response_strategy", "neutral")
        
        return emotional_context
    
    def _get_goal_insights(self, cognitive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract goal-related insights from cognitive results."""
        goal_insights = {
            "identified_goals": [],
            "goal_count": 0,
            "primary_goal": None
        }
        
        if "goal_hierarchy" in cognitive_results:
            goal_result = cognitive_results["goal_hierarchy"]
            if "extracted_goals" in goal_result:
                goals = goal_result["extracted_goals"]
                goal_insights.update({
                    "identified_goals": goals,
                    "goal_count": len(goals),
                    "primary_goal": goals[0] if goals else None
                })
        
        return goal_insights
    
    def _get_safety_assessment(self, cognitive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract safety assessment from cognitive results."""
        safety_assessment = {
            "is_safe": True,
            "risk_score": 0.0,
            "safety_level": "safe"
        }
        
        if "safety_guardrails" in cognitive_results:
            safety_result = cognitive_results["safety_guardrails"]
            if "safety_assessment" in safety_result:
                assessment = safety_result["safety_assessment"]
                safety_assessment.update({
                    "is_safe": assessment.get("is_safe", True),
                    "risk_score": assessment.get("risk_score", 0.0),
                    "safety_level": assessment.get("safety_level", "safe")
                })
        
        return safety_assessment
    
    def _format_response_for_framework(
        self,
        original_response,
        framework_specific_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format AI Brain response for framework consumption."""
        cognitive_results = original_response.cognitive_results
        
        formatted_response = {
            "success": original_response.success,
            "confidence": original_response.confidence,
            "processing_time_ms": original_response.processing_time_ms,
            "framework": self.framework_name,
            "cognitive_insights": self._extract_cognitive_insights(cognitive_results),
            "emotional_context": self._get_emotional_context(cognitive_results),
            "goal_insights": self._get_goal_insights(cognitive_results),
            "safety_assessment": self._get_safety_assessment(cognitive_results),
            "raw_cognitive_results": cognitive_results
        }
        
        # Add framework-specific data
        if framework_specific_data:
            formatted_response["framework_data"] = framework_specific_data
        
        return formatted_response
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the adapter."""
        health_status = {
            "adapter_name": self.framework_name,
            "is_initialized": self.is_initialized,
            "ai_brain_available": self.ai_brain is not None,
            "status": "healthy"
        }
        
        try:
            if self.ai_brain:
                # Test AI Brain connectivity
                test_context = CognitiveContext(
                    user_id="health_check",
                    session_id="health_check"
                )
                
                test_input = CognitiveInputData(
                    text="Health check test",
                    input_type="health_check",
                    context=test_context,
                    requested_systems=["emotional_intelligence"],
                    processing_priority=1
                )
                
                response = await self.ai_brain.process_input(test_input)
                health_status["ai_brain_responsive"] = response.success
                
                if not response.success:
                    health_status["status"] = "degraded"
            else:
                health_status["status"] = "unhealthy"
                health_status["error"] = "AI Brain not initialized"
                
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
