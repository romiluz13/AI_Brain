"""
Pydantic AI Adapter

Integrates the Universal AI Brain with Pydantic AI framework.
Provides type-safe AI agent integration with cognitive capabilities.

Features:
- Type-safe cognitive agent integration
- Pydantic model validation with AI Brain
- Enhanced agent tools with cognitive capabilities
- Structured output with cognitive insights
- Type-safe configuration and responses
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, Field, validator

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models import Model
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    # Fallback classes for when Pydantic AI is not installed
    class Agent:
        def __init__(self, model, **kwargs):
            self.model = model
        def run_sync(self, prompt, **kwargs):
            return type('Result', (), {'output': 'Pydantic AI not available'})()
        async def run(self, prompt, **kwargs):
            return type('Result', (), {'output': 'Pydantic AI not available'})()
    class RunContext:
        def __init__(self, deps):
            self.deps = deps
    class Model:
        pass
    PYDANTIC_AI_AVAILABLE = False

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext
from ai_brain_python.adapters.base_adapter import BaseFrameworkAdapter

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class CognitiveAgentConfig(BaseModel):
    """Configuration for cognitive-enhanced Pydantic AI agents."""
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    cognitive_systems: List[str] = Field(
        default=[
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "safety_guardrails"
        ],
        description="List of cognitive systems to enable"
    )
    enable_learning: bool = Field(default=True, description="Enable learning from interactions")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")
    safety_level: str = Field(default="standard", description="Safety level: strict, standard, or relaxed")
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v
    
    @validator('safety_level')
    def validate_safety_level(cls, v):
        if v not in ['strict', 'standard', 'relaxed']:
            raise ValueError('Safety level must be strict, standard, or relaxed')
        return v


class CognitiveResponse(BaseModel):
    """Type-safe response from cognitive-enhanced agents."""
    
    content: str = Field(..., description="Main response content")
    confidence: float = Field(..., description="Overall confidence score")
    emotional_context: Dict[str, Any] = Field(default_factory=dict, description="Emotional analysis")
    goal_insights: Dict[str, Any] = Field(default_factory=dict, description="Goal-related insights")
    safety_assessment: Dict[str, Any] = Field(default_factory=dict, description="Safety assessment")
    cognitive_insights: Dict[str, Any] = Field(default_factory=dict, description="Cognitive system insights")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    agent_id: str = Field(..., description="ID of the agent that generated the response")
    
    class Config:
        extra = "allow"


class CognitiveAgentState(BaseModel):
    """State management for cognitive agents."""
    
    agent_id: str
    emotional_state: str = "neutral"
    confidence_level: float = 0.8
    learning_enabled: bool = True
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    memory_entries: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class CognitiveAgent(Generic[T]):
    """
    Cognitive-enhanced Pydantic AI Agent.
    
    Extends Pydantic AI Agent with AI Brain cognitive capabilities
    while maintaining type safety and Pydantic validation.
    """
    
    def __init__(
        self,
        model: Optional[Model] = None,
        result_type: Type[T] = CognitiveResponse,
        config: Optional[CognitiveAgentConfig] = None,
        ai_brain_config: Optional[UniversalAIBrainConfig] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError("Pydantic AI is not installed. Please install it with: pip install pydantic-ai")
        
        self.config = config or CognitiveAgentConfig(agent_id=f"pydantic_agent_{id(self)}")
        self.ai_brain_config = ai_brain_config
        self.ai_brain: Optional[UniversalAIBrain] = None
        self.result_type = result_type
        
        # Agent state
        self.state = CognitiveAgentState(agent_id=self.config.agent_id)
        
        # Initialize Pydantic AI agent with exact API
        self.pydantic_agent = PydanticAgent(
            model=model or 'openai:gpt-4o',
            deps_type=type(None),  # Can be overridden when running
            output_type=result_type,
            system_prompt=system_prompt or self._generate_cognitive_system_prompt(),
            **kwargs
        ) if PYDANTIC_AI_AVAILABLE else None

        # Register cognitive tools using @agent.tool decorator
        if self.pydantic_agent:
            self._register_cognitive_tools()
    
    async def initialize_ai_brain(self) -> None:
        """Initialize the AI Brain for this agent."""
        if self.ai_brain_config and not self.ai_brain:
            self.ai_brain = UniversalAIBrain(self.ai_brain_config)
            await self.ai_brain.initialize()
            logger.info(f"AI Brain initialized for Pydantic AI agent: {self.config.agent_id}")
    
    async def run(
        self,
        user_prompt: str,
        message_history: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> CognitiveResponse:
        """
        Run the agent with cognitive enhancement.
        
        Processes the user prompt through AI Brain cognitive systems
        and returns a type-safe cognitive response.
        """
        if not self.ai_brain:
            await self.initialize_ai_brain()
        
        start_time = datetime.utcnow()
        
        try:
            # Process through AI Brain first
            cognitive_analysis = await self._process_with_ai_brain(user_prompt)
            
            # Update agent state
            await self._update_agent_state(cognitive_analysis)
            
            # Generate enhanced prompt with cognitive insights
            enhanced_prompt = await self._enhance_prompt_with_cognitive_insights(
                user_prompt, cognitive_analysis
            )
            
            # Run Pydantic AI agent with enhanced prompt using exact API
            if self.pydantic_agent:
                pydantic_result = await self.pydantic_agent.run(enhanced_prompt, **kwargs)
                # Extract output from RunResult
                content = pydantic_result.output if hasattr(pydantic_result, 'output') else str(pydantic_result)
            else:
                content = f"Cognitive analysis: {user_prompt}"
            
            # Create cognitive response
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response = CognitiveResponse(
                content=content,
                confidence=cognitive_analysis.confidence,
                emotional_context=self._extract_emotional_context(cognitive_analysis),
                goal_insights=self._extract_goal_insights(cognitive_analysis),
                safety_assessment=self._extract_safety_assessment(cognitive_analysis),
                cognitive_insights=self._extract_cognitive_insights(cognitive_analysis),
                processing_time_ms=processing_time,
                agent_id=self.config.agent_id
            )
            
            # Learn from interaction
            if self.config.enable_learning:
                await self._learn_from_interaction(user_prompt, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in cognitive agent run: {e}")
            # Return error response
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return CognitiveResponse(
                content=f"Error processing request: {str(e)}",
                confidence=0.0,
                processing_time_ms=processing_time,
                agent_id=self.config.agent_id
            )
    
    def run_sync(self, user_prompt: str, **kwargs) -> CognitiveResponse:
        """Synchronous version of run method using Pydantic AI's run_sync."""
        if self.pydantic_agent:
            # Use Pydantic AI's run_sync method
            import asyncio
            return asyncio.run(self.run(user_prompt, **kwargs))
        else:
            # Fallback
            return CognitiveResponse(
                content="Pydantic AI not available",
                confidence=0.0,
                processing_time_ms=0.0,
                agent_id=self.config.agent_id
            )
    
    def add_cognitive_tool(self, tool: Tool) -> None:
        """Add a cognitive-enhanced tool to the agent."""
        self.cognitive_tools.append(tool)
        if self.pydantic_agent:
            # Add tool to Pydantic AI agent
            pass  # Implementation depends on Pydantic AI tool system
    
    async def _process_with_ai_brain(self, user_prompt: str):
        """Process user prompt through AI Brain cognitive systems."""
        if not self.ai_brain:
            raise RuntimeError("AI Brain not initialized")
        
        cognitive_context = CognitiveContext(
            user_id=self.config.agent_id,
            session_id=f"pydantic_session_{datetime.utcnow().timestamp()}"
        )
        
        input_data = CognitiveInputData(
            text=user_prompt,
            input_type="pydantic_ai_request",
            context=cognitive_context,
            requested_systems=self.config.cognitive_systems,
            processing_priority=8
        )
        
        return await self.ai_brain.process_input(input_data)
    
    async def _update_agent_state(self, cognitive_analysis) -> None:
        """Update agent state based on cognitive analysis."""
        self.state.interaction_count += 1
        self.state.last_interaction = datetime.utcnow()
        
        # Update emotional state
        cognitive_results = cognitive_analysis.cognitive_results
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            if "emotional_state" in emotional_result:
                self.state.emotional_state = emotional_result["emotional_state"].get("primary_emotion", "neutral")
        
        # Update confidence level
        self.state.confidence_level = cognitive_analysis.confidence
    
    async def _enhance_prompt_with_cognitive_insights(self, original_prompt: str, cognitive_analysis) -> str:
        """Enhance the original prompt with cognitive insights."""
        enhanced_prompt = original_prompt
        
        cognitive_results = cognitive_analysis.cognitive_results
        
        # Add emotional context
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            if "emotional_state" in emotional_result:
                emotional_state = emotional_result["emotional_state"]
                enhanced_prompt += f"\n\nEmotional Context: {emotional_state.get('primary_emotion', 'neutral')}"
        
        # Add goal insights
        if "goal_hierarchy" in cognitive_results:
            goal_result = cognitive_results["goal_hierarchy"]
            if "extracted_goals" in goal_result:
                goals = goal_result["extracted_goals"]
                if goals:
                    goal_text = ", ".join([g.get("text", "") for g in goals[:2]])
                    enhanced_prompt += f"\n\nIdentified Goals: {goal_text}"
        
        # Add confidence context
        enhanced_prompt += f"\n\nConfidence Level: {cognitive_analysis.confidence:.2f}"
        
        return enhanced_prompt
    
    def _extract_emotional_context(self, cognitive_analysis) -> Dict[str, Any]:
        """Extract emotional context from cognitive analysis."""
        emotional_context = {}
        
        cognitive_results = cognitive_analysis.cognitive_results
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            if "emotional_state" in emotional_result:
                emotional_context = emotional_result["emotional_state"]
        
        return emotional_context
    
    def _extract_goal_insights(self, cognitive_analysis) -> Dict[str, Any]:
        """Extract goal insights from cognitive analysis."""
        goal_insights = {}
        
        cognitive_results = cognitive_analysis.cognitive_results
        if "goal_hierarchy" in cognitive_results:
            goal_result = cognitive_results["goal_hierarchy"]
            goal_insights = {
                "extracted_goals": goal_result.get("extracted_goals", []),
                "goal_count": len(goal_result.get("extracted_goals", []))
            }
        
        return goal_insights
    
    def _extract_safety_assessment(self, cognitive_analysis) -> Dict[str, Any]:
        """Extract safety assessment from cognitive analysis."""
        safety_assessment = {}
        
        cognitive_results = cognitive_analysis.cognitive_results
        if "safety_guardrails" in cognitive_results:
            safety_result = cognitive_results["safety_guardrails"]
            if "safety_assessment" in safety_result:
                safety_assessment = safety_result["safety_assessment"]
        
        return safety_assessment
    
    def _extract_cognitive_insights(self, cognitive_analysis) -> Dict[str, Any]:
        """Extract cognitive insights from analysis."""
        insights = {}
        
        for system, result in cognitive_analysis.cognitive_results.items():
            if isinstance(result, dict) and result.get("status") == "completed":
                insights[system] = {
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": result.get("processing_time_ms", 0)
                }
        
        return insights
    
    async def _learn_from_interaction(self, user_prompt: str, response: CognitiveResponse) -> None:
        """Learn from the interaction for future improvements."""
        learning_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_prompt": user_prompt,
            "response_confidence": response.confidence,
            "emotional_state": self.state.emotional_state,
            "processing_time": response.processing_time_ms
        }
        
        self.state.memory_entries.append(learning_entry)
        
        # Keep only recent memory entries
        if len(self.state.memory_entries) > 100:
            self.state.memory_entries = self.state.memory_entries[-100:]
    
    def _generate_cognitive_system_prompt(self) -> str:
        """Generate system prompt with cognitive capabilities."""
        return f"""
You are a cognitive-enhanced AI agent with advanced emotional intelligence, goal understanding, and safety awareness.

Agent ID: {self.config.agent_id}
Cognitive Systems: {', '.join(self.config.cognitive_systems)}
Safety Level: {self.config.safety_level}

You have access to:
- Emotional intelligence for understanding user emotions and responding empathetically
- Goal hierarchy analysis for understanding user objectives
- Confidence tracking for assessing response reliability
- Safety guardrails for ensuring safe and appropriate responses

Always consider the emotional context, user goals, and safety implications in your responses.
Provide helpful, accurate, and emotionally appropriate assistance.
"""
    
    def _register_cognitive_tools(self) -> None:
        """Register cognitive-enhanced tools using Pydantic AI @agent.tool decorator."""
        if not self.pydantic_agent:
            return

        # Register cognitive analysis tool
        @self.pydantic_agent.tool
        async def cognitive_analysis(ctx: RunContext, query: str) -> str:
            """Analyze query using AI Brain cognitive systems."""
            if self.ai_brain:
                cognitive_context = CognitiveContext(
                    user_id=self.config.agent_id,
                    session_id=f"tool_session_{datetime.utcnow().timestamp()}"
                )

                input_data = CognitiveInputData(
                    text=query,
                    input_type="tool_analysis",
                    context=cognitive_context,
                    requested_systems=self.config.cognitive_systems,
                    processing_priority=7
                )

                response = await self.ai_brain.process_input(input_data)
                return f"Cognitive analysis: Confidence {response.confidence:.2f}, Systems: {list(response.cognitive_results.keys())}"
            else:
                return "AI Brain not available for cognitive analysis"
    
    def get_agent_state(self) -> CognitiveAgentState:
        """Get current agent state."""
        return self.state
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive capabilities and state."""
        return {
            "agent_id": self.config.agent_id,
            "cognitive_systems": self.config.cognitive_systems,
            "current_state": {
                "emotional_state": self.state.emotional_state,
                "confidence_level": self.state.confidence_level,
                "interaction_count": self.state.interaction_count,
                "learning_enabled": self.config.enable_learning
            },
            "memory_entries": len(self.state.memory_entries),
            "last_interaction": self.state.last_interaction.isoformat() if self.state.last_interaction else None
        }


class PydanticAIAdapter(BaseFrameworkAdapter):
    """
    Pydantic AI Framework Adapter for Universal AI Brain.
    
    Provides type-safe integration between AI Brain and Pydantic AI framework.
    """
    
    def __init__(self, ai_brain_config: UniversalAIBrainConfig):
        super().__init__("pydantic_ai", ai_brain_config)
        
        if not PYDANTIC_AI_AVAILABLE:
            logger.warning("Pydantic AI is not installed. Some features may not be available.")
    
    async def _framework_specific_initialization(self) -> None:
        """Pydantic AI specific initialization."""
        logger.debug("Pydantic AI adapter initialization complete")
    
    async def _framework_specific_shutdown(self) -> None:
        """Pydantic AI specific shutdown."""
        logger.debug("Pydantic AI adapter shutdown complete")
    
    def _get_default_systems(self) -> List[str]:
        """Get default cognitive systems for Pydantic AI."""
        return [
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "safety_guardrails",
            "semantic_memory"
        ]
    
    async def _enhance_response(self, response, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Pydantic AI specific enhancements."""
        return self._format_response_for_framework(response, {
            "pydantic_ai_features": [
                "Type Safety",
                "Model Validation",
                "Structured Output",
                "Cognitive Enhancement"
            ]
        })
    
    def create_cognitive_agent(
        self,
        model: Optional[Model] = None,
        result_type: Type[T] = CognitiveResponse,
        config: Optional[CognitiveAgentConfig] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> CognitiveAgent[T]:
        """Create a cognitive-enhanced Pydantic AI agent."""
        return CognitiveAgent(
            model=model,
            result_type=result_type,
            config=config,
            ai_brain_config=self.ai_brain_config,
            system_prompt=system_prompt,
            **kwargs
        )
    
    def create_cognitive_response_model(
        self,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Type[CognitiveResponse]:
        """Create a custom cognitive response model with additional fields."""
        if additional_fields:
            # Create dynamic model with additional fields
            fields = CognitiveResponse.__fields__.copy()
            fields.update(additional_fields)
            
            return type(
                "CustomCognitiveResponse",
                (CognitiveResponse,),
                {"__annotations__": {k: v.type_ for k, v in fields.items()}}
            )
        
        return CognitiveResponse
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get Pydantic AI framework information."""
        return {
            "framework": "Pydantic AI",
            "version": "latest",
            "available": PYDANTIC_AI_AVAILABLE,
            "cognitive_features": [
                "Type-Safe Cognitive Agents",
                "Structured Cognitive Responses",
                "Model Validation",
                "Cognitive State Management",
                "Enhanced Tools",
                "Learning Integration"
            ],
            "integration_level": "native",
            "type_safety": True
        }
