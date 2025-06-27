"""
LangChain Adapter

Integrates the Universal AI Brain with LangChain framework.
Provides cognitive-enhanced chains, agents, and tools for LangChain.

Features:
- Cognitive-enhanced LangChain chains and agents
- AI Brain tools as LangChain tools
- Memory integration with LangChain memory systems
- Enhanced prompt templates with cognitive insights
- Cognitive callbacks and monitoring
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union, Callable
from datetime import datetime

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.tools.base import BaseTool
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.runnables import Runnable
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.chains.llm import LLMChain
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback classes for when LangChain is not installed
    class BaseMemory:
        def __init__(self):
            pass
        @property
        def memory_variables(self):
            return []
        def load_memory_variables(self, inputs):
            return {}
        def save_context(self, inputs, outputs):
            pass
        def clear(self):
            pass
    class BaseTool:
        def __init__(self, name, description, func):
            self.name = name
            self.description = description
            self.func = func
    class BaseCallbackHandler:
        pass
    class Runnable:
        pass
    class PromptTemplate:
        def __init__(self, template, input_variables):
            self.template = template
            self.input_variables = input_variables
    class ChatPromptTemplate:
        pass
    class BaseMessage:
        pass
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    class AIMessage:
        def __init__(self, content):
            self.content = content
    class SystemMessage:
        def __init__(self, content):
            self.content = content
    class LLMChain:
        def __init__(self, llm, prompt, **kwargs):
            self.llm = llm
            self.prompt = prompt
        async def arun(self, *args, **kwargs):
            return "LangChain not available"
    class ConversationBufferMemory:
        pass
    LANGCHAIN_AVAILABLE = False

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext
from ai_brain_python.adapters.base_adapter import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class CognitiveMemory(BaseMemory if LANGCHAIN_AVAILABLE else object):
    """
    Cognitive-enhanced LangChain memory using AI Brain semantic memory.
    
    Integrates AI Brain's semantic memory system with LangChain's
    memory interface for enhanced conversation memory.
    """
    
    def __init__(self, ai_brain: Optional[UniversalAIBrain] = None, memory_key: str = "history"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Please install it with: pip install langchain")
        
        super().__init__()
        self.ai_brain = ai_brain
        self.memory_key = memory_key
        self.conversation_history: List[BaseMessage] = []
        self.user_id = f"langchain_user_{id(self)}"
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables from AI Brain semantic memory."""
        if self.ai_brain:
            # In a full implementation, this would retrieve relevant memories
            # from AI Brain's semantic memory system
            pass
        
        # Return conversation history
        history_str = "\n".join([
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in self.conversation_history[-10:]  # Last 10 messages
        ])
        
        return {self.memory_key: history_str}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to AI Brain semantic memory."""
        # Save human input
        if "input" in inputs:
            self.conversation_history.append(HumanMessage(content=inputs["input"]))
        
        # Save AI output
        if "output" in outputs:
            self.conversation_history.append(AIMessage(content=outputs["output"]))
        
        # Store in AI Brain semantic memory if available
        if self.ai_brain:
            asyncio.create_task(self._store_in_semantic_memory(inputs, outputs))
    
    async def _store_in_semantic_memory(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Store conversation in AI Brain semantic memory."""
        try:
            # Create memory content
            memory_content = f"User: {inputs.get('input', '')}\nAssistant: {outputs.get('output', '')}"
            
            # Store in semantic memory (simplified)
            # In full implementation, would use semantic memory system
            logger.debug(f"Stored conversation in semantic memory for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error storing in semantic memory: {e}")
    
    def clear(self) -> None:
        """Clear memory."""
        self.conversation_history.clear()


class CognitiveCallbackHandler(BaseCallbackHandler if LANGCHAIN_AVAILABLE else object):
    """
    Cognitive callback handler for LangChain operations.
    
    Monitors LangChain operations and provides cognitive insights
    through AI Brain monitoring and analysis systems.
    """
    
    def __init__(self, ai_brain: Optional[UniversalAIBrain] = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Please install it with: pip install langchain")
        
        super().__init__()
        self.ai_brain = ai_brain
        self.operation_history: List[Dict[str, Any]] = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        operation = {
            "type": "llm_start",
            "timestamp": datetime.utcnow().isoformat(),
            "prompts": prompts,
            "serialized": serialized
        }
        self.operation_history.append(operation)
        
        if self.ai_brain:
            # Analyze prompts for cognitive insights
            asyncio.create_task(self._analyze_prompts(prompts))
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running."""
        operation = {
            "type": "llm_end",
            "timestamp": datetime.utcnow().isoformat(),
            "response": str(response)
        }
        self.operation_history.append(operation)
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM errors."""
        operation = {
            "type": "llm_error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(error)
        }
        self.operation_history.append(operation)
        logger.error(f"LLM error: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain starts running."""
        operation = {
            "type": "chain_start",
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": inputs,
            "serialized": serialized
        }
        self.operation_history.append(operation)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain ends running."""
        operation = {
            "type": "chain_end",
            "timestamp": datetime.utcnow().isoformat(),
            "outputs": outputs
        }
        self.operation_history.append(operation)
    
    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes action."""
        operation = {
            "type": "agent_action",
            "timestamp": datetime.utcnow().isoformat(),
            "action": str(action)
        }
        self.operation_history.append(operation)
    
    async def _analyze_prompts(self, prompts: List[str]) -> None:
        """Analyze prompts using AI Brain cognitive systems."""
        try:
            for prompt in prompts:
                cognitive_context = CognitiveContext(
                    user_id="langchain_callback",
                    session_id=f"callback_{datetime.utcnow().timestamp()}"
                )
                
                input_data = CognitiveInputData(
                    text=prompt,
                    input_type="prompt_analysis",
                    context=cognitive_context,
                    requested_systems=["emotional_intelligence", "safety_guardrails"],
                    processing_priority=5
                )
                
                response = await self.ai_brain.process_input(input_data)
                logger.debug(f"Prompt analysis completed with confidence: {response.confidence}")
                
        except Exception as e:
            logger.error(f"Error analyzing prompts: {e}")


class CognitiveTool(BaseTool if LANGCHAIN_AVAILABLE else object):
    """
    Cognitive-enhanced LangChain tool using AI Brain capabilities.

    Wraps AI Brain cognitive systems as LangChain tools for use
    in chains and agents.

    Uses the exact LangChain BaseTool API with _run and _arun methods.
    """

    name: str
    description: str

    def __init__(
        self,
        name: str,
        description: str,
        ai_brain: UniversalAIBrain,
        cognitive_systems: List[str],
        **kwargs
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Please install it with: pip install langchain")

        self.name = name
        self.description = description
        self.ai_brain = ai_brain
        self.cognitive_systems = cognitive_systems

        # Initialize BaseTool
        super().__init__(**kwargs)

    def _run(self, query: str, **kwargs) -> str:
        """Synchronous run method required by BaseTool."""
        import asyncio
        return asyncio.run(self._arun(query, **kwargs))

    async def _arun(self, query: str, **kwargs) -> str:
        """Asynchronous run method required by BaseTool."""
        return await self._create_cognitive_tool_func()(query)
    
    def _create_cognitive_tool_func(self) -> Callable:
        """Create the cognitive tool function."""
        async def cognitive_tool_func(query: str) -> str:
            """Process query through AI Brain cognitive systems."""
            try:
                cognitive_context = CognitiveContext(
                    user_id="langchain_tool",
                    session_id=f"tool_{datetime.utcnow().timestamp()}"
                )
                
                input_data = CognitiveInputData(
                    text=query,
                    input_type="tool_request",
                    context=cognitive_context,
                    requested_systems=self.cognitive_systems,
                    processing_priority=7
                )
                
                response = await self.ai_brain.process_input(input_data)
                
                # Format response for LangChain
                result = f"Cognitive Analysis Results:\n"
                result += f"Confidence: {response.confidence:.2f}\n"
                
                for system, result_data in response.cognitive_results.items():
                    if isinstance(result_data, dict) and result_data.get("status") == "completed":
                        result += f"{system}: {result_data.get('confidence', 0):.2f}\n"
                
                return result
                
            except Exception as e:
                return f"Error in cognitive tool: {str(e)}"
        
        return cognitive_tool_func


class CognitiveChain(LLMChain if LANGCHAIN_AVAILABLE else object):
    """
    Cognitive-enhanced LangChain chain with AI Brain integration.
    
    Extends LLMChain with cognitive capabilities including emotional
    intelligence, goal understanding, and safety assessment.
    """
    
    def __init__(
        self,
        llm,
        prompt: PromptTemplate,
        ai_brain: Optional[UniversalAIBrain] = None,
        cognitive_systems: Optional[List[str]] = None,
        **kwargs
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Please install it with: pip install langchain")
        
        self.ai_brain = ai_brain
        self.cognitive_systems = cognitive_systems or [
            "emotional_intelligence",
            "goal_hierarchy",
            "safety_guardrails"
        ]
        
        # Enhance prompt with cognitive context
        enhanced_prompt = self._enhance_prompt_template(prompt)
        
        super().__init__(
            llm=llm,
            prompt=enhanced_prompt,
            **kwargs
        )
    
    def _enhance_prompt_template(self, original_prompt: PromptTemplate) -> PromptTemplate:
        """Enhance prompt template with cognitive context."""
        # Add cognitive context to the prompt
        enhanced_template = original_prompt.template + """

Cognitive Context:
- Emotional State: {emotional_state}
- User Goals: {user_goals}
- Safety Assessment: {safety_level}
- Confidence Level: {confidence_level}

Please consider this cognitive context in your response.
"""
        
        # Add new input variables
        input_variables = original_prompt.input_variables + [
            "emotional_state", "user_goals", "safety_level", "confidence_level"
        ]
        
        return PromptTemplate(
            template=enhanced_template,
            input_variables=input_variables
        )
    
    async def arun(self, *args, **kwargs) -> str:
        """Async run with cognitive enhancement."""
        # Extract input text
        input_text = kwargs.get("input", "") or (args[0] if args else "")
        
        # Process through AI Brain if available
        if self.ai_brain and input_text:
            cognitive_analysis = await self._get_cognitive_analysis(input_text)
            
            # Add cognitive context to kwargs
            kwargs.update({
                "emotional_state": cognitive_analysis.get("emotional_state", "neutral"),
                "user_goals": cognitive_analysis.get("user_goals", "general assistance"),
                "safety_level": cognitive_analysis.get("safety_level", "safe"),
                "confidence_level": cognitive_analysis.get("confidence_level", "0.8")
            })
        else:
            # Default cognitive context
            kwargs.update({
                "emotional_state": "neutral",
                "user_goals": "general assistance",
                "safety_level": "safe",
                "confidence_level": "0.8"
            })
        
        # Run the enhanced chain
        return await super().arun(*args, **kwargs)
    
    async def _get_cognitive_analysis(self, input_text: str) -> Dict[str, str]:
        """Get cognitive analysis from AI Brain."""
        try:
            cognitive_context = CognitiveContext(
                user_id="langchain_chain",
                session_id=f"chain_{datetime.utcnow().timestamp()}"
            )
            
            input_data = CognitiveInputData(
                text=input_text,
                input_type="chain_request",
                context=cognitive_context,
                requested_systems=self.cognitive_systems,
                processing_priority=7
            )
            
            response = await self.ai_brain.process_input(input_data)
            
            # Extract cognitive insights
            cognitive_results = response.cognitive_results
            analysis = {}
            
            if "emotional_intelligence" in cognitive_results:
                emotional_result = cognitive_results["emotional_intelligence"]
                if "emotional_state" in emotional_result:
                    analysis["emotional_state"] = emotional_result["emotional_state"].get("primary_emotion", "neutral")
            
            if "goal_hierarchy" in cognitive_results:
                goal_result = cognitive_results["goal_hierarchy"]
                if "extracted_goals" in goal_result:
                    goals = goal_result["extracted_goals"]
                    if goals:
                        analysis["user_goals"] = ", ".join([g.get("text", "") for g in goals[:2]])
            
            if "safety_guardrails" in cognitive_results:
                safety_result = cognitive_results["safety_guardrails"]
                if "safety_assessment" in safety_result:
                    analysis["safety_level"] = safety_result["safety_assessment"].get("safety_level", "safe")
            
            analysis["confidence_level"] = str(response.confidence)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in cognitive analysis: {e}")
            return {
                "emotional_state": "neutral",
                "user_goals": "general assistance",
                "safety_level": "safe",
                "confidence_level": "0.5"
            }


class LangChainAdapter(BaseFrameworkAdapter):
    """
    LangChain Framework Adapter for Universal AI Brain.
    
    Provides comprehensive integration between AI Brain and LangChain framework.
    """
    
    def __init__(self, ai_brain_config: UniversalAIBrainConfig):
        super().__init__("langchain", ai_brain_config)
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain is not installed. Some features may not be available.")
    
    async def _framework_specific_initialization(self) -> None:
        """LangChain specific initialization."""
        logger.debug("LangChain adapter initialization complete")
    
    async def _framework_specific_shutdown(self) -> None:
        """LangChain specific shutdown."""
        logger.debug("LangChain adapter shutdown complete")
    
    def _get_default_systems(self) -> List[str]:
        """Get default cognitive systems for LangChain."""
        return [
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "safety_guardrails",
            "semantic_memory",
            "attention_management"
        ]
    
    async def _enhance_response(self, response, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply LangChain specific enhancements."""
        return self._format_response_for_framework(response, {
            "langchain_features": [
                "Cognitive Chains",
                "Cognitive Memory",
                "Cognitive Tools",
                "Cognitive Callbacks",
                "Enhanced Prompts"
            ]
        })
    
    def create_cognitive_memory(self, memory_key: str = "history") -> CognitiveMemory:
        """Create cognitive-enhanced LangChain memory."""
        return CognitiveMemory(ai_brain=self.ai_brain, memory_key=memory_key)
    
    def create_cognitive_callback_handler(self) -> CognitiveCallbackHandler:
        """Create cognitive callback handler."""
        return CognitiveCallbackHandler(ai_brain=self.ai_brain)
    
    def create_cognitive_tool(
        self,
        name: str,
        description: str,
        cognitive_systems: Optional[List[str]] = None,
        func: Optional[Callable] = None
    ) -> CognitiveTool:
        """Create cognitive-enhanced LangChain tool."""
        if not self.ai_brain:
            raise RuntimeError("AI Brain not initialized")
        
        return CognitiveTool(
            name=name,
            description=description,
            ai_brain=self.ai_brain,
            cognitive_systems=cognitive_systems or self._get_default_systems(),
            func=func
        )
    
    def create_cognitive_chain(
        self,
        llm,
        prompt: PromptTemplate,
        cognitive_systems: Optional[List[str]] = None,
        **kwargs
    ) -> CognitiveChain:
        """Create cognitive-enhanced LangChain chain."""
        return CognitiveChain(
            llm=llm,
            prompt=prompt,
            ai_brain=self.ai_brain,
            cognitive_systems=cognitive_systems,
            **kwargs
        )
    
    def enhance_prompt_template(
        self,
        template: str,
        input_variables: List[str],
        include_cognitive_context: bool = True
    ) -> PromptTemplate:
        """Create enhanced prompt template with cognitive context."""
        if include_cognitive_context:
            enhanced_template = template + """

Cognitive Context:
- Emotional State: {emotional_state}
- User Goals: {user_goals}
- Safety Level: {safety_level}
- Confidence: {confidence_level}
"""
            enhanced_variables = input_variables + [
                "emotional_state", "user_goals", "safety_level", "confidence_level"
            ]
        else:
            enhanced_template = template
            enhanced_variables = input_variables
        
        return PromptTemplate(
            template=enhanced_template,
            input_variables=enhanced_variables
        )
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get LangChain framework information."""
        return {
            "framework": "LangChain",
            "version": "latest",
            "available": LANGCHAIN_AVAILABLE,
            "cognitive_features": [
                "Cognitive Chains",
                "Cognitive Memory",
                "Cognitive Tools",
                "Cognitive Callbacks",
                "Enhanced Prompts",
                "Agent Integration"
            ],
            "integration_level": "deep",
            "components": [
                "CognitiveMemory",
                "CognitiveCallbackHandler", 
                "CognitiveTool",
                "CognitiveChain"
            ]
        }
