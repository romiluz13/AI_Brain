"""
LangGraph Adapter

Integrates the Universal AI Brain with LangGraph framework.
Provides cognitive-enhanced stateful multi-actor applications.

Features:
- Cognitive-enhanced LangGraph nodes and edges
- AI Brain state management integration
- Cognitive workflow orchestration
- Multi-agent cognitive coordination
- Stateful cognitive processing
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union, Callable, TypedDict
from datetime import datetime

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint import BaseCheckpointSaver
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback classes for when LangGraph is not installed
    class StateGraph:
        def __init__(self, state_schema, **kwargs):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = []
        def add_node(self, name, func=None):
            self.nodes[name] = func
        def add_edge(self, from_node, to_node):
            self.edges.append((from_node, to_node))
        def compile(self, **kwargs):
            return self
        async def ainvoke(self, state, **kwargs):
            return state
    class BaseCheckpointSaver:
        pass
    class ToolExecutor:
        pass
    class ToolInvocation:
        pass
    START = "START"
    END = "END"
    LANGGRAPH_AVAILABLE = False

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext
from ai_brain_python.adapters.base_adapter import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class CognitiveState(TypedDict):
    """
    Cognitive state for LangGraph applications.
    
    Extends LangGraph state with AI Brain cognitive capabilities
    and cross-node cognitive context.
    """
    
    # Standard LangGraph state
    messages: List[Dict[str, Any]]
    
    # Cognitive state extensions
    emotional_context: Dict[str, Any]
    goal_hierarchy: Dict[str, Any]
    confidence_tracking: Dict[str, Any]
    attention_state: Dict[str, Any]
    safety_assessment: Dict[str, Any]
    cognitive_memory: Dict[str, Any]
    
    # Processing metadata
    current_node: str
    processing_history: List[Dict[str, Any]]
    cognitive_insights: Dict[str, Any]


class CognitiveNode:
    """
    Cognitive-enhanced LangGraph node with AI Brain integration.
    
    Wraps LangGraph nodes with cognitive processing capabilities
    including emotional intelligence, goal understanding, and safety.
    """
    
    def __init__(
        self,
        name: str,
        ai_brain: UniversalAIBrain,
        cognitive_systems: Optional[List[str]] = None,
        node_function: Optional[Callable] = None
    ):
        self.name = name
        self.ai_brain = ai_brain
        self.cognitive_systems = cognitive_systems or [
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "safety_guardrails"
        ]
        self.node_function = node_function or self._default_cognitive_function
        self.processing_history: List[Dict[str, Any]] = []
    
    async def __call__(self, state: CognitiveState) -> CognitiveState:
        """Execute cognitive node with AI Brain processing."""
        start_time = datetime.utcnow()
        
        try:
            # Update current node in state
            state["current_node"] = self.name
            
            # Extract input from state
            input_text = self._extract_input_from_state(state)
            
            # Process through AI Brain
            cognitive_analysis = await self._process_with_ai_brain(input_text, state)
            
            # Update cognitive state
            updated_state = await self._update_cognitive_state(state, cognitive_analysis)
            
            # Execute node-specific function
            if self.node_function:
                updated_state = await self.node_function(updated_state, cognitive_analysis)
            
            # Record processing history
            processing_entry = {
                "node": self.name,
                "timestamp": start_time.isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "confidence": cognitive_analysis.confidence if cognitive_analysis else 0.5
            }
            
            updated_state["processing_history"].append(processing_entry)
            self.processing_history.append(processing_entry)
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in cognitive node {self.name}: {e}")
            # Return state with error information
            state["cognitive_insights"]["errors"] = state["cognitive_insights"].get("errors", [])
            state["cognitive_insights"]["errors"].append({
                "node": self.name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return state
    
    async def _process_with_ai_brain(self, input_text: str, state: CognitiveState):
        """Process input through AI Brain cognitive systems."""
        cognitive_context = CognitiveContext(
            user_id=f"langgraph_node_{self.name}",
            session_id=f"node_session_{datetime.utcnow().timestamp()}"
        )
        
        input_data = CognitiveInputData(
            text=input_text,
            input_type="langgraph_node",
            context=cognitive_context,
            requested_systems=self.cognitive_systems,
            processing_priority=7
        )
        
        return await self.ai_brain.process_input(input_data)
    
    async def _update_cognitive_state(self, state: CognitiveState, cognitive_analysis) -> CognitiveState:
        """Update cognitive state with AI Brain analysis results."""
        if not cognitive_analysis:
            return state
        
        cognitive_results = cognitive_analysis.cognitive_results
        
        # Update emotional context
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            state["emotional_context"] = emotional_result.get("emotional_state", {})
        
        # Update goal hierarchy
        if "goal_hierarchy" in cognitive_results:
            goal_result = cognitive_results["goal_hierarchy"]
            state["goal_hierarchy"] = {
                "extracted_goals": goal_result.get("extracted_goals", []),
                "goal_hierarchy": goal_result.get("goal_hierarchy", {})
            }
        
        # Update confidence tracking
        if "confidence_tracking" in cognitive_results:
            confidence_result = cognitive_results["confidence_tracking"]
            state["confidence_tracking"] = confidence_result.get("confidence_assessment", {})
        
        # Update safety assessment
        if "safety_guardrails" in cognitive_results:
            safety_result = cognitive_results["safety_guardrails"]
            state["safety_assessment"] = safety_result.get("safety_assessment", {})
        
        # Update cognitive insights
        state["cognitive_insights"]["overall_confidence"] = cognitive_analysis.confidence
        state["cognitive_insights"]["processing_time"] = cognitive_analysis.processing_time_ms
        
        return state
    
    def _extract_input_from_state(self, state: CognitiveState) -> str:
        """Extract input text from LangGraph state."""
        messages = state.get("messages", [])
        if messages:
            # Get the last message content
            last_message = messages[-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "")
            else:
                return str(last_message)
        return ""
    
    async def _default_cognitive_function(self, state: CognitiveState, cognitive_analysis) -> CognitiveState:
        """Default cognitive processing function."""
        # Add a response message based on cognitive analysis
        response_content = f"Cognitive analysis completed by {self.name}. "
        
        if cognitive_analysis:
            response_content += f"Confidence: {cognitive_analysis.confidence:.2f}"
            
            # Add emotional context
            emotional_context = state.get("emotional_context", {})
            if emotional_context.get("primary_emotion"):
                response_content += f", Emotion: {emotional_context['primary_emotion']}"
        
        # Add response to messages
        state["messages"].append({
            "role": "assistant",
            "content": response_content,
            "node": self.name,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state


class CognitiveGraph:
    """
    Cognitive-enhanced LangGraph with AI Brain integration.
    
    Provides stateful multi-actor applications with cognitive
    capabilities across all nodes and edges.
    """
    
    def __init__(
        self,
        ai_brain: UniversalAIBrain,
        state_schema: Type[CognitiveState] = CognitiveState,
        checkpoint_saver: Optional[BaseCheckpointSaver] = None
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is not installed. Please install it with: pip install langgraph")
        
        self.ai_brain = ai_brain
        self.state_schema = state_schema
        self.checkpoint_saver = checkpoint_saver
        
        # Initialize LangGraph StateGraph
        self.graph = StateGraph(state_schema)
        
        # Cognitive nodes registry
        self.cognitive_nodes: Dict[str, CognitiveNode] = {}
        
        # Graph execution history
        self.execution_history: List[Dict[str, Any]] = []
    
    def add_cognitive_node(
        self,
        name: str,
        cognitive_systems: Optional[List[str]] = None,
        node_function: Optional[Callable] = None
    ) -> None:
        """Add a cognitive-enhanced node to the graph."""
        cognitive_node = CognitiveNode(
            name=name,
            ai_brain=self.ai_brain,
            cognitive_systems=cognitive_systems,
            node_function=node_function
        )
        
        self.cognitive_nodes[name] = cognitive_node
        self.graph.add_node(name, cognitive_node)
        
        logger.info(f"Added cognitive node: {name}")
    
    def add_cognitive_edge(
        self,
        start_node: str,
        end_node: str,
        condition: Optional[Callable] = None
    ) -> None:
        """Add an edge between cognitive nodes."""
        if condition:
            # Conditional edge
            self.graph.add_conditional_edges(start_node, condition)
        else:
            # Simple edge
            self.graph.add_edge(start_node, end_node)
        
        logger.info(f"Added edge: {start_node} -> {end_node}")
    
    def set_entry_point(self, node_name: str) -> None:
        """Set the entry point for the graph using exact LangGraph API."""
        self.graph.add_edge(START, node_name)

    def set_finish_point(self, node_name: str) -> None:
        """Set the finish point for the graph using exact LangGraph API."""
        self.graph.add_edge(node_name, END)
    
    def compile(self) -> StateGraph:
        """Compile the cognitive graph."""
        compiled_graph = self.graph.compile(checkpointer=self.checkpoint_saver)
        logger.info("Cognitive graph compiled successfully")
        return compiled_graph
    
    async def invoke(
        self,
        initial_input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> CognitiveState:
        """Invoke the cognitive graph with initial input."""
        start_time = datetime.utcnow()
        
        try:
            # Initialize cognitive state
            initial_state = self._initialize_cognitive_state(initial_input)
            
            # Compile and run graph
            compiled_graph = self.compile()
            result = await compiled_graph.ainvoke(initial_state, config=config)
            
            # Record execution history
            execution_entry = {
                "timestamp": start_time.isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "nodes_executed": len(result.get("processing_history", [])),
                "final_confidence": result.get("cognitive_insights", {}).get("overall_confidence", 0.0)
            }
            
            self.execution_history.append(execution_entry)
            
            return result
            
        except Exception as e:
            logger.error(f"Error invoking cognitive graph: {e}")
            raise
    
    def _initialize_cognitive_state(self, initial_input: Dict[str, Any]) -> CognitiveState:
        """Initialize cognitive state from input."""
        return CognitiveState(
            messages=[{
                "role": "user",
                "content": initial_input.get("input", ""),
                "timestamp": datetime.utcnow().isoformat()
            }],
            emotional_context={},
            goal_hierarchy={},
            confidence_tracking={},
            attention_state={},
            safety_assessment={},
            cognitive_memory={},
            current_node="",
            processing_history=[],
            cognitive_insights={}
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of graph executions."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total_executions = len(self.execution_history)
        avg_processing_time = sum(e["processing_time_ms"] for e in self.execution_history) / total_executions
        avg_confidence = sum(e["final_confidence"] for e in self.execution_history) / total_executions
        
        return {
            "total_executions": total_executions,
            "average_processing_time_ms": avg_processing_time,
            "average_confidence": avg_confidence,
            "cognitive_nodes": len(self.cognitive_nodes),
            "last_execution": self.execution_history[-1]["timestamp"]
        }


class LangGraphAdapter(BaseFrameworkAdapter):
    """
    LangGraph Framework Adapter for Universal AI Brain.
    
    Provides stateful multi-actor applications with cognitive enhancement.
    """
    
    def __init__(self, ai_brain_config: UniversalAIBrainConfig):
        super().__init__("langgraph", ai_brain_config)
        
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph is not installed. Some features may not be available.")
    
    async def _framework_specific_initialization(self) -> None:
        """LangGraph specific initialization."""
        logger.debug("LangGraph adapter initialization complete")
    
    async def _framework_specific_shutdown(self) -> None:
        """LangGraph specific shutdown."""
        logger.debug("LangGraph adapter shutdown complete")
    
    def _get_default_systems(self) -> List[str]:
        """Get default cognitive systems for LangGraph."""
        return [
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "attention_management",
            "safety_guardrails",
            "workflow_orchestration"
        ]
    
    async def _enhance_response(self, response, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply LangGraph specific enhancements."""
        return self._format_response_for_framework(response, {
            "langgraph_features": [
                "Cognitive State Management",
                "Cognitive Nodes",
                "Stateful Processing",
                "Multi-Actor Coordination",
                "Cognitive Workflows"
            ]
        })
    
    def create_cognitive_graph(
        self,
        state_schema: Type[CognitiveState] = CognitiveState,
        checkpoint_saver: Optional[BaseCheckpointSaver] = None
    ) -> CognitiveGraph:
        """Create a cognitive-enhanced LangGraph."""
        if not self.ai_brain:
            raise RuntimeError("AI Brain not initialized")
        
        return CognitiveGraph(
            ai_brain=self.ai_brain,
            state_schema=state_schema,
            checkpoint_saver=checkpoint_saver
        )
    
    def create_cognitive_node(
        self,
        name: str,
        cognitive_systems: Optional[List[str]] = None,
        node_function: Optional[Callable] = None
    ) -> CognitiveNode:
        """Create a cognitive-enhanced node."""
        if not self.ai_brain:
            raise RuntimeError("AI Brain not initialized")
        
        return CognitiveNode(
            name=name,
            ai_brain=self.ai_brain,
            cognitive_systems=cognitive_systems,
            node_function=node_function
        )
    
    def create_workflow_graph(
        self,
        workflow_definition: Dict[str, Any]
    ) -> CognitiveGraph:
        """Create a cognitive workflow graph from definition."""
        graph = self.create_cognitive_graph()
        
        # Add nodes from workflow definition
        for node_name, node_config in workflow_definition.get("nodes", {}).items():
            graph.add_cognitive_node(
                name=node_name,
                cognitive_systems=node_config.get("cognitive_systems"),
                node_function=node_config.get("function")
            )
        
        # Add edges from workflow definition
        for edge in workflow_definition.get("edges", []):
            graph.add_cognitive_edge(
                start_node=edge["from"],
                end_node=edge["to"],
                condition=edge.get("condition")
            )
        
        # Set entry and finish points
        if "entry_point" in workflow_definition:
            graph.set_entry_point(workflow_definition["entry_point"])
        
        if "finish_point" in workflow_definition:
            graph.set_finish_point(workflow_definition["finish_point"])
        
        return graph
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get LangGraph framework information."""
        return {
            "framework": "LangGraph",
            "version": "latest",
            "available": LANGGRAPH_AVAILABLE,
            "cognitive_features": [
                "Cognitive State Management",
                "Cognitive Nodes",
                "Stateful Multi-Actor Applications",
                "Cognitive Workflow Orchestration",
                "Checkpoint Integration",
                "Conditional Cognitive Routing"
            ],
            "integration_level": "deep",
            "components": [
                "CognitiveGraph",
                "CognitiveNode",
                "CognitiveState"
            ],
            "stateful": True
        }
