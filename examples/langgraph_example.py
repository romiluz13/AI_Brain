"""
LangGraph Integration Example with AI Brain

This example demonstrates how to integrate the Universal AI Brain
with LangGraph workflows for enhanced cognitive capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, TypedDict

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.adapters.langgraph_adapter import LangGraphAdapter
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext

# LangGraph imports (when available)
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint import MemorySaver
    from typing_extensions import Annotated
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("LangGraph not installed. Install with: pip install langgraph")
    LANGGRAPH_AVAILABLE = False
    # Fallback
    class TypedDict:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# State definitions for different workflows
class EmotionalAnalysisState(TypedDict):
    """State for emotional analysis workflow."""
    input_text: str
    emotional_analysis: Optional[Dict[str, Any]]
    empathy_response: Optional[str]
    confidence_score: Optional[float]
    processing_complete: bool


class GoalPlanningState(TypedDict):
    """State for goal planning workflow."""
    input_text: str
    extracted_goals: Optional[List[str]]
    goal_hierarchy: Optional[Dict[str, Any]]
    strategic_plan: Optional[Dict[str, Any]]
    timeline_analysis: Optional[Dict[str, Any]]
    processing_complete: bool


class ComprehensiveAnalysisState(TypedDict):
    """State for comprehensive analysis workflow."""
    input_text: str
    emotional_analysis: Optional[Dict[str, Any]]
    goal_analysis: Optional[Dict[str, Any]]
    safety_assessment: Optional[Dict[str, Any]]
    confidence_tracking: Optional[Dict[str, Any]]
    final_recommendations: Optional[List[str]]
    processing_complete: bool


async def create_emotional_analysis_workflow():
    """Create a LangGraph workflow for emotional analysis."""
    
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available. Please install LangGraph to run this example.")
        return None, None
    
    print("🧠 LangGraph + AI Brain: Emotional Analysis Workflow")
    print("=" * 60)
    
    # Initialize AI Brain
    config = UniversalAIBrainConfig(
        mongodb_uri="mongodb://localhost:27017",
        database_name="ai_brain_langgraph_example",
        enable_safety_systems=True,
        cognitive_systems_config={
            "emotional_intelligence": {"sensitivity": 0.9},
            "empathy_response": {"response_style": "supportive"}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Initialize LangGraph adapter
    adapter = LangGraphAdapter(ai_brain_config=config, ai_brain=brain)
    
    # Create cognitive graph
    graph = adapter.create_cognitive_graph(
        ai_brain=brain,
        state_schema=EmotionalAnalysisState
    )
    
    # Add emotional analysis node
    graph.add_cognitive_node(
        name="emotional_analysis",
        cognitive_systems=["emotional_intelligence", "confidence_tracking"],
        node_function=async_emotional_analysis_node
    )
    
    # Add empathy response node
    graph.add_cognitive_node(
        name="empathy_response",
        cognitive_systems=["empathy_response", "communication_protocol"],
        node_function=async_empathy_response_node
    )
    
    # Add workflow edges
    graph.set_entry_point("emotional_analysis")
    graph.add_cognitive_edge("emotional_analysis", "empathy_response")
    graph.set_finish_point("empathy_response")
    
    # Compile the graph
    compiled_graph = graph.compile(checkpointer=MemorySaver())
    
    return compiled_graph, brain


async def async_emotional_analysis_node(state: EmotionalAnalysisState, ai_brain: UniversalAIBrain) -> EmotionalAnalysisState:
    """Node function for emotional analysis."""
    try:
        # Create cognitive input
        input_data = CognitiveInputData(
            text=state["input_text"],
            input_type="emotional_analysis",
            context=CognitiveContext(
                user_id="langgraph_user",
                session_id="emotional_workflow"
            ),
            requested_systems=["emotional_intelligence", "confidence_tracking"],
            processing_priority=8
        )
        
        # Process through AI Brain
        response = await ai_brain.process_input(input_data)
        
        # Update state
        state["emotional_analysis"] = {
            "primary_emotion": response.emotional_state.primary_emotion,
            "emotion_intensity": response.emotional_state.emotion_intensity,
            "emotional_valence": response.emotional_state.emotional_valence,
            "emotion_explanation": response.emotional_state.emotion_explanation
        }
        state["confidence_score"] = response.confidence
        
        print(f"😊 Emotional Analysis: {response.emotional_state.primary_emotion} (intensity: {response.emotional_state.emotion_intensity:.2f})")
        
        return state
        
    except Exception as e:
        logger.error(f"Emotional analysis node error: {e}")
        state["emotional_analysis"] = {"error": str(e)}
        state["confidence_score"] = 0.0
        return state


async def async_empathy_response_node(state: EmotionalAnalysisState, ai_brain: UniversalAIBrain) -> EmotionalAnalysisState:
    """Node function for empathy response generation."""
    try:
        # Create cognitive input for empathy response
        input_data = CognitiveInputData(
            text=state["input_text"],
            input_type="empathy_response",
            context=CognitiveContext(
                user_id="langgraph_user",
                session_id="emotional_workflow",
                emotional_context=state["emotional_analysis"]
            ),
            requested_systems=["empathy_response", "communication_protocol"],
            processing_priority=7
        )
        
        # Process through AI Brain
        response = await ai_brain.process_input(input_data)
        
        # Update state
        state["empathy_response"] = response.emotional_state.empathy_response
        state["processing_complete"] = True
        
        print(f"💝 Empathy Response: {response.emotional_state.empathy_response}")
        
        return state
        
    except Exception as e:
        logger.error(f"Empathy response node error: {e}")
        state["empathy_response"] = f"Error generating empathy response: {str(e)}"
        state["processing_complete"] = True
        return state


async def create_goal_planning_workflow():
    """Create a LangGraph workflow for goal planning."""
    
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available. Please install LangGraph to run this example.")
        return None, None
    
    print("\n🎯 LangGraph + AI Brain: Goal Planning Workflow")
    print("=" * 60)
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = LangGraphAdapter(ai_brain=brain)
    
    # Create cognitive graph
    graph = adapter.create_cognitive_graph(
        ai_brain=brain,
        state_schema=GoalPlanningState
    )
    
    # Add goal extraction node
    async def goal_extraction_node(state: GoalPlanningState) -> GoalPlanningState:
        try:
            input_data = CognitiveInputData(
                text=state["input_text"],
                input_type="goal_extraction",
                context=CognitiveContext(user_id="langgraph_user", session_id="goal_workflow"),
                requested_systems=["goal_hierarchy"],
                processing_priority=8
            )
            
            response = await brain.process_input(input_data)
            
            state["extracted_goals"] = response.goal_hierarchy.sub_goals
            state["goal_hierarchy"] = {
                "primary_goal": response.goal_hierarchy.primary_goal,
                "goal_priority": response.goal_hierarchy.goal_priority,
                "goal_category": response.goal_hierarchy.goal_category
            }
            
            print(f"🎯 Extracted Goals: {response.goal_hierarchy.primary_goal}")
            return state
            
        except Exception as e:
            logger.error(f"Goal extraction error: {e}")
            state["extracted_goals"] = []
            state["goal_hierarchy"] = {"error": str(e)}
            return state
    
    # Add temporal planning node
    async def temporal_planning_node(state: GoalPlanningState) -> GoalPlanningState:
        try:
            input_data = CognitiveInputData(
                text=state["input_text"],
                input_type="temporal_planning",
                context=CognitiveContext(
                    user_id="langgraph_user", 
                    session_id="goal_workflow",
                    goal_context=state["goal_hierarchy"]
                ),
                requested_systems=["temporal_planning", "strategic_thinking"],
                processing_priority=7
            )
            
            response = await brain.process_input(input_data)
            
            state["timeline_analysis"] = {
                "estimated_timeline": response.goal_hierarchy.estimated_timeline,
                "goal_dependencies": response.goal_hierarchy.goal_dependencies
            }
            state["processing_complete"] = True
            
            print(f"⏰ Timeline: {response.goal_hierarchy.estimated_timeline}")
            return state
            
        except Exception as e:
            logger.error(f"Temporal planning error: {e}")
            state["timeline_analysis"] = {"error": str(e)}
            state["processing_complete"] = True
            return state
    
    # Add nodes to graph
    graph.add_node("goal_extraction", goal_extraction_node)
    graph.add_node("temporal_planning", temporal_planning_node)
    
    # Add workflow edges
    graph.set_entry_point("goal_extraction")
    graph.add_edge("goal_extraction", "temporal_planning")
    graph.set_finish_point("temporal_planning")
    
    # Compile the graph
    compiled_graph = graph.compile()
    
    return compiled_graph, brain


async def create_comprehensive_analysis_workflow():
    """Create a comprehensive analysis workflow with multiple cognitive systems."""
    
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available. Please install LangGraph to run this example.")
        return None, None
    
    print("\n🔍 LangGraph + AI Brain: Comprehensive Analysis Workflow")
    print("=" * 60)
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = LangGraphAdapter(ai_brain=brain)
    
    # Create cognitive graph
    graph = adapter.create_cognitive_graph(
        ai_brain=brain,
        state_schema=ComprehensiveAnalysisState
    )
    
    # Define node functions
    async def emotional_node(state: ComprehensiveAnalysisState) -> ComprehensiveAnalysisState:
        input_data = CognitiveInputData(
            text=state["input_text"],
            input_type="emotional_analysis",
            context=CognitiveContext(user_id="comprehensive_user", session_id="comp_workflow"),
            requested_systems=["emotional_intelligence"],
            processing_priority=8
        )
        response = await brain.process_input(input_data)
        state["emotional_analysis"] = {
            "emotion": response.emotional_state.primary_emotion,
            "intensity": response.emotional_state.emotion_intensity
        }
        return state
    
    async def goal_node(state: ComprehensiveAnalysisState) -> ComprehensiveAnalysisState:
        input_data = CognitiveInputData(
            text=state["input_text"],
            input_type="goal_analysis",
            context=CognitiveContext(user_id="comprehensive_user", session_id="comp_workflow"),
            requested_systems=["goal_hierarchy"],
            processing_priority=7
        )
        response = await brain.process_input(input_data)
        state["goal_analysis"] = {
            "primary_goal": response.goal_hierarchy.primary_goal,
            "priority": response.goal_hierarchy.goal_priority
        }
        return state
    
    async def safety_node(state: ComprehensiveAnalysisState) -> ComprehensiveAnalysisState:
        input_data = CognitiveInputData(
            text=state["input_text"],
            input_type="safety_assessment",
            context=CognitiveContext(user_id="comprehensive_user", session_id="comp_workflow"),
            requested_systems=["safety_guardrails", "confidence_tracking"],
            processing_priority=9
        )
        response = await brain.process_input(input_data)
        state["safety_assessment"] = {"confidence": response.confidence}
        state["confidence_tracking"] = {"overall_confidence": response.confidence}
        return state
    
    async def synthesis_node(state: ComprehensiveAnalysisState) -> ComprehensiveAnalysisState:
        # Synthesize all analyses into recommendations
        recommendations = []
        
        if state.get("emotional_analysis"):
            emotion = state["emotional_analysis"]["emotion"]
            recommendations.append(f"Address {emotion} emotion with appropriate support")
        
        if state.get("goal_analysis"):
            goal = state["goal_analysis"]["primary_goal"]
            recommendations.append(f"Focus on achieving: {goal}")
        
        if state.get("confidence_tracking"):
            confidence = state["confidence_tracking"]["overall_confidence"]
            if confidence < 0.7:
                recommendations.append("Consider additional validation due to lower confidence")
        
        state["final_recommendations"] = recommendations
        state["processing_complete"] = True
        return state
    
    # Add nodes to graph
    graph.add_node("emotional_analysis", emotional_node)
    graph.add_node("goal_analysis", goal_node)
    graph.add_node("safety_assessment", safety_node)
    graph.add_node("synthesis", synthesis_node)
    
    # Create parallel processing for analysis nodes
    graph.set_entry_point("emotional_analysis")
    graph.set_entry_point("goal_analysis")
    graph.set_entry_point("safety_assessment")
    
    # All analysis nodes feed into synthesis
    graph.add_edge("emotional_analysis", "synthesis")
    graph.add_edge("goal_analysis", "synthesis")
    graph.add_edge("safety_assessment", "synthesis")
    
    graph.set_finish_point("synthesis")
    
    # Compile the graph
    compiled_graph = graph.compile()
    
    return compiled_graph, brain


async def run_emotional_workflow_example():
    """Run the emotional analysis workflow example."""
    
    workflow_brain_pair = await create_emotional_analysis_workflow()
    if not workflow_brain_pair[0]:
        return
    
    workflow, brain = workflow_brain_pair
    
    try:
        # Test input
        test_input = """I just received news that I didn't get the promotion I was 
        hoping for. I've been working towards this for months and I'm feeling 
        disappointed and questioning my abilities."""
        
        print(f"📝 Input: {test_input}")
        print("\n🔍 Processing through emotional workflow...")
        
        # Run the workflow
        initial_state = EmotionalAnalysisState(
            input_text=test_input,
            emotional_analysis=None,
            empathy_response=None,
            confidence_score=None,
            processing_complete=False
        )
        
        result = await workflow.ainvoke(initial_state)
        
        print(f"\n📊 Workflow Results:")
        print(f"Emotional Analysis: {result['emotional_analysis']}")
        print(f"Empathy Response: {result['empathy_response']}")
        print(f"Confidence Score: {result['confidence_score']:.2f}")
        
    except Exception as e:
        logger.error(f"Emotional workflow example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def run_goal_planning_workflow_example():
    """Run the goal planning workflow example."""
    
    workflow_brain_pair = await create_goal_planning_workflow()
    if not workflow_brain_pair[0]:
        return
    
    workflow, brain = workflow_brain_pair
    
    try:
        # Test input
        test_input = """I want to transition from my current role in marketing to 
        become a data scientist. I need to learn Python, statistics, machine learning, 
        and build a portfolio. I'd like to make this transition within 18 months."""
        
        print(f"📝 Goal Input: {test_input}")
        print("\n🔍 Processing through goal planning workflow...")
        
        # Run the workflow
        initial_state = GoalPlanningState(
            input_text=test_input,
            extracted_goals=None,
            goal_hierarchy=None,
            strategic_plan=None,
            timeline_analysis=None,
            processing_complete=False
        )
        
        result = await workflow.ainvoke(initial_state)
        
        print(f"\n📊 Goal Planning Results:")
        print(f"Goal Hierarchy: {result['goal_hierarchy']}")
        print(f"Extracted Goals: {result['extracted_goals']}")
        print(f"Timeline Analysis: {result['timeline_analysis']}")
        
    except Exception as e:
        logger.error(f"Goal planning workflow example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


if __name__ == "__main__":
    print("🧠 LangGraph + AI Brain Integration Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(run_emotional_workflow_example())
    asyncio.run(run_goal_planning_workflow_example())
    
    print("\n✅ All LangGraph examples completed!")
