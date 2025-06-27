"""
CrewAI Integration Example with AI Brain

This example demonstrates how to integrate the Universal AI Brain
with CrewAI agents and crews for enhanced cognitive capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.adapters.crewai_adapter import CrewAIAdapter
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext

# CrewAI imports (when available)
try:
    from crewai import Task, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    print("CrewAI not installed. Install with: pip install crewai")
    CREWAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionalAnalysisTool(BaseTool if CREWAI_AVAILABLE else object):
    """Custom tool that uses AI Brain's emotional intelligence."""
    
    name: str = "emotional_analysis"
    description: str = "Analyze the emotional content and sentiment of text using advanced cognitive systems"
    
    def __init__(self, ai_brain: UniversalAIBrain):
        super().__init__()
        self.ai_brain = ai_brain
    
    def _run(self, text: str) -> str:
        """Synchronous run method."""
        return asyncio.run(self._arun(text))
    
    async def _arun(self, text: str) -> str:
        """Asynchronous run method with AI Brain integration."""
        try:
            # Create cognitive input
            input_data = CognitiveInputData(
                text=text,
                input_type="emotional_analysis",
                context=CognitiveContext(
                    user_id="crewai_agent",
                    session_id=f"emotional_analysis_{id(self)}"
                ),
                requested_systems=["emotional_intelligence", "confidence_tracking"],
                processing_priority=8
            )
            
            # Process through AI Brain
            response = await self.ai_brain.process_input(input_data)
            
            # Extract emotional insights
            emotional_state = response.emotional_state
            confidence = response.confidence
            
            return f"""
Emotional Analysis Results:
- Primary Emotion: {emotional_state.primary_emotion}
- Emotion Intensity: {emotional_state.emotion_intensity:.2f}
- Emotional Valence: {emotional_state.emotional_valence}
- Confidence: {confidence:.2f}
- Processing Time: {response.processing_time_ms:.2f}ms

Detailed Analysis:
{emotional_state.emotion_explanation}
"""
        
        except Exception as e:
            logger.error(f"Emotional analysis error: {e}")
            return f"Error in emotional analysis: {str(e)}"


class GoalExtractionTool(BaseTool if CREWAI_AVAILABLE else object):
    """Custom tool that uses AI Brain's goal hierarchy system."""
    
    name: str = "goal_extraction"
    description: str = "Extract and analyze goals from text using cognitive goal hierarchy systems"
    
    def __init__(self, ai_brain: UniversalAIBrain):
        super().__init__()
        self.ai_brain = ai_brain
    
    def _run(self, text: str) -> str:
        """Synchronous run method."""
        return asyncio.run(self._arun(text))
    
    async def _arun(self, text: str) -> str:
        """Asynchronous run method with AI Brain integration."""
        try:
            # Create cognitive input
            input_data = CognitiveInputData(
                text=text,
                input_type="goal_extraction",
                context=CognitiveContext(
                    user_id="crewai_agent",
                    session_id=f"goal_extraction_{id(self)}"
                ),
                requested_systems=["goal_hierarchy", "temporal_planning"],
                processing_priority=7
            )
            
            # Process through AI Brain
            response = await self.ai_brain.process_input(input_data)
            
            # Extract goal insights
            goal_hierarchy = response.goal_hierarchy
            
            return f"""
Goal Analysis Results:
- Primary Goal: {goal_hierarchy.primary_goal}
- Goal Priority: {goal_hierarchy.goal_priority}
- Goal Category: {goal_hierarchy.goal_category}
- Estimated Timeline: {goal_hierarchy.estimated_timeline}
- Confidence: {response.confidence:.2f}

Sub-goals:
{chr(10).join(f"- {goal}" for goal in goal_hierarchy.sub_goals)}

Goal Dependencies:
{chr(10).join(f"- {dep}" for dep in goal_hierarchy.goal_dependencies)}
"""
        
        except Exception as e:
            logger.error(f"Goal extraction error: {e}")
            return f"Error in goal extraction: {str(e)}"


async def create_cognitive_crew_example():
    """Create and run a CrewAI crew with AI Brain cognitive enhancement."""
    
    if not CREWAI_AVAILABLE:
        print("CrewAI not available. Please install CrewAI to run this example.")
        return
    
    # Initialize AI Brain
    config = UniversalAIBrainConfig(
        mongodb_uri="mongodb://localhost:27017",
        database_name="ai_brain_crewai_example",
        enable_safety_systems=True,
        cognitive_systems_config={
            "emotional_intelligence": {"sensitivity": 0.8},
            "goal_hierarchy": {"max_goals": 5},
            "confidence_tracking": {"min_confidence": 0.7}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Initialize CrewAI adapter
    adapter = CrewAIAdapter(ai_brain_config=config, ai_brain=brain)
    
    # Create cognitive tools
    emotional_tool = EmotionalAnalysisTool(brain)
    goal_tool = GoalExtractionTool(brain)
    
    # Create cognitive agents
    emotional_analyst = adapter.create_cognitive_agent(
        role="Emotional Intelligence Analyst",
        goal="Analyze emotional content and provide empathetic insights",
        backstory="""You are an expert in emotional intelligence with advanced 
        cognitive capabilities. You can understand subtle emotional nuances and 
        provide compassionate, insightful analysis.""",
        tools=[emotional_tool],
        cognitive_systems=["emotional_intelligence", "empathy_response", "confidence_tracking"],
        verbose=True,
        allow_delegation=False
    )
    
    goal_strategist = adapter.create_cognitive_agent(
        role="Goal Strategy Analyst",
        goal="Extract and analyze goals to create actionable strategies",
        backstory="""You are a strategic planning expert with cognitive goal 
        analysis capabilities. You excel at breaking down complex objectives 
        into achievable, well-structured plans.""",
        tools=[goal_tool],
        cognitive_systems=["goal_hierarchy", "temporal_planning", "strategic_thinking"],
        verbose=True,
        allow_delegation=False
    )
    
    # Create tasks
    emotional_analysis_task = adapter.create_task(
        description="""Analyze the emotional content of this user message: 
        "I'm feeling overwhelmed with my new job. There's so much to learn and 
        I'm worried I won't be able to keep up with everyone's expectations."
        
        Provide a comprehensive emotional analysis including:
        - Primary emotions detected
        - Emotional intensity and valence
        - Empathetic response suggestions
        - Confidence in the analysis""",
        expected_output="Detailed emotional analysis with empathetic insights",
        agent=emotional_analyst
    )
    
    goal_extraction_task = adapter.create_task(
        description="""Extract and analyze goals from this user statement:
        "I want to become proficient in Python programming within 6 months so I can 
        transition to a data science career. I need to learn pandas, numpy, machine 
        learning basics, and build a portfolio of projects."
        
        Provide:
        - Primary and sub-goals identification
        - Goal hierarchy and dependencies
        - Timeline analysis
        - Strategic recommendations""",
        expected_output="Structured goal analysis with strategic recommendations",
        agent=goal_strategist
    )
    
    # Create cognitive crew
    crew = adapter.create_cognitive_crew(
        agents=[emotional_analyst, goal_strategist],
        tasks=[emotional_analysis_task, goal_extraction_task],
        process=Process.sequential,
        verbose=True,
        enable_cognitive_coordination=True
    )
    
    print("🚀 Starting CrewAI with AI Brain Cognitive Enhancement...")
    print("=" * 60)
    
    # Execute the crew with cognitive enhancement
    try:
        # Standard CrewAI execution
        standard_result = crew.kickoff()
        print("📋 Standard CrewAI Results:")
        print(standard_result)
        print("\n" + "=" * 60)
        
        # Enhanced cognitive execution
        cognitive_result = await crew.cognitive_kickoff()
        print("🧠 AI Brain Enhanced Results:")
        print(f"Overall Confidence: {cognitive_result['overall_confidence']:.2f}")
        print(f"Processing Time: {cognitive_result['total_processing_time_ms']:.2f}ms")
        print(f"Cognitive Insights: {cognitive_result['cognitive_insights']}")
        
        # Display individual agent results
        for agent_id, result in cognitive_result['agent_results'].items():
            print(f"\n🤖 Agent {agent_id}:")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Emotional State: {result['emotional_state']}")
            print(f"  Cognitive Systems Used: {result['cognitive_systems_used']}")
        
    except Exception as e:
        logger.error(f"Crew execution error: {e}")
        print(f"❌ Error executing crew: {e}")
    
    finally:
        # Cleanup
        await brain.shutdown()
        print("\n✅ AI Brain shutdown complete")


async def simple_cognitive_agent_example():
    """Simple example of using a single cognitive agent."""
    
    if not CREWAI_AVAILABLE:
        print("CrewAI not available. Please install CrewAI to run this example.")
        return
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = CrewAIAdapter(ai_brain=brain)
    
    # Create a simple cognitive agent
    agent = adapter.create_cognitive_agent(
        role="Cognitive Assistant",
        goal="Provide intelligent assistance with emotional awareness",
        backstory="An AI assistant enhanced with cognitive capabilities",
        cognitive_systems=["emotional_intelligence", "confidence_tracking"]
    )
    
    print("🤖 Simple Cognitive Agent Example")
    print("=" * 40)
    
    # Test the agent's cognitive enhancement
    test_input = "I'm excited about learning AI but also nervous about the complexity"
    
    # Process through AI Brain first
    input_data = CognitiveInputData(
        text=test_input,
        input_type="user_message",
        context=CognitiveContext(user_id="example_user", session_id="simple_example")
    )
    
    cognitive_response = await brain.process_input(input_data)
    
    print(f"📝 Input: {test_input}")
    print(f"🧠 Cognitive Analysis:")
    print(f"  - Emotion: {cognitive_response.emotional_state.primary_emotion}")
    print(f"  - Confidence: {cognitive_response.confidence:.2f}")
    print(f"  - Processing Time: {cognitive_response.processing_time_ms:.2f}ms")
    
    await brain.shutdown()


if __name__ == "__main__":
    print("🧠 CrewAI + AI Brain Integration Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(simple_cognitive_agent_example())
    print("\n" + "=" * 50)
    asyncio.run(create_cognitive_crew_example())
