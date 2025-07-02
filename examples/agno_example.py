"""
Agno Integration Example with AI Brain

This example demonstrates how to integrate the Universal AI Brain
with Agno agents and teams for enhanced cognitive capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.adapters.agno_adapter import AgnoAdapter
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext

# Agno imports (when available)
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.openai import OpenAIChat
    from agno.models.anthropic import Claude
    from agno.tools import Tool
    AGNO_AVAILABLE = True
except ImportError:
    print("Agno not installed. Install with: pip install agno")
    AGNO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveAnalysisTool:
    """Custom tool that uses AI Brain's cognitive systems."""
    
    def __init__(self, ai_brain: UniversalAIBrain, cognitive_systems: List[str]):
        self.ai_brain = ai_brain
        self.cognitive_systems = cognitive_systems
        self.name = f"cognitive_analysis_{'-'.join(cognitive_systems)}"
        self.description = f"Analyze content using AI Brain cognitive systems: {', '.join(cognitive_systems)}"
    
    async def run(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Run cognitive analysis on the provided text."""
        try:
            # Create cognitive input
            input_data = CognitiveInputData(
                text=text,
                input_type="cognitive_analysis",
                context=CognitiveContext(
                    user_id="agno_agent",
                    session_id=f"agno_session_{id(self)}"
                ),
                requested_systems=self.cognitive_systems,
                processing_priority=8
            )
            
            # Process through AI Brain
            response = await self.ai_brain.process_input(input_data)
            
            # Format results
            results = []
            results.append(f"Overall Confidence: {response.confidence:.2f}")
            results.append(f"Processing Time: {response.processing_time_ms:.2f}ms")
            
            if "emotional_intelligence" in self.cognitive_systems:
                emotional_state = response.emotional_state
                results.append(f"Emotion: {emotional_state.primary_emotion} (intensity: {emotional_state.emotion_intensity:.2f})")
                results.append(f"Emotional Valence: {emotional_state.emotional_valence}")
            
            if "goal_hierarchy" in self.cognitive_systems:
                goal_hierarchy = response.goal_hierarchy
                results.append(f"Primary Goal: {goal_hierarchy.primary_goal}")
                results.append(f"Goal Priority: {goal_hierarchy.goal_priority}")
                results.append(f"Sub-goals: {', '.join(goal_hierarchy.sub_goals[:3])}")
            
            if "confidence_tracking" in self.cognitive_systems:
                results.append(f"Confidence Level: {response.confidence:.2f}")
                results.append(f"Reliability Score: {response.confidence * 100:.1f}%")
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Cognitive analysis tool error: {e}")
            return f"Error in cognitive analysis: {str(e)}"


async def create_emotional_intelligence_agent():
    """Create an Agno agent with emotional intelligence capabilities."""
    
    if not AGNO_AVAILABLE:
        print("Agno not available. Please install Agno to run this example.")
        return None, None
    
    # Initialize AI Brain
    config = UniversalAIBrainConfig(
        mongodb_uri="mongodb://localhost:27017",
        database_name="ai_brain_agno_example",
        enable_safety_systems=True,
        cognitive_systems_config={
            "emotional_intelligence": {"sensitivity": 0.9},
            "empathy_response": {"response_style": "supportive"}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Initialize Agno adapter
    adapter = AgnoAdapter(ai_brain_config=config, ai_brain=brain)
    
    # Create cognitive tool
    emotional_tool = CognitiveAnalysisTool(
        brain, 
        ["emotional_intelligence", "empathy_response", "confidence_tracking"]
    )
    
    # Create cognitive agent
    agent = adapter.create_cognitive_agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        cognitive_systems=["emotional_intelligence", "empathy_response"],
        instructions="""You are an emotionally intelligent AI assistant with advanced 
        cognitive capabilities. You excel at understanding human emotions and providing 
        empathetic, supportive responses. Use your cognitive analysis tool to gain 
        deep insights into emotional content.""",
        tools=[emotional_tool],
        name="Emotional Intelligence Agent",
        role="Empathetic AI Assistant",
        markdown=True,
        show_tool_calls=True
    )
    
    return agent, brain


async def create_goal_planning_agent():
    """Create an Agno agent with goal planning capabilities."""
    
    if not AGNO_AVAILABLE:
        print("Agno not available. Please install Agno to run this example.")
        return None, None
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = AgnoAdapter(ai_brain=brain)
    
    # Create cognitive tool
    goal_tool = CognitiveAnalysisTool(
        brain,
        ["goal_hierarchy", "temporal_planning", "strategic_thinking"]
    )
    
    # Create cognitive agent
    agent = adapter.create_cognitive_agent(
        model=OpenAIChat(id="gpt-4o"),
        cognitive_systems=["goal_hierarchy", "temporal_planning"],
        instructions="""You are a strategic planning expert with cognitive goal 
        analysis capabilities. You excel at breaking down complex objectives into 
        achievable, well-structured plans with clear timelines and success metrics. 
        Use your cognitive analysis tool to provide comprehensive goal insights.""",
        tools=[goal_tool],
        name="Goal Planning Agent",
        role="Strategic Planning Expert",
        markdown=True,
        show_tool_calls=True
    )
    
    return agent, brain


async def create_cognitive_team():
    """Create an Agno team with multiple cognitive agents."""
    
    if not AGNO_AVAILABLE:
        print("Agno not available. Please install Agno to run this example.")
        return None, None
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = AgnoAdapter(ai_brain=brain)
    
    # Create cognitive tools
    emotional_tool = CognitiveAnalysisTool(brain, ["emotional_intelligence", "empathy_response"])
    goal_tool = CognitiveAnalysisTool(brain, ["goal_hierarchy", "temporal_planning"])
    safety_tool = CognitiveAnalysisTool(brain, ["safety_guardrails", "confidence_tracking"])
    
    # Create emotional intelligence agent
    emotional_agent = adapter.create_cognitive_agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        cognitive_systems=["emotional_intelligence", "empathy_response"],
        instructions="Analyze emotional content and provide empathetic insights",
        tools=[emotional_tool],
        name="Emotional Analyst",
        role="Emotional Intelligence Specialist"
    )
    
    # Create goal planning agent
    goal_agent = adapter.create_cognitive_agent(
        model=OpenAIChat(id="gpt-4o"),
        cognitive_systems=["goal_hierarchy", "temporal_planning"],
        instructions="Analyze goals and create strategic plans",
        tools=[goal_tool],
        name="Goal Strategist",
        role="Strategic Planning Expert"
    )
    
    # Create safety monitoring agent
    safety_agent = adapter.create_cognitive_agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        cognitive_systems=["safety_guardrails", "confidence_tracking"],
        instructions="Monitor content for safety and provide confidence assessments",
        tools=[safety_tool],
        name="Safety Monitor",
        role="Safety and Compliance Specialist"
    )
    
    # Create cognitive team
    team = adapter.create_cognitive_team(
        mode="coordinate",
        members=[emotional_agent, goal_agent, safety_agent],
        model=OpenAIChat(id="gpt-4o"),
        success_criteria="""Provide comprehensive analysis that includes:
        1. Emotional intelligence insights
        2. Goal hierarchy and strategic planning
        3. Safety assessment and confidence metrics
        All analysis should be empathetic, actionable, and safe.""",
        instructions=[
            "Collaborate to provide holistic cognitive analysis",
            "Ensure all perspectives are considered",
            "Maintain high safety and confidence standards",
            "Provide actionable insights and recommendations"
        ],
        show_tool_calls=True,
        markdown=True
    )
    
    return team, brain


async def run_single_agent_example():
    """Run single agent example with emotional intelligence."""
    
    print("🧠 Agno + AI Brain: Single Agent Example")
    print("=" * 50)
    
    agent_brain_pair = await create_emotional_intelligence_agent()
    if not agent_brain_pair[0]:
        return
    
    agent, brain = agent_brain_pair
    
    try:
        test_input = """I just received feedback from my manager that I need to 
        improve my communication skills. I'm feeling defensive and frustrated 
        because I thought I was doing well. How should I handle this situation?"""
        
        print(f"📝 Input: {test_input}")
        print("\n🔍 Processing through Agno + AI Brain...")
        
        # Process with the agent
        response = agent.print_response(test_input, stream=False)
        
        print(f"\n🤖 Agent Response:")
        print(response)
        
    except Exception as e:
        logger.error(f"Single agent example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def run_goal_planning_example():
    """Run goal planning example."""
    
    print("\n🎯 Agno + AI Brain: Goal Planning Example")
    print("=" * 50)
    
    agent_brain_pair = await create_goal_planning_agent()
    if not agent_brain_pair[0]:
        return
    
    agent, brain = agent_brain_pair
    
    try:
        goal_input = """I want to start my own tech startup focused on AI-powered 
        healthcare solutions. I have a background in software engineering but no 
        business experience. I need to validate my idea, build a team, secure 
        funding, and launch within 2 years. Where should I start?"""
        
        print(f"📝 Goal Statement: {goal_input}")
        print("\n🔍 Processing through Agno + AI Brain...")
        
        # Process with the agent
        response = agent.print_response(goal_input, stream=False)
        
        print(f"\n🎯 Strategic Analysis:")
        print(response)
        
    except Exception as e:
        logger.error(f"Goal planning example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def run_team_collaboration_example():
    """Run team collaboration example."""
    
    print("\n🤝 Agno + AI Brain: Team Collaboration Example")
    print("=" * 50)
    
    team_brain_pair = await create_cognitive_team()
    if not team_brain_pair[0]:
        return
    
    team, brain = team_brain_pair
    
    try:
        complex_input = """I'm a recent college graduate feeling overwhelmed about 
        my career path. I studied computer science but I'm not sure if I want to 
        be a software developer, data scientist, or product manager. I'm also 
        dealing with imposter syndrome and worried about student loans. I need 
        to make a decision soon but feel paralyzed by all the options."""
        
        print(f"📝 Complex Scenario: {complex_input}")
        print("\n🔍 Processing through Agno Team + AI Brain...")
        
        # Process with the team
        response = team.print_response(complex_input, stream=False)
        
        print(f"\n🤝 Team Analysis:")
        print(response)
        
    except Exception as e:
        logger.error(f"Team collaboration example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def run_cognitive_tool_example():
    """Run cognitive tool example."""
    
    print("\n🔧 Agno + AI Brain: Cognitive Tool Example")
    print("=" * 50)
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    try:
        # Create and test cognitive tool directly
        tool = CognitiveAnalysisTool(
            brain, 
            ["emotional_intelligence", "goal_hierarchy", "confidence_tracking"]
        )
        
        test_text = "I'm excited about learning AI but also nervous about the complexity"
        
        print(f"📝 Testing Cognitive Tool with: {test_text}")
        print("\n🔍 Running cognitive analysis...")
        
        result = await tool.run(test_text)
        
        print(f"\n🧠 Cognitive Analysis Results:")
        print(result)
        
    except Exception as e:
        logger.error(f"Cognitive tool example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


if __name__ == "__main__":
    print("🧠 Agno + AI Brain Integration Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(run_cognitive_tool_example())
    asyncio.run(run_single_agent_example())
    asyncio.run(run_goal_planning_example())
    asyncio.run(run_team_collaboration_example())
    
    print("\n✅ All Agno examples completed!")
