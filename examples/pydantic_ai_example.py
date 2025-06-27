"""
Pydantic AI Integration Example with AI Brain

This example demonstrates how to integrate the Universal AI Brain
with Pydantic AI agents for enhanced cognitive capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.adapters.pydantic_ai_adapter import PydanticAIAdapter
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext

# Pydantic AI imports (when available)
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic import BaseModel, Field
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    print("Pydantic AI not installed. Install with: pip install pydantic-ai")
    PYDANTIC_AI_AVAILABLE = False
    # Fallback classes
    class BaseModel:
        pass
    class RunContext:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionalAnalysisRequest(BaseModel):
    """Request model for emotional analysis."""
    text: str = Field(description="Text to analyze for emotional content")
    context: Optional[str] = Field(default=None, description="Additional context for analysis")
    sensitivity: float = Field(default=0.8, ge=0.0, le=1.0, description="Analysis sensitivity level")


class EmotionalAnalysisResponse(BaseModel):
    """Response model for emotional analysis."""
    primary_emotion: str = Field(description="Primary emotion detected")
    emotion_intensity: float = Field(description="Intensity of the emotion (0.0-1.0)")
    emotional_valence: str = Field(description="Positive, negative, or neutral valence")
    confidence: float = Field(description="Confidence in the analysis (0.0-1.0)")
    empathy_response: str = Field(description="Empathetic response suggestion")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class GoalAnalysisRequest(BaseModel):
    """Request model for goal analysis."""
    text: str = Field(description="Text to analyze for goals")
    time_horizon: Optional[str] = Field(default="medium", description="Short, medium, or long-term focus")
    priority_level: Optional[int] = Field(default=5, ge=1, le=10, description="Priority level (1-10)")


class GoalAnalysisResponse(BaseModel):
    """Response model for goal analysis."""
    primary_goal: str = Field(description="Main goal identified")
    sub_goals: List[str] = Field(description="Supporting sub-goals")
    goal_category: str = Field(description="Category of the goal")
    estimated_timeline: str = Field(description="Estimated timeline for completion")
    goal_priority: int = Field(description="Priority level (1-10)")
    success_metrics: List[str] = Field(description="Suggested success metrics")
    confidence: float = Field(description="Confidence in the analysis")


async def create_emotional_intelligence_agent():
    """Create a Pydantic AI agent with emotional intelligence capabilities."""
    
    if not PYDANTIC_AI_AVAILABLE:
        print("Pydantic AI not available. Please install Pydantic AI to run this example.")
        return None
    
    # Initialize AI Brain
    config = UniversalAIBrainConfig(
        mongodb_uri="mongodb://localhost:27017",
        database_name="ai_brain_pydantic_example",
        enable_safety_systems=True,
        cognitive_systems_config={
            "emotional_intelligence": {"sensitivity": 0.9},
            "empathy_response": {"response_style": "supportive"},
            "confidence_tracking": {"min_confidence": 0.7}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Initialize Pydantic AI adapter
    adapter = PydanticAIAdapter(ai_brain_config=config, ai_brain=brain)
    
    # Create cognitive agent with emotional intelligence
    agent = adapter.create_cognitive_agent(
        model=OpenAIModel('gpt-4o'),
        result_type=EmotionalAnalysisResponse,
        cognitive_systems=["emotional_intelligence", "empathy_response", "confidence_tracking"],
        system_prompt="""You are an emotionally intelligent AI assistant with advanced 
        cognitive capabilities. You excel at understanding human emotions, providing 
        empathetic responses, and offering supportive guidance. Use your cognitive 
        systems to provide deep emotional insights."""
    )
    
    # Register cognitive tools
    @agent.pydantic_agent.tool
    async def analyze_emotional_context(ctx: RunContext, request: EmotionalAnalysisRequest) -> str:
        """Analyze emotional context using AI Brain cognitive systems."""
        try:
            # Create cognitive input
            input_data = CognitiveInputData(
                text=request.text,
                input_type="emotional_analysis",
                context=CognitiveContext(
                    user_id="pydantic_ai_user",
                    session_id=f"emotional_session_{datetime.now().timestamp()}"
                ),
                requested_systems=["emotional_intelligence", "empathy_response"],
                processing_priority=8
            )
            
            # Process through AI Brain
            response = await brain.process_input(input_data)
            
            return f"""
Cognitive Emotional Analysis:
- Primary Emotion: {response.emotional_state.primary_emotion}
- Intensity: {response.emotional_state.emotion_intensity:.2f}
- Valence: {response.emotional_state.emotional_valence}
- Confidence: {response.confidence:.2f}
- Empathy Response: {response.emotional_state.empathy_response}
"""
        
        except Exception as e:
            logger.error(f"Emotional analysis tool error: {e}")
            return f"Error in emotional analysis: {str(e)}"
    
    return agent, brain


async def create_goal_planning_agent():
    """Create a Pydantic AI agent with goal planning capabilities."""
    
    if not PYDANTIC_AI_AVAILABLE:
        print("Pydantic AI not available. Please install Pydantic AI to run this example.")
        return None
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = PydanticAIAdapter(ai_brain=brain)
    
    # Create cognitive agent with goal planning
    agent = adapter.create_cognitive_agent(
        model=OpenAIModel('gpt-4o'),
        result_type=GoalAnalysisResponse,
        cognitive_systems=["goal_hierarchy", "temporal_planning", "strategic_thinking"],
        system_prompt="""You are a strategic planning AI with advanced cognitive 
        goal analysis capabilities. You excel at breaking down complex objectives 
        into achievable plans with clear timelines and success metrics."""
    )
    
    # Register goal analysis tool
    @agent.pydantic_agent.tool
    async def analyze_goal_structure(ctx: RunContext, request: GoalAnalysisRequest) -> str:
        """Analyze goal structure using AI Brain cognitive systems."""
        try:
            # Create cognitive input
            input_data = CognitiveInputData(
                text=request.text,
                input_type="goal_analysis",
                context=CognitiveContext(
                    user_id="pydantic_ai_user",
                    session_id=f"goal_session_{datetime.now().timestamp()}"
                ),
                requested_systems=["goal_hierarchy", "temporal_planning"],
                processing_priority=7
            )
            
            # Process through AI Brain
            response = await brain.process_input(input_data)
            
            return f"""
Cognitive Goal Analysis:
- Primary Goal: {response.goal_hierarchy.primary_goal}
- Goal Category: {response.goal_hierarchy.goal_category}
- Priority: {response.goal_hierarchy.goal_priority}
- Timeline: {response.goal_hierarchy.estimated_timeline}
- Sub-goals: {', '.join(response.goal_hierarchy.sub_goals)}
- Dependencies: {', '.join(response.goal_hierarchy.goal_dependencies)}
- Confidence: {response.confidence:.2f}
"""
        
        except Exception as e:
            logger.error(f"Goal analysis tool error: {e}")
            return f"Error in goal analysis: {str(e)}"
    
    return agent, brain


async def run_emotional_intelligence_example():
    """Run emotional intelligence example with Pydantic AI."""
    
    print("🧠 Pydantic AI + AI Brain: Emotional Intelligence Example")
    print("=" * 60)
    
    agent_brain_pair = await create_emotional_intelligence_agent()
    if not agent_brain_pair:
        return
    
    agent, brain = agent_brain_pair
    
    try:
        # Test emotional analysis
        test_message = """I just got promoted at work, but I'm feeling anxious about 
        the new responsibilities. Part of me is excited about the opportunity, but 
        another part is worried I might not be good enough for the role."""
        
        print(f"📝 Analyzing: {test_message}")
        print("\n🔍 Processing through Pydantic AI + AI Brain...")
        
        # Run the agent
        result = await agent.run(test_message)
        
        print(f"\n📊 Emotional Analysis Results:")
        print(f"Primary Emotion: {result.primary_emotion}")
        print(f"Intensity: {result.emotion_intensity:.2f}")
        print(f"Valence: {result.emotional_valence}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")
        print(f"\n💝 Empathy Response:")
        print(result.empathy_response)
        
    except Exception as e:
        logger.error(f"Emotional intelligence example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def run_goal_planning_example():
    """Run goal planning example with Pydantic AI."""
    
    print("\n🎯 Pydantic AI + AI Brain: Goal Planning Example")
    print("=" * 60)
    
    agent_brain_pair = await create_goal_planning_agent()
    if not agent_brain_pair:
        return
    
    agent, brain = agent_brain_pair
    
    try:
        # Test goal analysis
        goal_statement = """I want to transition from my current marketing role to 
        become a data scientist within the next 18 months. I need to learn Python, 
        statistics, machine learning, and build a portfolio of projects to showcase 
        my skills to potential employers."""
        
        print(f"📝 Analyzing Goal: {goal_statement}")
        print("\n🔍 Processing through Pydantic AI + AI Brain...")
        
        # Run the agent
        result = await agent.run(goal_statement)
        
        print(f"\n📊 Goal Analysis Results:")
        print(f"Primary Goal: {result.primary_goal}")
        print(f"Category: {result.goal_category}")
        print(f"Priority: {result.goal_priority}/10")
        print(f"Timeline: {result.estimated_timeline}")
        print(f"Confidence: {result.confidence:.2f}")
        
        print(f"\n🎯 Sub-goals:")
        for i, sub_goal in enumerate(result.sub_goals, 1):
            print(f"  {i}. {sub_goal}")
        
        print(f"\n📈 Success Metrics:")
        for i, metric in enumerate(result.success_metrics, 1):
            print(f"  {i}. {metric}")
        
    except Exception as e:
        logger.error(f"Goal planning example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def run_multi_agent_conversation():
    """Run a multi-agent conversation example."""
    
    print("\n🤝 Pydantic AI + AI Brain: Multi-Agent Conversation")
    print("=" * 60)
    
    if not PYDANTIC_AI_AVAILABLE:
        print("Pydantic AI not available. Please install Pydantic AI to run this example.")
        return
    
    # Create both agents
    emotional_pair = await create_emotional_intelligence_agent()
    goal_pair = await create_goal_planning_agent()
    
    if not emotional_pair or not goal_pair:
        return
    
    emotional_agent, emotional_brain = emotional_pair
    goal_agent, goal_brain = goal_pair
    
    try:
        user_input = """I'm feeling overwhelmed with my career goals. I want to 
        become a software engineer, but I don't know where to start, and I'm 
        scared I'm too old to make this transition at 35."""
        
        print(f"👤 User: {user_input}")
        
        # First, analyze emotions
        print("\n🧠 Emotional Intelligence Agent analyzing...")
        emotional_result = await emotional_agent.run(user_input)
        
        print(f"😊 Emotional Analysis:")
        print(f"  - Primary Emotion: {emotional_result.primary_emotion}")
        print(f"  - Intensity: {emotional_result.emotion_intensity:.2f}")
        print(f"  - Empathy Response: {emotional_result.empathy_response}")
        
        # Then, analyze goals
        print(f"\n🎯 Goal Planning Agent analyzing...")
        goal_result = await goal_agent.run(user_input)
        
        print(f"📋 Goal Analysis:")
        print(f"  - Primary Goal: {goal_result.primary_goal}")
        print(f"  - Timeline: {goal_result.estimated_timeline}")
        print(f"  - Priority: {goal_result.goal_priority}/10")
        
        print(f"\n🎯 Recommended Sub-goals:")
        for i, sub_goal in enumerate(goal_result.sub_goals, 1):
            print(f"  {i}. {sub_goal}")
        
        # Combine insights
        print(f"\n🤝 Combined AI Brain Insights:")
        print(f"The user is experiencing {emotional_result.primary_emotion} with intensity {emotional_result.emotion_intensity:.2f}")
        print(f"while pursuing the goal: {goal_result.primary_goal}")
        print(f"Recommended approach: Address emotional concerns while creating structured learning plan")
        
    except Exception as e:
        logger.error(f"Multi-agent conversation error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await emotional_brain.shutdown()
        await goal_brain.shutdown()


if __name__ == "__main__":
    print("🧠 Pydantic AI + AI Brain Integration Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(run_emotional_intelligence_example())
    asyncio.run(run_goal_planning_example())
    asyncio.run(run_multi_agent_conversation())
    
    print("\n✅ All examples completed!")
