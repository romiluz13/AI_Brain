"""
Comprehensive AI Brain Integration Example

This example demonstrates the Universal AI Brain working with all supported
frameworks in a unified cognitive architecture.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext
from ai_brain_python.safety import IntegratedSafetySystem, SafetySystemConfig

# Framework availability checks
FRAMEWORKS_AVAILABLE = {}

try:
    from ai_brain_python.adapters.crewai_adapter import CrewAIAdapter
    FRAMEWORKS_AVAILABLE["crewai"] = True
except ImportError:
    FRAMEWORKS_AVAILABLE["crewai"] = False

try:
    from ai_brain_python.adapters.pydantic_ai_adapter import PydanticAIAdapter
    FRAMEWORKS_AVAILABLE["pydantic_ai"] = True
except ImportError:
    FRAMEWORKS_AVAILABLE["pydantic_ai"] = False

try:
    from ai_brain_python.adapters.agno_adapter import AgnoAdapter
    FRAMEWORKS_AVAILABLE["agno"] = True
except ImportError:
    FRAMEWORKS_AVAILABLE["agno"] = False

try:
    from ai_brain_python.adapters.langchain_adapter import LangChainAdapter
    FRAMEWORKS_AVAILABLE["langchain"] = True
except ImportError:
    FRAMEWORKS_AVAILABLE["langchain"] = False

try:
    from ai_brain_python.adapters.langgraph_adapter import LangGraphAdapter
    FRAMEWORKS_AVAILABLE["langgraph"] = True
except ImportError:
    FRAMEWORKS_AVAILABLE["langgraph"] = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveOrchestrator:
    """
    Orchestrates multiple AI frameworks with shared AI Brain cognitive capabilities.
    """
    
    def __init__(self):
        self.brain: Optional[UniversalAIBrain] = None
        self.safety_system: Optional[IntegratedSafetySystem] = None
        self.adapters: Dict[str, Any] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the cognitive orchestrator with all available frameworks."""
        
        print("🧠 Initializing Comprehensive AI Brain System")
        print("=" * 60)
        
        # Initialize AI Brain with comprehensive configuration
        config = UniversalAIBrainConfig(
            mongodb_uri="mongodb://localhost:27017",
            database_name="ai_brain_comprehensive_example",
            enable_safety_systems=True,
            cognitive_systems_config={
                "emotional_intelligence": {"sensitivity": 0.8},
                "goal_hierarchy": {"max_goals": 10},
                "confidence_tracking": {"min_confidence": 0.7},
                "safety_guardrails": {"safety_level": "moderate"},
                "semantic_memory": {"memory_depth": 5},
                "temporal_planning": {"planning_horizon": "medium"},
                "empathy_response": {"response_style": "supportive"},
                "strategic_thinking": {"analysis_depth": "comprehensive"}
            }
        )
        
        self.brain = UniversalAIBrain(config)
        await self.brain.initialize()
        print("✅ AI Brain initialized with 16 cognitive systems")
        
        # Initialize safety system
        safety_config = SafetySystemConfig()
        self.safety_system = IntegratedSafetySystem(safety_config)
        await self.safety_system.initialize()
        print("✅ Integrated Safety System initialized")
        
        # Initialize available framework adapters
        await self._initialize_adapters(config)
        
        self.initialized = True
        print(f"✅ Cognitive Orchestrator initialized with {len(self.adapters)} frameworks")
    
    async def _initialize_adapters(self, config: UniversalAIBrainConfig):
        """Initialize all available framework adapters."""
        
        if FRAMEWORKS_AVAILABLE["crewai"]:
            self.adapters["crewai"] = CrewAIAdapter(ai_brain_config=config, ai_brain=self.brain)
            print("✅ CrewAI adapter initialized")
        
        if FRAMEWORKS_AVAILABLE["pydantic_ai"]:
            self.adapters["pydantic_ai"] = PydanticAIAdapter(ai_brain_config=config, ai_brain=self.brain)
            print("✅ Pydantic AI adapter initialized")
        
        if FRAMEWORKS_AVAILABLE["agno"]:
            self.adapters["agno"] = AgnoAdapter(ai_brain_config=config, ai_brain=self.brain)
            print("✅ Agno adapter initialized")
        
        if FRAMEWORKS_AVAILABLE["langchain"]:
            self.adapters["langchain"] = LangChainAdapter(ai_brain_config=config, ai_brain=self.brain)
            print("✅ LangChain adapter initialized")
        
        if FRAMEWORKS_AVAILABLE["langgraph"]:
            self.adapters["langgraph"] = LangGraphAdapter(ai_brain_config=config, ai_brain=self.brain)
            print("✅ LangGraph adapter initialized")
    
    async def comprehensive_analysis(
        self, 
        text: str, 
        user_id: str = "comprehensive_user",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using all cognitive systems and safety checks.
        """
        
        if not self.initialized:
            await self.initialize()
        
        session_id = session_id or f"comprehensive_{datetime.now().timestamp()}"
        
        print(f"\n🔍 Comprehensive Cognitive Analysis")
        print(f"📝 Input: {text}")
        print("=" * 60)
        
        # Step 1: Safety Check
        print("🛡️ Step 1: Safety Assessment...")
        safety_result = await self.safety_system.comprehensive_safety_check(
            text=text,
            user_id=user_id,
            session_id=session_id
        )
        
        if not safety_result["overall_safe"]:
            print("⚠️ Safety concerns detected!")
            return {
                "safe": False,
                "safety_result": safety_result,
                "message": "Content flagged by safety systems"
            }
        
        print("✅ Content passed safety checks")
        
        # Step 2: Core Cognitive Analysis
        print("\n🧠 Step 2: Core Cognitive Analysis...")
        
        input_data = CognitiveInputData(
            text=text,
            input_type="comprehensive_analysis",
            context=CognitiveContext(
                user_id=user_id,
                session_id=session_id
            ),
            requested_systems=[
                "emotional_intelligence",
                "goal_hierarchy", 
                "confidence_tracking",
                "semantic_memory",
                "temporal_planning",
                "empathy_response",
                "strategic_thinking",
                "attention_management"
            ],
            processing_priority=9
        )
        
        cognitive_response = await self.brain.process_input(input_data)
        
        print(f"✅ Cognitive analysis complete (confidence: {cognitive_response.confidence:.2f})")
        
        # Step 3: Framework-Specific Enhancements
        print("\n🔧 Step 3: Framework-Specific Enhancements...")
        
        framework_results = {}
        
        # Test each available framework
        for framework_name, adapter in self.adapters.items():
            try:
                framework_result = await self._test_framework(framework_name, adapter, text)
                framework_results[framework_name] = framework_result
                print(f"✅ {framework_name} analysis complete")
            except Exception as e:
                logger.error(f"Framework {framework_name} error: {e}")
                framework_results[framework_name] = {"error": str(e)}
                print(f"❌ {framework_name} analysis failed: {e}")
        
        # Step 4: Synthesis and Recommendations
        print("\n🎯 Step 4: Synthesis and Recommendations...")
        
        recommendations = self._generate_comprehensive_recommendations(
            cognitive_response, framework_results, safety_result
        )
        
        return {
            "safe": True,
            "cognitive_analysis": {
                "emotional_state": {
                    "primary_emotion": cognitive_response.emotional_state.primary_emotion,
                    "emotion_intensity": cognitive_response.emotional_state.emotion_intensity,
                    "emotional_valence": cognitive_response.emotional_state.emotional_valence,
                    "empathy_response": cognitive_response.emotional_state.empathy_response
                },
                "goal_hierarchy": {
                    "primary_goal": cognitive_response.goal_hierarchy.primary_goal,
                    "goal_priority": cognitive_response.goal_hierarchy.goal_priority,
                    "sub_goals": cognitive_response.goal_hierarchy.sub_goals,
                    "estimated_timeline": cognitive_response.goal_hierarchy.estimated_timeline
                },
                "confidence": cognitive_response.confidence,
                "processing_time_ms": cognitive_response.processing_time_ms
            },
            "framework_results": framework_results,
            "safety_assessment": safety_result,
            "recommendations": recommendations,
            "session_id": session_id
        }
    
    async def _test_framework(self, framework_name: str, adapter: Any, text: str) -> Dict[str, Any]:
        """Test a specific framework with the given text."""
        
        if framework_name == "crewai":
            # Create a simple cognitive agent for testing
            agent = adapter.create_cognitive_agent(
                role="Cognitive Analyst",
                goal="Analyze content with cognitive capabilities",
                backstory="An AI agent enhanced with cognitive systems",
                cognitive_systems=["emotional_intelligence", "confidence_tracking"]
            )
            return {"agent_created": True, "cognitive_systems": ["emotional_intelligence", "confidence_tracking"]}
        
        elif framework_name == "pydantic_ai":
            # Create a cognitive agent
            agent = adapter.create_cognitive_agent(
                model="gpt-4o",
                cognitive_systems=["emotional_intelligence", "goal_hierarchy"],
                system_prompt="Analyze content with cognitive capabilities"
            )
            return {"agent_created": True, "cognitive_systems": ["emotional_intelligence", "goal_hierarchy"]}
        
        elif framework_name == "agno":
            # Create a cognitive agent
            from agno.models.openai import OpenAIChat
            agent = adapter.create_cognitive_agent(
                model=OpenAIChat(id="gpt-4o"),
                cognitive_systems=["emotional_intelligence", "strategic_thinking"],
                instructions="Analyze content with cognitive capabilities"
            )
            return {"agent_created": True, "cognitive_systems": ["emotional_intelligence", "strategic_thinking"]}
        
        elif framework_name == "langchain":
            # Create cognitive tools
            tool = adapter.create_cognitive_tool(
                name="cognitive_analysis",
                description="Analyze content using AI Brain",
                ai_brain=self.brain,
                cognitive_systems=["emotional_intelligence", "goal_hierarchy"]
            )
            return {"tool_created": True, "cognitive_systems": ["emotional_intelligence", "goal_hierarchy"]}
        
        elif framework_name == "langgraph":
            # Create cognitive graph
            from typing import TypedDict
            
            class TestState(TypedDict):
                input_text: str
                analysis_complete: bool
            
            graph = adapter.create_cognitive_graph(
                ai_brain=self.brain,
                state_schema=TestState
            )
            return {"graph_created": True, "cognitive_systems": ["all_available"]}
        
        return {"status": "tested"}
    
    def _generate_comprehensive_recommendations(
        self, 
        cognitive_response, 
        framework_results: Dict[str, Any], 
        safety_result: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive recommendations based on all analyses."""
        
        recommendations = []
        
        # Emotional recommendations
        emotion = cognitive_response.emotional_state.primary_emotion
        intensity = cognitive_response.emotional_state.emotion_intensity
        
        if intensity > 0.7:
            recommendations.append(f"Address high-intensity {emotion} with appropriate support")
        
        # Goal-based recommendations
        primary_goal = cognitive_response.goal_hierarchy.primary_goal
        if primary_goal and primary_goal != "No specific goal detected":
            recommendations.append(f"Focus on achieving: {primary_goal}")
        
        # Confidence-based recommendations
        if cognitive_response.confidence < 0.7:
            recommendations.append("Consider additional validation due to lower confidence")
        
        # Framework-specific recommendations
        available_frameworks = [name for name, result in framework_results.items() if "error" not in result]
        if available_frameworks:
            recommendations.append(f"Leverage {', '.join(available_frameworks)} for enhanced capabilities")
        
        # Safety recommendations
        if safety_result.get("recommendations"):
            recommendations.extend(safety_result["recommendations"])
        
        return recommendations
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        if not self.initialized:
            return {"status": "not_initialized"}
        
        # Get AI Brain status
        brain_status = self.brain.get_system_status()
        
        # Get safety system status
        safety_dashboard = await self.safety_system.get_safety_dashboard()
        
        # Get framework availability
        framework_status = {
            name: {"available": available, "adapter_initialized": name in self.adapters}
            for name, available in FRAMEWORKS_AVAILABLE.items()
        }
        
        return {
            "status": "operational",
            "ai_brain": brain_status,
            "safety_system": safety_dashboard["health_status"],
            "frameworks": framework_status,
            "total_frameworks_available": sum(FRAMEWORKS_AVAILABLE.values()),
            "total_adapters_initialized": len(self.adapters)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the cognitive orchestrator."""
        
        if self.brain:
            await self.brain.shutdown()
        
        if self.safety_system:
            await self.safety_system.shutdown()
        
        print("✅ Cognitive Orchestrator shutdown complete")


async def run_comprehensive_example():
    """Run the comprehensive example with all frameworks."""
    
    print("🚀 Universal AI Brain - Comprehensive Framework Integration")
    print("=" * 70)
    
    # Initialize orchestrator
    orchestrator = CognitiveOrchestrator()
    
    try:
        # Test scenarios
        test_scenarios = [
            {
                "name": "Career Transition Anxiety",
                "text": """I'm feeling overwhelmed about transitioning from my marketing 
                career to data science. I want to make this change within 18 months, 
                but I'm worried about the technical challenges and whether I'm making 
                the right decision at age 35."""
            },
            {
                "name": "Leadership Challenge",
                "text": """I just got promoted to team lead, but I'm experiencing imposter 
                syndrome. I want to build confidence and develop my leadership skills 
                while maintaining good relationships with my former peers."""
            },
            {
                "name": "Learning Goal",
                "text": """I'm excited about learning AI and machine learning, but the 
                field seems so vast. I need help creating a structured learning path 
                that fits my schedule as a working parent."""
            }
        ]
        
        # Run comprehensive analysis for each scenario
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{'='*70}")
            print(f"🎯 Scenario {i}: {scenario['name']}")
            print(f"{'='*70}")
            
            result = await orchestrator.comprehensive_analysis(
                text=scenario["text"],
                user_id=f"test_user_{i}",
                session_id=f"scenario_{i}"
            )
            
            if result["safe"]:
                print(f"\n📊 Analysis Results:")
                print(f"Primary Emotion: {result['cognitive_analysis']['emotional_state']['primary_emotion']}")
                print(f"Emotion Intensity: {result['cognitive_analysis']['emotional_state']['emotion_intensity']:.2f}")
                print(f"Primary Goal: {result['cognitive_analysis']['goal_hierarchy']['primary_goal']}")
                print(f"Overall Confidence: {result['cognitive_analysis']['confidence']:.2f}")
                
                print(f"\n🎯 Recommendations:")
                for j, rec in enumerate(result["recommendations"], 1):
                    print(f"  {j}. {rec}")
                
                print(f"\n🔧 Framework Results:")
                for framework, result_data in result["framework_results"].items():
                    status = "✅" if "error" not in result_data else "❌"
                    print(f"  {status} {framework}: {result_data}")
            else:
                print(f"⚠️ Safety concerns detected: {result['message']}")
        
        # Show system status
        print(f"\n{'='*70}")
        print("📊 System Status")
        print(f"{'='*70}")
        
        status = await orchestrator.get_system_status()
        print(f"Overall Status: {status['status']}")
        print(f"Frameworks Available: {status['total_frameworks_available']}")
        print(f"Adapters Initialized: {status['total_adapters_initialized']}")
        print(f"AI Brain Status: {status['ai_brain']['status']}")
        print(f"Safety System: {status['safety_system']['status']}")
        
    except Exception as e:
        logger.error(f"Comprehensive example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    print("🧠 Universal AI Brain - Comprehensive Integration Example")
    print("=" * 60)
    print(f"Available Frameworks: {[name for name, available in FRAMEWORKS_AVAILABLE.items() if available]}")
    print("=" * 60)
    
    # Run the comprehensive example
    asyncio.run(run_comprehensive_example())
