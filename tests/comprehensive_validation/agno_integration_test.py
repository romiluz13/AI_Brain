"""
Agno Framework Integration Testing with AI Brain

This module tests the integration between the Agno framework and Universal AI Brain,
validating cognitive enhancement capabilities and performance improvements.
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext
from ai_brain_python.adapters.agno_adapter import AgnoAdapter

# Agno framework imports
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.openai import OpenAIChat
    from agno.tools.reasoning import ReasoningTools
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    print("⚠️ Agno not available. Install with: pip install agno")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgnoTestResult:
    """Test result for Agno integration."""
    test_name: str
    baseline_response: str
    enhanced_response: str
    baseline_time_ms: float
    enhanced_time_ms: float
    cognitive_systems_used: List[str]
    improvement_score: float
    success: bool
    error_message: Optional[str] = None


class AgnoIntegrationTester:
    """Test Agno framework integration with AI Brain cognitive capabilities."""
    
    def __init__(self, mongodb_uri: str):
        self.mongodb_uri = mongodb_uri
        self.brain: Optional[UniversalAIBrain] = None
        self.adapter: Optional[AgnoAdapter] = None
        self.baseline_agent: Optional[Agent] = None
        self.enhanced_agent: Optional[Agent] = None
        self.test_results: List[AgnoTestResult] = []
        
        # Test scenarios for Agno integration
        self.test_scenarios = [
            {
                "name": "emotional_intelligence_test",
                "prompt": "I'm feeling overwhelmed with my workload and stressed about meeting deadlines. How can you help me?",
                "expected_cognitive_systems": ["emotional_intelligence", "empathy_response"],
                "evaluation_criteria": ["emotion_detection", "empathy_response", "practical_advice"]
            },
            {
                "name": "goal_planning_test", 
                "prompt": "I want to transition from marketing to data science within 18 months. What's the best approach?",
                "expected_cognitive_systems": ["goal_hierarchy", "temporal_planning", "skill_capability"],
                "evaluation_criteria": ["goal_extraction", "timeline_planning", "skill_assessment"]
            },
            {
                "name": "complex_reasoning_test",
                "prompt": "I'm leading a cross-cultural team with members from Japan, Germany, and Brazil. We have different communication styles and I need to ensure effective collaboration while meeting our project deadline.",
                "expected_cognitive_systems": ["cultural_knowledge", "communication_protocol", "attention_management"],
                "evaluation_criteria": ["cultural_awareness", "communication_strategy", "project_management"]
            },
            {
                "name": "confidence_assessment_test",
                "prompt": "I'm not sure if my approach to this problem is correct. I think it might work, but I'm uncertain about the potential risks.",
                "expected_cognitive_systems": ["confidence_tracking", "strategic_thinking"],
                "evaluation_criteria": ["uncertainty_detection", "risk_assessment", "confidence_calibration"]
            },
            {
                "name": "learning_assistance_test",
                "prompt": "I'm struggling to focus while learning machine learning. There are so many concepts and I keep getting distracted. How can I improve my learning efficiency?",
                "expected_cognitive_systems": ["attention_management", "skill_capability", "self_improvement"],
                "evaluation_criteria": ["attention_analysis", "learning_strategy", "focus_improvement"]
            }
        ]
    
    async def initialize(self) -> bool:
        """Initialize AI Brain and Agno integration."""
        if not AGNO_AVAILABLE:
            logger.error("❌ Agno framework not available")
            return False
        
        try:
            # Initialize AI Brain
            config = UniversalAIBrainConfig(
                mongodb_uri=self.mongodb_uri,
                database_name="ai_brain_agno_test",
                enable_safety_systems=True,
                cognitive_systems_config={
                    "emotional_intelligence": {"sensitivity": 0.8, "enable_empathy_responses": True},
                    "goal_hierarchy": {"max_goals": 8, "enable_prioritization": True},
                    "confidence_tracking": {"min_confidence": 0.6, "enable_uncertainty_detection": True},
                    "attention_management": {"enable_focus_analysis": True, "distraction_detection": True},
                    "cultural_knowledge": {"cultural_adaptation": True, "cultural_sensitivity_level": "high"},
                    "skill_capability": {"skill_assessment": True, "learning_recommendations": True},
                    "communication_protocol": {"style_adaptation": True, "clarity_optimization": True},
                    "temporal_planning": {"timeline_planning": True, "schedule_optimization": True},
                    "strategic_thinking": {"analysis_depth": "comprehensive"},
                    "self_improvement": {"feedback_learning": True, "adaptation_strategies": True}
                }
            )
            
            self.brain = UniversalAIBrain(config)
            await self.brain.initialize()
            logger.info("✅ AI Brain initialized successfully")
            
            # Initialize Agno adapter
            self.adapter = AgnoAdapter(ai_brain_config=config, ai_brain=self.brain)
            logger.info("✅ Agno adapter initialized")
            
            # Create baseline agent (standard Agno without AI Brain)
            self.baseline_agent = Agent(
                model=OpenAIChat(id="gpt-4o"),
                instructions="""You are a helpful AI assistant. Provide clear, concise, and practical responses to user queries. 
                Focus on being helpful and informative while maintaining a professional tone.""",
                markdown=True,
                tools=[ReasoningTools()]
            )
            logger.info("✅ Baseline Agno agent created")
            
            # Create enhanced agent (Agno with AI Brain cognitive capabilities)
            self.enhanced_agent = self.adapter.create_cognitive_agent(
                model=OpenAIChat(id="gpt-4o"),
                cognitive_systems=[
                    "emotional_intelligence",
                    "goal_hierarchy",
                    "confidence_tracking", 
                    "attention_management",
                    "cultural_knowledge",
                    "skill_capability",
                    "communication_protocol",
                    "temporal_planning",
                    "strategic_thinking",
                    "self_improvement"
                ],
                instructions="""You are an emotionally intelligent AI assistant with advanced cognitive capabilities. 
                You can understand emotions, analyze goals, assess confidence levels, and provide culturally sensitive responses.
                Use your cognitive insights to provide more personalized and effective assistance.""",
                cognitive_config={
                    "enable_cognitive_preprocessing": True,
                    "include_confidence_tracking": True,
                    "enable_empathy_responses": True,
                    "cultural_adaptation": True
                }
            )
            logger.info("✅ Enhanced Agno agent with AI Brain created")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def run_baseline_test(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run baseline test with standard Agno agent."""
        try:
            start_time = time.time()
            
            response = await self.baseline_agent.arun(scenario["prompt"])
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            return {
                "response": response,
                "processing_time_ms": processing_time,
                "success": True,
                "response_length": len(response),
                "word_count": len(response.split())
            }
            
        except Exception as e:
            logger.error(f"❌ Baseline test failed for {scenario['name']}: {e}")
            return {
                "response": "",
                "processing_time_ms": 0.0,
                "success": False,
                "error": str(e)
            }
    
    async def run_enhanced_test(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced test with AI Brain cognitive capabilities."""
        try:
            start_time = time.time()
            
            # Run enhanced agent with cognitive capabilities
            response = await self.enhanced_agent.arun(scenario["prompt"])
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            # Get cognitive insights from AI Brain
            cognitive_insights = await self._get_cognitive_insights(scenario["prompt"])
            
            return {
                "response": response,
                "processing_time_ms": processing_time,
                "success": True,
                "response_length": len(response),
                "word_count": len(response.split()),
                "cognitive_insights": cognitive_insights,
                "cognitive_systems_used": list(cognitive_insights.get("cognitive_results", {}).keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced test failed for {scenario['name']}: {e}")
            return {
                "response": "",
                "processing_time_ms": 0.0,
                "success": False,
                "error": str(e),
                "cognitive_systems_used": []
            }
    
    async def _get_cognitive_insights(self, prompt: str) -> Dict[str, Any]:
        """Get cognitive insights from AI Brain for analysis."""
        try:
            input_data = CognitiveInputData(
                text=prompt,
                input_type="agno_integration_test",
                context=CognitiveContext(
                    user_id="agno_test_user",
                    session_id="agno_integration_session",
                    timestamp=datetime.utcnow()
                ),
                processing_priority=8
            )
            
            response = await self.brain.process_input(input_data)
            
            return {
                "confidence": response.confidence,
                "processing_time_ms": response.processing_time_ms,
                "cognitive_results": response.cognitive_results,
                "emotional_state": {
                    "primary_emotion": response.emotional_state.primary_emotion,
                    "emotion_intensity": response.emotional_state.emotion_intensity,
                    "empathy_response": response.emotional_state.empathy_response
                },
                "goal_hierarchy": {
                    "primary_goal": response.goal_hierarchy.primary_goal,
                    "goal_priority": response.goal_hierarchy.goal_priority,
                    "sub_goals": response.goal_hierarchy.sub_goals
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get cognitive insights: {e}")
            return {}
    
    def _calculate_improvement_score(self, baseline: Dict, enhanced: Dict, scenario: Dict) -> float:
        """Calculate improvement score based on multiple factors."""
        if not baseline.get("success") or not enhanced.get("success"):
            return 0.0
        
        score = 0.0
        
        # Response quality (length and depth)
        baseline_words = baseline.get("word_count", 0)
        enhanced_words = enhanced.get("word_count", 0)
        
        if baseline_words > 0:
            word_improvement = min((enhanced_words - baseline_words) / baseline_words, 1.0)
            score += word_improvement * 20  # Up to 20 points for response depth
        
        # Cognitive system utilization
        cognitive_systems_used = enhanced.get("cognitive_systems_used", [])
        expected_systems = scenario.get("expected_cognitive_systems", [])
        
        if expected_systems:
            system_coverage = len(set(cognitive_systems_used) & set(expected_systems)) / len(expected_systems)
            score += system_coverage * 30  # Up to 30 points for cognitive system coverage
        
        # Cognitive insights quality
        cognitive_insights = enhanced.get("cognitive_insights", {})
        if cognitive_insights:
            confidence = cognitive_insights.get("confidence", 0.0)
            score += confidence * 25  # Up to 25 points for confidence
            
            # Emotional intelligence bonus
            emotional_state = cognitive_insights.get("emotional_state", {})
            if emotional_state.get("primary_emotion") and emotional_state.get("empathy_response"):
                score += 15  # 15 points for emotional intelligence
            
            # Goal analysis bonus
            goal_hierarchy = cognitive_insights.get("goal_hierarchy", {})
            if goal_hierarchy.get("primary_goal") and goal_hierarchy.get("sub_goals"):
                score += 10  # 10 points for goal analysis
        
        return min(score, 100.0)  # Cap at 100
    
    async def run_comprehensive_tests(self) -> List[AgnoTestResult]:
        """Run comprehensive tests comparing baseline vs enhanced Agno agents."""
        logger.info("🚀 Running comprehensive Agno integration tests...")
        
        test_results = []
        
        for scenario in self.test_scenarios:
            logger.info(f"🧪 Testing scenario: {scenario['name']}")
            
            # Run baseline test
            baseline_result = await self.run_baseline_test(scenario)
            
            # Run enhanced test
            enhanced_result = await self.run_enhanced_test(scenario)
            
            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(baseline_result, enhanced_result, scenario)
            
            # Create test result
            test_result = AgnoTestResult(
                test_name=scenario["name"],
                baseline_response=baseline_result.get("response", ""),
                enhanced_response=enhanced_result.get("response", ""),
                baseline_time_ms=baseline_result.get("processing_time_ms", 0.0),
                enhanced_time_ms=enhanced_result.get("processing_time_ms", 0.0),
                cognitive_systems_used=enhanced_result.get("cognitive_systems_used", []),
                improvement_score=improvement_score,
                success=baseline_result.get("success", False) and enhanced_result.get("success", False),
                error_message=enhanced_result.get("error") or baseline_result.get("error")
            )
            
            test_results.append(test_result)
            
            if test_result.success:
                logger.info(f"✅ {scenario['name']}: {improvement_score:.1f}% improvement score")
            else:
                logger.error(f"❌ {scenario['name']}: Test failed")
        
        self.test_results = test_results
        return test_results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_results:
            return {"error": "No test results available"}
        
        successful_tests = [t for t in self.test_results if t.success]
        
        if successful_tests:
            avg_improvement = sum(t.improvement_score for t in successful_tests) / len(successful_tests)
            max_improvement = max(t.improvement_score for t in successful_tests)
            min_improvement = min(t.improvement_score for t in successful_tests)
            avg_baseline_time = sum(t.baseline_time_ms for t in successful_tests) / len(successful_tests)
            avg_enhanced_time = sum(t.enhanced_time_ms for t in successful_tests) / len(successful_tests)
        else:
            avg_improvement = max_improvement = min_improvement = 0.0
            avg_baseline_time = avg_enhanced_time = 0.0
        
        # Analyze cognitive system usage
        all_systems_used = set()
        for test in successful_tests:
            all_systems_used.update(test.cognitive_systems_used)
        
        return {
            "test_timestamp": datetime.utcnow().isoformat(),
            "total_tests": len(self.test_results),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(self.test_results) if self.test_results else 0.0,
            "average_improvement_score": avg_improvement,
            "max_improvement_score": max_improvement,
            "min_improvement_score": min_improvement,
            "average_baseline_time_ms": avg_baseline_time,
            "average_enhanced_time_ms": avg_enhanced_time,
            "cognitive_systems_utilized": list(all_systems_used),
            "tests_meeting_90_percent": len([t for t in successful_tests if t.improvement_score >= 90.0]),
            "detailed_results": [
                {
                    "test_name": t.test_name,
                    "improvement_score": t.improvement_score,
                    "cognitive_systems": t.cognitive_systems_used,
                    "baseline_time_ms": t.baseline_time_ms,
                    "enhanced_time_ms": t.enhanced_time_ms,
                    "success": t.success
                }
                for t in self.test_results
            ]
        }
    
    def print_test_summary(self, report: Dict[str, Any]):
        """Print formatted test summary."""
        print("\n" + "="*70)
        print("🤖 AGNO FRAMEWORK + AI BRAIN INTEGRATION TEST RESULTS")
        print("="*70)
        
        print(f"📅 Test Time: {report.get('test_timestamp', 'Unknown')}")
        print(f"🧪 Total Tests: {report.get('total_tests', 0)}")
        print(f"✅ Successful Tests: {report.get('successful_tests', 0)}")
        print(f"📊 Success Rate: {report.get('success_rate', 0.0):.1%}")
        print(f"🎯 Average Improvement: {report.get('average_improvement_score', 0.0):.1f}%")
        print(f"🏆 Max Improvement: {report.get('max_improvement_score', 0.0):.1f}%")
        print(f"⏱️ Avg Baseline Time: {report.get('average_baseline_time_ms', 0.0):.1f}ms")
        print(f"⏱️ Avg Enhanced Time: {report.get('average_enhanced_time_ms', 0.0):.1f}ms")
        
        print(f"\n🧠 COGNITIVE SYSTEMS UTILIZED:")
        print("-" * 70)
        for system in report.get('cognitive_systems_utilized', []):
            print(f"✅ {system}")
        
        print(f"\n📋 DETAILED TEST RESULTS:")
        print("-" * 70)
        
        for test in report.get('detailed_results', []):
            status = "✅" if test['success'] else "❌"
            score = test['improvement_score']
            print(f"{status} {test['test_name']:30} | {score:6.1f}% | {len(test['cognitive_systems'])} systems")
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        print("-" * 70)
        
        avg_improvement = report.get('average_improvement_score', 0.0)
        tests_90_percent = report.get('tests_meeting_90_percent', 0)
        total_tests = report.get('successful_tests', 0)
        
        criteria_met = avg_improvement >= 90.0
        print(f"{'✅' if criteria_met else '❌'} Average 90% Improvement: {avg_improvement:.1f}%")
        
        majority_met = tests_90_percent >= (total_tests * 0.8) if total_tests > 0 else False
        print(f"{'✅' if majority_met else '❌'} 80% Tests Meeting 90% Target: {tests_90_percent}/{total_tests}")
        
        print("="*70)
        
        if criteria_met and majority_met:
            print("🎉 AGNO INTEGRATION SUCCESS - AI BRAIN ENHANCEMENT VALIDATED!")
        else:
            print("⚠️ Some criteria not met - review individual test performance")
        
        print("="*70)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.brain:
            await self.brain.shutdown()


async def main():
    """Main function to run Agno integration tests."""
    mongodb_uri = "mongodb+srv://romiluz:H97r3aQBnxWawZbx@aibrain.tnv45wr.mongodb.net/?retryWrites=true&w=majority&appName=aibrain"
    
    tester = AgnoIntegrationTester(mongodb_uri)
    
    try:
        if await tester.initialize():
            test_results = await tester.run_comprehensive_tests()
            report = tester.generate_test_report()
            tester.print_test_summary(report)
            return report
        else:
            print("❌ Failed to initialize Agno integration tester")
            return None
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    if not AGNO_AVAILABLE:
        print("❌ Agno framework not available. Install with: pip install agno")
    else:
        asyncio.run(main())
