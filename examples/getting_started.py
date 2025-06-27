"""
Getting Started with AI Brain Python

This example provides a simple introduction to using the Universal AI Brain
with basic cognitive capabilities.
"""

import asyncio
import logging
from typing import Dict, Any

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_example():
    """Basic example of using the AI Brain."""
    
    print("🧠 AI Brain Python - Getting Started")
    print("=" * 40)
    
    # Step 1: Initialize the AI Brain
    print("1️⃣ Initializing AI Brain...")
    
    brain = UniversalAIBrain()
    await brain.initialize()
    
    print("✅ AI Brain initialized with 16 cognitive systems")
    
    # Step 2: Process some text
    print("\n2️⃣ Processing text through cognitive systems...")
    
    input_text = "I'm excited about learning AI but also nervous about the complexity"
    
    input_data = CognitiveInputData(
        text=input_text,
        input_type="user_message",
        context=CognitiveContext(
            user_id="getting_started_user",
            session_id="basic_example"
        )
    )
    
    response = await brain.process_input(input_data)
    
    # Step 3: Display results
    print(f"\n3️⃣ Results:")
    print(f"📝 Input: {input_text}")
    print(f"😊 Emotion: {response.emotional_state.primary_emotion}")
    print(f"📊 Intensity: {response.emotional_state.emotion_intensity:.2f}")
    print(f"🎯 Goal: {response.goal_hierarchy.primary_goal}")
    print(f"🔍 Confidence: {response.confidence:.2f}")
    print(f"⏱️ Processing Time: {response.processing_time_ms:.2f}ms")
    
    # Step 4: Cleanup
    await brain.shutdown()
    print("\n✅ Example complete!")


async def emotional_intelligence_example():
    """Example focusing on emotional intelligence."""
    
    print("\n😊 Emotional Intelligence Example")
    print("=" * 40)
    
    # Initialize with emotional intelligence focus
    config = UniversalAIBrainConfig(
        cognitive_systems_config={
            "emotional_intelligence": {"sensitivity": 0.9},
            "empathy_response": {"response_style": "supportive"}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Test different emotional scenarios
    scenarios = [
        "I just got promoted and I'm thrilled!",
        "I'm feeling overwhelmed with my workload",
        "I'm disappointed that my project was cancelled",
        "I'm anxious about the presentation tomorrow"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. Analyzing: {scenario}")
        
        input_data = CognitiveInputData(
            text=scenario,
            input_type="emotional_analysis",
            context=CognitiveContext(
                user_id="emotional_user",
                session_id=f"emotion_test_{i}"
            ),
            requested_systems=["emotional_intelligence", "empathy_response"]
        )
        
        response = await brain.process_input(input_data)
        
        print(f"   Emotion: {response.emotional_state.primary_emotion}")
        print(f"   Intensity: {response.emotional_state.emotion_intensity:.2f}")
        print(f"   Empathy: {response.emotional_state.empathy_response}")
    
    await brain.shutdown()


async def goal_planning_example():
    """Example focusing on goal planning."""
    
    print("\n🎯 Goal Planning Example")
    print("=" * 40)
    
    # Initialize with goal planning focus
    config = UniversalAIBrainConfig(
        cognitive_systems_config={
            "goal_hierarchy": {"max_goals": 5},
            "temporal_planning": {"planning_horizon": "medium"}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Test goal scenarios
    goal_statements = [
        "I want to learn Python programming to advance my career",
        "I need to improve my public speaking skills for presentations",
        "I want to start a side business in web development",
        "I plan to get certified in cloud computing within 6 months"
    ]
    
    for i, goal_statement in enumerate(goal_statements, 1):
        print(f"\n{i}. Analyzing goal: {goal_statement}")
        
        input_data = CognitiveInputData(
            text=goal_statement,
            input_type="goal_analysis",
            context=CognitiveContext(
                user_id="goal_user",
                session_id=f"goal_test_{i}"
            ),
            requested_systems=["goal_hierarchy", "temporal_planning"]
        )
        
        response = await brain.process_input(input_data)
        
        print(f"   Primary Goal: {response.goal_hierarchy.primary_goal}")
        print(f"   Priority: {response.goal_hierarchy.goal_priority}/10")
        print(f"   Timeline: {response.goal_hierarchy.estimated_timeline}")
        print(f"   Sub-goals: {', '.join(response.goal_hierarchy.sub_goals[:3])}")
    
    await brain.shutdown()


async def safety_example():
    """Example demonstrating safety features."""
    
    print("\n🛡️ Safety Features Example")
    print("=" * 40)
    
    # Initialize with safety systems enabled
    config = UniversalAIBrainConfig(
        enable_safety_systems=True,
        cognitive_systems_config={
            "safety_guardrails": {"safety_level": "strict"}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Test various inputs including potentially problematic ones
    test_inputs = [
        "This is a normal, safe message",
        "My email is john.doe@example.com and my phone is 555-123-4567",  # PII
        "I'm feeling great about my new project!",  # Safe emotional content
        "Here's some made-up research from 2025 that proves my point"  # Potential hallucination
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{i}. Testing: {test_input[:50]}...")
        
        input_data = CognitiveInputData(
            text=test_input,
            input_type="safety_test",
            context=CognitiveContext(
                user_id="safety_user",
                session_id=f"safety_test_{i}"
            ),
            requested_systems=["safety_guardrails", "confidence_tracking"]
        )
        
        response = await brain.process_input(input_data)
        
        # Check if safety assessment is available
        safety_info = response.cognitive_results.get("safety_guardrails", {})
        
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Safety Status: {safety_info.get('safety_status', 'Unknown')}")
        
        if response.confidence < 0.7:
            print("   ⚠️ Low confidence - review recommended")
    
    await brain.shutdown()


async def custom_configuration_example():
    """Example showing custom configuration options."""
    
    print("\n⚙️ Custom Configuration Example")
    print("=" * 40)
    
    # Create custom configuration
    config = UniversalAIBrainConfig(
        mongodb_uri="mongodb://localhost:27017",
        database_name="ai_brain_custom_example",
        enable_safety_systems=True,
        cognitive_systems_config={
            "emotional_intelligence": {
                "sensitivity": 0.8,
                "enable_empathy_responses": True
            },
            "goal_hierarchy": {
                "max_goals": 8,
                "enable_goal_prioritization": True
            },
            "confidence_tracking": {
                "min_confidence": 0.6,
                "enable_uncertainty_detection": True
            },
            "semantic_memory": {
                "memory_depth": 10,
                "enable_context_retention": True
            }
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    print("✅ AI Brain initialized with custom configuration")
    
    # Test with custom settings
    input_data = CognitiveInputData(
        text="I'm planning to transition my career from marketing to data science",
        input_type="comprehensive_analysis",
        context=CognitiveContext(
            user_id="custom_user",
            session_id="custom_example"
        ),
        requested_systems=[
            "emotional_intelligence",
            "goal_hierarchy", 
            "confidence_tracking",
            "semantic_memory"
        ],
        processing_priority=8
    )
    
    response = await brain.process_input(input_data)
    
    print(f"\n📊 Custom Analysis Results:")
    print(f"Emotion: {response.emotional_state.primary_emotion}")
    print(f"Goal: {response.goal_hierarchy.primary_goal}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Systems Used: {list(response.cognitive_results.keys())}")
    
    # Get system status
    status = brain.get_system_status()
    print(f"\n📈 System Status:")
    print(f"Status: {status['status']}")
    print(f"Active Systems: {len(status['cognitive_systems'])}")
    print(f"Total Requests: {status['total_requests']}")
    
    await brain.shutdown()


async def framework_availability_check():
    """Check which AI frameworks are available."""
    
    print("\n🔍 Framework Availability Check")
    print("=" * 40)
    
    from ai_brain_python import check_framework_availability
    
    frameworks = check_framework_availability()
    
    print("Available AI Frameworks:")
    for framework, version in frameworks.items():
        status = "✅" if version else "❌"
        version_info = f"(v{version})" if version else "(not installed)"
        print(f"  {status} {framework.title()}: {version_info}")
    
    # Show how to get adapters
    print(f"\n📦 Getting Adapters:")
    
    from ai_brain_python import get_adapter
    
    for framework, version in frameworks.items():
        if version:
            try:
                adapter_class = get_adapter(framework)
                print(f"  ✅ {framework}: {adapter_class.__name__}")
            except Exception as e:
                print(f"  ❌ {framework}: Error - {e}")


if __name__ == "__main__":
    print("🚀 AI Brain Python - Getting Started Examples")
    print("=" * 50)
    
    # Run all examples
    asyncio.run(framework_availability_check())
    asyncio.run(basic_example())
    asyncio.run(emotional_intelligence_example())
    asyncio.run(goal_planning_example())
    asyncio.run(safety_example())
    asyncio.run(custom_configuration_example())
    
    print("\n🎉 All getting started examples completed!")
    print("\nNext steps:")
    print("1. Try the framework-specific examples (crewai_example.py, etc.)")
    print("2. Run the comprehensive_example.py for full integration")
    print("3. Check the API documentation for advanced features")
    print("4. Configure MongoDB for persistent memory storage")
