"""
LangChain Integration Example with AI Brain

This example demonstrates how to integrate the Universal AI Brain
with LangChain chains, tools, and memory for enhanced cognitive capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.adapters.langchain_adapter import LangChainAdapter
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext

# LangChain imports (when available)
try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_openai import ChatOpenAI
    from langchain.chains.llm import LLMChain
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("LangChain not installed. Install with: pip install langchain langchain-openai")
    LANGCHAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_cognitive_memory_example():
    """Create and demonstrate cognitive memory with LangChain."""
    
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available. Please install LangChain to run this example.")
        return
    
    print("🧠 LangChain + AI Brain: Cognitive Memory Example")
    print("=" * 55)
    
    # Initialize AI Brain
    config = UniversalAIBrainConfig(
        mongodb_uri="mongodb://localhost:27017",
        database_name="ai_brain_langchain_example",
        enable_safety_systems=True,
        cognitive_systems_config={
            "semantic_memory": {"memory_depth": 5},
            "emotional_intelligence": {"sensitivity": 0.8}
        }
    )
    
    brain = UniversalAIBrain(config)
    await brain.initialize()
    
    # Initialize LangChain adapter
    adapter = LangChainAdapter(ai_brain_config=config, ai_brain=brain)
    
    try:
        # Create cognitive memory
        memory = adapter.create_cognitive_memory(
            ai_brain=brain,
            memory_key="chat_history",
            cognitive_systems=["semantic_memory", "emotional_intelligence"]
        )
        
        # Test memory operations
        print("📝 Testing Cognitive Memory Operations...")
        
        # Save some context
        inputs = {"input": "I'm feeling excited about learning AI"}
        outputs = {"output": "That's wonderful! Your excitement will help you learn faster."}
        
        memory.save_context(inputs, outputs)
        print(f"💾 Saved context: {inputs['input']} -> {outputs['output']}")
        
        # Load memory variables
        loaded_memory = memory.load_memory_variables({"input": "Tell me about my previous emotions"})
        print(f"🔍 Loaded memory: {loaded_memory}")
        
        # Test cognitive enhancement
        enhanced_memory = await memory.get_cognitive_context(
            current_input="How am I feeling now?",
            cognitive_systems=["emotional_intelligence", "semantic_memory"]
        )
        print(f"🧠 Cognitive context: {enhanced_memory}")
        
    except Exception as e:
        logger.error(f"Cognitive memory example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def create_cognitive_tools_example():
    """Create and demonstrate cognitive tools with LangChain."""
    
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available. Please install LangChain to run this example.")
        return
    
    print("\n🔧 LangChain + AI Brain: Cognitive Tools Example")
    print("=" * 55)
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = LangChainAdapter(ai_brain=brain)
    
    try:
        # Create emotional analysis tool
        emotional_tool = adapter.create_cognitive_tool(
            name="emotional_analysis",
            description="Analyze emotional content using AI Brain cognitive systems",
            ai_brain=brain,
            cognitive_systems=["emotional_intelligence", "empathy_response"]
        )
        
        # Create goal analysis tool
        goal_tool = adapter.create_cognitive_tool(
            name="goal_analysis",
            description="Analyze goals and create strategic plans using AI Brain",
            ai_brain=brain,
            cognitive_systems=["goal_hierarchy", "temporal_planning"]
        )
        
        # Test emotional analysis tool
        print("😊 Testing Emotional Analysis Tool...")
        emotional_result = await emotional_tool._arun(
            "I'm feeling overwhelmed with my workload but excited about the new project"
        )
        print(f"Result: {emotional_result}")
        
        # Test goal analysis tool
        print("\n🎯 Testing Goal Analysis Tool...")
        goal_result = await goal_tool._arun(
            "I want to become a machine learning engineer within 2 years"
        )
        print(f"Result: {goal_result}")
        
    except Exception as e:
        logger.error(f"Cognitive tools example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def create_cognitive_chain_example():
    """Create and demonstrate cognitive chains with LangChain."""
    
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available. Please install LangChain to run this example.")
        return
    
    print("\n⛓️ LangChain + AI Brain: Cognitive Chain Example")
    print("=" * 55)
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = LangChainAdapter(ai_brain=brain)
    
    try:
        # Create LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Create cognitive memory
        memory = adapter.create_cognitive_memory(
            ai_brain=brain,
            memory_key="chat_history"
        )
        
        # Create prompt template
        prompt = PromptTemplate(
            template="""You are an emotionally intelligent AI assistant with cognitive capabilities.
            
            Chat History: {chat_history}
            Cognitive Context: {cognitive_context}
            
            Human: {input}
            
            Provide a response that demonstrates emotional intelligence and strategic thinking:""",
            input_variables=["chat_history", "cognitive_context", "input"]
        )
        
        # Create cognitive chain
        chain = adapter.create_cognitive_chain(
            llm=llm,
            prompt=prompt,
            ai_brain=brain,
            memory=memory,
            cognitive_systems=["emotional_intelligence", "goal_hierarchy", "confidence_tracking"]
        )
        
        # Test the chain
        print("🔗 Testing Cognitive Chain...")
        
        test_inputs = [
            "I'm starting a new job next week and I'm nervous about fitting in",
            "I want to set some professional goals for this year",
            "How can I build confidence in my new role?"
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n📝 Input {i}: {test_input}")
            
            result = await chain.arun(input=test_input)
            print(f"🤖 Response: {result}")
            
            # Show cognitive insights
            cognitive_insights = await chain.get_cognitive_insights()
            print(f"🧠 Cognitive Insights: {cognitive_insights}")
        
    except Exception as e:
        logger.error(f"Cognitive chain example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def create_lcel_cognitive_example():
    """Create LCEL (LangChain Expression Language) example with cognitive enhancement."""
    
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available. Please install LangChain to run this example.")
        return
    
    print("\n🔗 LangChain + AI Brain: LCEL Cognitive Example")
    print("=" * 55)
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    try:
        # Create LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Create cognitive enhancement function
        async def cognitive_enhancement(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Enhance inputs with cognitive analysis."""
            try:
                # Create cognitive input
                input_data = CognitiveInputData(
                    text=inputs["input"],
                    input_type="lcel_enhancement",
                    context=CognitiveContext(
                        user_id="langchain_user",
                        session_id="lcel_session"
                    ),
                    requested_systems=["emotional_intelligence", "goal_hierarchy", "confidence_tracking"],
                    processing_priority=7
                )
                
                # Process through AI Brain
                response = await brain.process_input(input_data)
                
                # Add cognitive context
                inputs["cognitive_analysis"] = {
                    "emotion": response.emotional_state.primary_emotion,
                    "emotion_intensity": response.emotional_state.emotion_intensity,
                    "primary_goal": response.goal_hierarchy.primary_goal,
                    "confidence": response.confidence
                }
                
                inputs["cognitive_prompt_addition"] = f"""
                
Cognitive Analysis:
- Detected emotion: {response.emotional_state.primary_emotion} (intensity: {response.emotional_state.emotion_intensity:.2f})
- Primary goal: {response.goal_hierarchy.primary_goal}
- Confidence level: {response.confidence:.2f}

Please incorporate this cognitive understanding into your response."""
                
                return inputs
                
            except Exception as e:
                logger.error(f"Cognitive enhancement error: {e}")
                inputs["cognitive_analysis"] = {"error": str(e)}
                inputs["cognitive_prompt_addition"] = ""
                return inputs
        
        # Create LCEL chain with cognitive enhancement
        prompt = ChatPromptTemplate.from_template("""
You are an emotionally intelligent AI assistant with cognitive capabilities.

User Input: {input}
{cognitive_prompt_addition}

Provide a thoughtful, empathetic response that addresses both the emotional and practical aspects:
""")
        
        # Build the chain using LCEL
        chain = (
            RunnableLambda(cognitive_enhancement)
            | prompt
            | llm
        )
        
        # Test the LCEL chain
        print("🔗 Testing LCEL Cognitive Chain...")
        
        test_cases = [
            "I'm struggling to balance work and personal life",
            "I want to learn data science but don't know where to start",
            "I'm feeling imposter syndrome in my new leadership role"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}: {test_input}")
            
            # Run the chain
            result = await chain.ainvoke({"input": test_input})
            
            print(f"🤖 Enhanced Response: {result.content}")
        
    except Exception as e:
        logger.error(f"LCEL cognitive example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


async def create_multi_tool_chain_example():
    """Create a multi-tool chain example with cognitive capabilities."""
    
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available. Please install LangChain to run this example.")
        return
    
    print("\n🛠️ LangChain + AI Brain: Multi-Tool Chain Example")
    print("=" * 55)
    
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Initialize adapter
    adapter = LangChainAdapter(ai_brain=brain)
    
    try:
        # Create multiple cognitive tools
        emotional_tool = adapter.create_cognitive_tool(
            name="emotional_analysis",
            description="Analyze emotional content",
            ai_brain=brain,
            cognitive_systems=["emotional_intelligence", "empathy_response"]
        )
        
        goal_tool = adapter.create_cognitive_tool(
            name="goal_planning",
            description="Analyze goals and create plans",
            ai_brain=brain,
            cognitive_systems=["goal_hierarchy", "temporal_planning"]
        )
        
        safety_tool = adapter.create_cognitive_tool(
            name="safety_check",
            description="Check content safety and confidence",
            ai_brain=brain,
            cognitive_systems=["safety_guardrails", "confidence_tracking"]
        )
        
        # Create comprehensive analysis function
        async def comprehensive_analysis(text: str) -> Dict[str, Any]:
            """Run comprehensive analysis using multiple cognitive tools."""
            results = {}
            
            # Run all tools concurrently
            emotional_task = emotional_tool._arun(text)
            goal_task = goal_tool._arun(text)
            safety_task = safety_tool._arun(text)
            
            emotional_result, goal_result, safety_result = await asyncio.gather(
                emotional_task, goal_task, safety_task, return_exceptions=True
            )
            
            results["emotional_analysis"] = emotional_result if not isinstance(emotional_result, Exception) else str(emotional_result)
            results["goal_analysis"] = goal_result if not isinstance(goal_result, Exception) else str(goal_result)
            results["safety_analysis"] = safety_result if not isinstance(safety_result, Exception) else str(safety_result)
            
            return results
        
        # Test comprehensive analysis
        test_input = """I'm feeling anxious about my career transition from marketing 
        to data science. I want to complete this transition within 18 months, but I'm 
        worried about the technical challenges and whether I'm making the right decision."""
        
        print(f"📝 Comprehensive Analysis Input: {test_input}")
        print("\n🔍 Running multi-tool cognitive analysis...")
        
        analysis_results = await comprehensive_analysis(test_input)
        
        print(f"\n📊 Analysis Results:")
        for tool_name, result in analysis_results.items():
            print(f"\n{tool_name.upper()}:")
            print(f"{result}")
        
    except Exception as e:
        logger.error(f"Multi-tool chain example error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await brain.shutdown()


if __name__ == "__main__":
    print("🧠 LangChain + AI Brain Integration Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(create_cognitive_memory_example())
    asyncio.run(create_cognitive_tools_example())
    asyncio.run(create_cognitive_chain_example())
    asyncio.run(create_lcel_cognitive_example())
    asyncio.run(create_multi_tool_chain_example())
    
    print("\n✅ All LangChain examples completed!")
