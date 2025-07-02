# 🦜 LangChain Integration Guide

Complete guide for integrating Universal AI Brain with LangChain to create cognitive tools, memory, and chains.

## 📦 Installation

```bash
# Install AI Brain with LangChain support
pip install ai-brain-python[langchain]

# Or install LangChain separately
pip install ai-brain-python
pip install langchain langchain-openai
```

## 🚀 Quick Start

### Basic Cognitive Tool

```python
import asyncio
from ai_brain_python import UniversalAIBrain
from ai_brain_python.adapters import LangChainAdapter

async def main():
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Create LangChain adapter
    adapter = LangChainAdapter(ai_brain=brain)
    
    # Create cognitive tool
    emotional_tool = adapter.create_cognitive_tool(
        name="emotional_analysis",
        description="Analyze emotional content using AI Brain cognitive systems",
        cognitive_systems=["emotional_intelligence", "empathy_response"]
    )
    
    # Use the tool
    result = await emotional_tool._arun("I'm feeling overwhelmed with my workload")
    print(f"Emotional analysis: {result}")
    
    await brain.shutdown()

asyncio.run(main())
```

### Cognitive Memory Integration

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from ai_brain_python import UniversalAIBrain
from ai_brain_python.adapters import LangChainAdapter

async def main():
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Create adapter
    adapter = LangChainAdapter(ai_brain=brain)
    
    # Create cognitive memory
    memory = adapter.create_cognitive_memory(
        ai_brain=brain,
        memory_key="chat_history",
        cognitive_systems=["semantic_memory", "emotional_intelligence"]
    )
    
    # Create LLM and chain
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    chain = ConversationChain(llm=llm, memory=memory)
    
    # Use chain with cognitive memory
    response1 = await chain.arun("I'm excited about learning AI")
    print(f"Response 1: {response1}")
    
    response2 = await chain.arun("How should I start?")
    print(f"Response 2: {response2}")
    
    # Get cognitive context
    cognitive_context = await memory.get_cognitive_context("How should I start?")
    print(f"Cognitive context: {cognitive_context}")
    
    await brain.shutdown()

asyncio.run(main())
```

## 🔧 Cognitive Tools

### Creating Specialized Cognitive Tools

```python
# Emotional Intelligence Tool
emotional_tool = adapter.create_cognitive_tool(
    name="emotional_intelligence",
    description="Analyze emotions and provide empathetic responses",
    cognitive_systems=["emotional_intelligence", "empathy_response"],
    tool_config={
        "return_format": "structured",
        "include_confidence": True,
        "include_empathy_response": True
    }
)

# Goal Analysis Tool
goal_tool = adapter.create_cognitive_tool(
    name="goal_analysis", 
    description="Extract and prioritize goals from text",
    cognitive_systems=["goal_hierarchy", "temporal_planning"],
    tool_config={
        "max_goals": 5,
        "include_timeline": True,
        "include_dependencies": True
    }
)

# Safety Assessment Tool
safety_tool = adapter.create_cognitive_tool(
    name="safety_assessment",
    description="Assess content safety and compliance",
    cognitive_systems=["safety_guardrails", "confidence_tracking"],
    tool_config={
        "safety_level": "strict",
        "include_recommendations": True
    }
)

# Attention Management Tool
attention_tool = adapter.create_cognitive_tool(
    name="attention_management",
    description="Analyze attention patterns and provide focus recommendations",
    cognitive_systems=["attention_management", "strategic_thinking"],
    tool_config={
        "include_distraction_analysis": True,
        "include_focus_recommendations": True
    }
)
```

### Using Cognitive Tools in Chains

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Create agent with cognitive tools
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

cognitive_tools = [
    emotional_tool,
    goal_tool,
    safety_tool,
    attention_tool
]

agent = initialize_agent(
    tools=cognitive_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use agent with cognitive capabilities
result = await agent.arun(
    "I'm feeling stressed about my career goals and having trouble focusing. "
    "Can you help me analyze my situation and provide recommendations?"
)
```

## 🧠 Cognitive Memory Systems

### Enhanced Memory with Cognitive Context

```python
# Create cognitive memory with multiple systems
cognitive_memory = adapter.create_cognitive_memory(
    ai_brain=brain,
    memory_key="enhanced_chat_history",
    cognitive_systems=[
        "semantic_memory",        # Enhanced memory storage and retrieval
        "emotional_intelligence", # Emotional context preservation
        "goal_hierarchy",        # Goal context tracking
        "attention_management"   # Attention pattern memory
    ],
    memory_config={
        "max_memory_items": 100,
        "enable_emotional_context": True,
        "enable_goal_context": True,
        "enable_attention_context": True,
        "memory_decay_factor": 0.95
    }
)

# Use in conversation chain
chain = ConversationChain(
    llm=llm,
    memory=cognitive_memory,
    verbose=True
)
```

### Memory with Cognitive Insights

```python
# Get cognitive insights from memory
async def analyze_conversation_patterns(memory):
    """Analyze conversation patterns using cognitive memory."""
    
    # Get emotional patterns
    emotional_patterns = await memory.get_emotional_patterns()
    print(f"Emotional trends: {emotional_patterns}")
    
    # Get goal evolution
    goal_evolution = await memory.get_goal_evolution()
    print(f"Goal progression: {goal_evolution}")
    
    # Get attention patterns
    attention_patterns = await memory.get_attention_patterns()
    print(f"Attention focus: {attention_patterns}")
    
    return {
        "emotional_patterns": emotional_patterns,
        "goal_evolution": goal_evolution,
        "attention_patterns": attention_patterns
    }

# Use after conversation
patterns = await analyze_conversation_patterns(cognitive_memory)
```

## ⛓️ Cognitive Chains

### Creating Cognitive-Enhanced Chains

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create cognitive chain
cognitive_chain = adapter.create_cognitive_chain(
    llm=llm,
    prompt=PromptTemplate(
        template="""You are an emotionally intelligent AI assistant.
        
        Cognitive Analysis: {cognitive_analysis}
        User Input: {input}
        
        Provide a response that demonstrates emotional intelligence and goal awareness:""",
        input_variables=["cognitive_analysis", "input"]
    ),
    ai_brain=brain,
    cognitive_systems=["emotional_intelligence", "goal_hierarchy", "empathy_response"],
    chain_config={
        "include_cognitive_preprocessing": True,
        "include_confidence_tracking": True,
        "enable_safety_checks": True
    }
)

# Use cognitive chain
result = await cognitive_chain.arun(
    input="I'm considering a career change but I'm scared of making the wrong decision"
)
```

### Sequential Cognitive Processing Chain

```python
from langchain.chains import SequentialChain

# Create sequential cognitive processing
emotional_chain = adapter.create_cognitive_chain(
    llm=llm,
    prompt=PromptTemplate(
        template="Analyze the emotional content: {input}",
        input_variables=["input"]
    ),
    cognitive_systems=["emotional_intelligence"],
    output_key="emotional_analysis"
)

goal_chain = adapter.create_cognitive_chain(
    llm=llm,
    prompt=PromptTemplate(
        template="Based on emotional analysis: {emotional_analysis}\nExtract goals from: {input}",
        input_variables=["emotional_analysis", "input"]
    ),
    cognitive_systems=["goal_hierarchy"],
    output_key="goal_analysis"
)

synthesis_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="""Synthesize the analysis:
        Emotional: {emotional_analysis}
        Goals: {goal_analysis}
        Original: {input}
        
        Provide comprehensive recommendations:""",
        input_variables=["emotional_analysis", "goal_analysis", "input"]
    ),
    output_key="recommendations"
)

# Combine into sequential chain
sequential_cognitive_chain = SequentialChain(
    chains=[emotional_chain, goal_chain, synthesis_chain],
    input_variables=["input"],
    output_variables=["emotional_analysis", "goal_analysis", "recommendations"],
    verbose=True
)
```

## 🔄 LCEL (LangChain Expression Language) Integration

### Cognitive Enhancement with LCEL

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Create cognitive enhancement function
async def cognitive_enhancement(inputs):
    """Enhance inputs with cognitive analysis."""
    
    # Process through AI Brain
    input_data = CognitiveInputData(
        text=inputs["input"],
        input_type="lcel_enhancement",
        context=CognitiveContext(
            user_id=inputs.get("user_id", "default"),
            session_id=inputs.get("session_id", "default")
        ),
        requested_systems=["emotional_intelligence", "goal_hierarchy"]
    )
    
    response = await brain.process_input(input_data)
    
    # Add cognitive context to inputs
    inputs["cognitive_context"] = {
        "emotion": response.emotional_state.primary_emotion,
        "emotion_intensity": response.emotional_state.emotion_intensity,
        "primary_goal": response.goal_hierarchy.primary_goal,
        "confidence": response.confidence
    }
    
    return inputs

# Create LCEL chain with cognitive enhancement
prompt = ChatPromptTemplate.from_template("""
You are an emotionally intelligent assistant.

Cognitive Analysis:
- Emotion: {cognitive_context[emotion]} (intensity: {cognitive_context[emotion_intensity]})
- Primary Goal: {cognitive_context[primary_goal]}
- Confidence: {cognitive_context[confidence]}

User Input: {input}

Provide a thoughtful response that addresses both emotional and practical aspects:
""")

# Build the chain
cognitive_lcel_chain = (
    RunnableLambda(cognitive_enhancement)
    | prompt
    | llm
)

# Use the chain
result = await cognitive_lcel_chain.ainvoke({
    "input": "I'm struggling to balance work and personal life",
    "user_id": "user123",
    "session_id": "session456"
})
```

### Parallel Cognitive Processing with LCEL

```python
from langchain_core.runnables import RunnableParallel

# Create parallel cognitive processors
async def emotional_processor(inputs):
    """Process emotional content."""
    # Implementation here
    return {"emotional_analysis": "..."}

async def goal_processor(inputs):
    """Process goal content.""" 
    # Implementation here
    return {"goal_analysis": "..."}

async def safety_processor(inputs):
    """Process safety assessment."""
    # Implementation here
    return {"safety_analysis": "..."}

# Create parallel cognitive chain
parallel_cognitive_chain = RunnableParallel({
    "emotional": RunnableLambda(emotional_processor),
    "goals": RunnableLambda(goal_processor),
    "safety": RunnableLambda(safety_processor),
    "original": RunnablePassthrough()
})

# Synthesis function
async def synthesize_cognitive_results(inputs):
    """Synthesize all cognitive analyses."""
    synthesis_prompt = f"""
    Emotional Analysis: {inputs['emotional']}
    Goal Analysis: {inputs['goals']}
    Safety Analysis: {inputs['safety']}
    Original Input: {inputs['original']['input']}
    
    Provide comprehensive recommendations:
    """
    
    return await llm.ainvoke(synthesis_prompt)

# Complete chain
complete_cognitive_chain = (
    parallel_cognitive_chain
    | RunnableLambda(synthesize_cognitive_results)
)
```

## 🛡️ Safety Integration

### Safety-Aware Cognitive Tools

```python
# Create safety-enhanced cognitive tool
safe_cognitive_tool = adapter.create_cognitive_tool(
    name="safe_emotional_analysis",
    description="Emotionally intelligent analysis with safety checks",
    cognitive_systems=[
        "emotional_intelligence",
        "empathy_response", 
        "safety_guardrails",
        "confidence_tracking"
    ],
    safety_config={
        "enable_safety_checks": True,
        "safety_threshold": 0.8,
        "auto_safety_intervention": True,
        "log_safety_events": True
    }
)

# Use in agent with safety monitoring
safe_agent = initialize_agent(
    tools=[safe_cognitive_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
```

### Safety Monitoring Chain

```python
# Create safety monitoring wrapper
async def safety_monitored_chain(chain, inputs):
    """Execute chain with safety monitoring."""
    
    # Pre-execution safety check
    safety_check = await brain.safety_system.comprehensive_safety_check(
        text=inputs.get("input", ""),
        user_id=inputs.get("user_id", "default")
    )
    
    if not safety_check["overall_safe"]:
        return {
            "result": "Content flagged by safety systems",
            "safety_violation": True,
            "safety_details": safety_check
        }
    
    # Execute chain
    try:
        result = await chain.arun(inputs)
        
        # Post-execution safety check
        output_safety = await brain.safety_system.comprehensive_safety_check(
            text=result,
            user_id=inputs.get("user_id", "default")
        )
        
        return {
            "result": result,
            "safety_violation": False,
            "input_safety": safety_check,
            "output_safety": output_safety
        }
        
    except Exception as e:
        return {
            "result": f"Error: {str(e)}",
            "safety_violation": False,
            "error": True
        }
```

## 📊 Performance Optimization

### Caching Cognitive Results

```python
# Create cached cognitive tool
cached_tool = adapter.create_cognitive_tool(
    name="cached_emotional_analysis",
    description="Cached emotional analysis for better performance",
    cognitive_systems=["emotional_intelligence"],
    performance_config={
        "enable_caching": True,
        "cache_ttl": 3600,  # 1 hour
        "cache_key_strategy": "content_hash",
        "enable_batch_processing": True
    }
)
```

### Async Batch Processing

```python
# Process multiple inputs efficiently
async def batch_cognitive_processing(inputs_list):
    """Process multiple inputs in batches."""
    
    batch_size = 10
    results = []
    
    for i in range(0, len(inputs_list), batch_size):
        batch = inputs_list[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [
            cognitive_chain.arun(input_data) 
            for input_data in batch
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results
```

## 📈 Best Practices

### 1. Tool Selection Strategy

```python
# Choose tools based on use case
use_case_tools = {
    "customer_support": ["emotional_intelligence", "empathy_response", "safety_guardrails"],
    "content_analysis": ["semantic_memory", "safety_guardrails", "confidence_tracking"],
    "goal_coaching": ["goal_hierarchy", "temporal_planning", "self_improvement"],
    "educational": ["attention_management", "skill_capability", "self_improvement"]
}

def get_tools_for_use_case(use_case):
    """Get recommended cognitive tools for a use case."""
    systems = use_case_tools.get(use_case, ["emotional_intelligence"])
    return [
        adapter.create_cognitive_tool(
            name=f"{system}_tool",
            description=f"Tool for {system}",
            cognitive_systems=[system]
        )
        for system in systems
    ]
```

### 2. Memory Management

```python
# Efficient memory management
memory = adapter.create_cognitive_memory(
    ai_brain=brain,
    memory_key="optimized_memory",
    cognitive_systems=["semantic_memory", "emotional_intelligence"],
    memory_config={
        "max_memory_items": 50,        # Limit memory size
        "enable_memory_compression": True,  # Compress old memories
        "memory_decay_factor": 0.9,    # Gradual forgetting
        "enable_selective_memory": True # Remember important items longer
    }
)
```

### 3. Error Handling

```python
# Robust error handling for cognitive chains
async def robust_cognitive_execution(chain, inputs):
    """Execute cognitive chain with comprehensive error handling."""
    
    try:
        result = await chain.arun(inputs)
        return {"success": True, "result": result}
        
    except CognitiveProcessingError as e:
        # Fallback to basic processing
        fallback_result = await basic_chain.arun(inputs)
        return {"success": False, "fallback": fallback_result, "error": str(e)}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## 📚 Additional Resources

- **[LangChain Documentation](https://python.langchain.com/)**
- **[AI Brain Core Documentation](../API_REFERENCE.md)**
- **[Cognitive Systems Guide](../cognitive_systems/)**
- **[Examples](../../examples/langchain_example.py)**

For more advanced LangChain integration patterns, see the [examples directory](../../examples/) and [API reference](../API_REFERENCE.md).
