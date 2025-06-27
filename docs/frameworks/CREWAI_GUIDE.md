# 🤖 CrewAI Integration Guide

Complete guide for integrating Universal AI Brain with CrewAI to create cognitive agents and crews.

## 📦 Installation

```bash
# Install AI Brain with CrewAI support
pip install ai-brain-python[crewai]

# Or install CrewAI separately
pip install ai-brain-python
pip install crewai
```

## 🚀 Quick Start

### Basic Cognitive Agent

```python
import asyncio
from ai_brain_python import UniversalAIBrain
from ai_brain_python.adapters import CrewAIAdapter

async def main():
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Create CrewAI adapter
    adapter = CrewAIAdapter(ai_brain=brain)
    
    # Create cognitive agent
    agent = adapter.create_cognitive_agent(
        role="Emotional Intelligence Analyst",
        goal="Provide empathetic and insightful analysis of user emotions",
        backstory="You are an expert in emotional intelligence with deep understanding of human psychology",
        cognitive_systems=["emotional_intelligence", "empathy_response", "communication_protocol"]
    )
    
    # Use the agent
    result = await agent.execute("I'm feeling overwhelmed with my workload")
    print(f"Agent response: {result}")
    
    await brain.shutdown()

asyncio.run(main())
```

### Cognitive Crew Example

```python
import asyncio
from crewai import Task
from ai_brain_python import UniversalAIBrain
from ai_brain_python.adapters import CrewAIAdapter

async def main():
    # Initialize AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Create adapter
    adapter = CrewAIAdapter(ai_brain=brain)
    
    # Create specialized agents
    emotional_analyst = adapter.create_cognitive_agent(
        role="Emotional Intelligence Analyst",
        goal="Analyze emotional content and provide empathetic responses",
        backstory="Expert in emotional intelligence and human psychology",
        cognitive_systems=["emotional_intelligence", "empathy_response"]
    )
    
    goal_strategist = adapter.create_cognitive_agent(
        role="Goal Strategy Advisor",
        goal="Help users identify and prioritize their goals",
        backstory="Strategic planning expert with focus on goal achievement",
        cognitive_systems=["goal_hierarchy", "temporal_planning", "strategic_thinking"]
    )
    
    communication_expert = adapter.create_cognitive_agent(
        role="Communication Specialist",
        goal="Optimize communication style and clarity",
        backstory="Expert in effective communication and interpersonal skills",
        cognitive_systems=["communication_protocol", "cultural_knowledge"]
    )
    
    # Create tasks
    emotional_task = adapter.create_task(
        description="Analyze the emotional content and provide empathetic insights",
        expected_output="Emotional analysis with empathy response and recommendations",
        agent=emotional_analyst
    )
    
    goal_task = adapter.create_task(
        description="Identify goals and create strategic plan",
        expected_output="Goal hierarchy with prioritized action plan",
        agent=goal_strategist
    )
    
    communication_task = adapter.create_task(
        description="Optimize communication approach based on emotional and goal analysis",
        expected_output="Communication strategy and style recommendations",
        agent=communication_expert
    )
    
    # Create cognitive crew
    crew = adapter.create_cognitive_crew(
        agents=[emotional_analyst, goal_strategist, communication_expert],
        tasks=[emotional_task, goal_task, communication_task],
        enable_cognitive_coordination=True,
        shared_cognitive_context=True
    )
    
    # Execute crew
    result = await crew.cognitive_kickoff(
        inputs={"user_input": "I want to advance my career but I'm nervous about taking risks"}
    )
    
    print("Crew Results:")
    print(f"Emotional Analysis: {result['emotional_analysis']}")
    print(f"Goal Strategy: {result['goal_strategy']}")
    print(f"Communication Plan: {result['communication_plan']}")
    
    await brain.shutdown()

asyncio.run(main())
```

## 🧠 Cognitive Agent Features

### Available Cognitive Systems

When creating cognitive agents, you can specify which cognitive systems to integrate:

```python
# Emotional Intelligence Focus
emotional_agent = adapter.create_cognitive_agent(
    role="Empathy Expert",
    goal="Provide emotional support",
    cognitive_systems=[
        "emotional_intelligence",    # Emotion detection and analysis
        "empathy_response",         # Generate empathetic responses
        "communication_protocol"    # Adapt communication style
    ]
)

# Goal-Oriented Focus
goal_agent = adapter.create_cognitive_agent(
    role="Goal Achievement Coach",
    goal="Help users achieve their objectives",
    cognitive_systems=[
        "goal_hierarchy",          # Goal extraction and prioritization
        "temporal_planning",       # Time-based planning
        "strategic_thinking",      # Strategic analysis
        "self_improvement"         # Continuous improvement
    ]
)

# Safety-Focused Agent
safety_agent = adapter.create_cognitive_agent(
    role="Content Safety Monitor",
    goal="Ensure safe and appropriate interactions",
    cognitive_systems=[
        "safety_guardrails",       # Content safety checks
        "confidence_tracking",     # Confidence assessment
        "cultural_knowledge"       # Cultural sensitivity
    ]
)
```

### Cognitive Agent Configuration

```python
# Advanced agent configuration
agent = adapter.create_cognitive_agent(
    role="Advanced Cognitive Assistant",
    goal="Provide comprehensive cognitive assistance",
    backstory="Multi-faceted AI assistant with deep cognitive capabilities",
    
    # Cognitive systems
    cognitive_systems=[
        "emotional_intelligence",
        "goal_hierarchy", 
        "confidence_tracking",
        "attention_management"
    ],
    
    # Cognitive configuration
    cognitive_config={
        "emotional_intelligence": {
            "sensitivity": 0.9,
            "enable_empathy_responses": True
        },
        "goal_hierarchy": {
            "max_goals": 8,
            "enable_prioritization": True
        }
    },
    
    # CrewAI specific settings
    verbose=True,
    allow_delegation=True,
    max_iter=5,
    memory=True
)
```

## 🔧 Advanced Features

### Cognitive Crew Coordination

```python
# Create crew with cognitive coordination
crew = adapter.create_cognitive_crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    
    # Cognitive coordination features
    enable_cognitive_coordination=True,    # Agents share cognitive insights
    shared_cognitive_context=True,         # Shared emotional/goal context
    cognitive_memory_sharing=True,         # Shared memory across agents
    
    # Cognitive processing settings
    cognitive_processing_mode="parallel",  # "sequential" or "parallel"
    cognitive_confidence_threshold=0.7,    # Minimum confidence for decisions
    
    # CrewAI settings
    verbose=True,
    process="sequential"  # or "hierarchical"
)
```

### Real-time Cognitive Monitoring

```python
# Monitor cognitive processing in real-time
async def monitor_crew_cognition(crew):
    """Monitor cognitive processing during crew execution."""
    
    # Get cognitive insights
    insights = await crew.get_cognitive_insights()
    
    print("Cognitive Monitoring:")
    print(f"Emotional State: {insights['emotional_state']}")
    print(f"Goal Progress: {insights['goal_progress']}")
    print(f"Confidence Levels: {insights['confidence_levels']}")
    print(f"Attention Focus: {insights['attention_focus']}")
    
    # Get safety assessment
    safety = await crew.get_safety_assessment()
    print(f"Safety Status: {safety['overall_safe']}")
    
    return insights

# Use during crew execution
result = await crew.cognitive_kickoff(inputs=user_inputs)
insights = await monitor_crew_cognition(crew)
```

### Custom Cognitive Tools

```python
# Create custom cognitive tools for agents
cognitive_tool = adapter.create_cognitive_tool(
    name="emotional_analysis_tool",
    description="Analyze emotional content with AI Brain cognitive systems",
    cognitive_systems=["emotional_intelligence", "empathy_response"],
    
    # Tool configuration
    tool_config={
        "return_format": "structured",
        "include_confidence": True,
        "include_recommendations": True
    }
)

# Add tool to agent
agent = adapter.create_cognitive_agent(
    role="Emotional Analyst",
    goal="Provide emotional insights",
    tools=[cognitive_tool],
    cognitive_systems=["emotional_intelligence"]
)
```

## 📊 Cognitive Crew Workflows

### Sequential Cognitive Processing

```python
# Sequential workflow with cognitive handoffs
async def sequential_cognitive_workflow():
    # Step 1: Emotional Analysis
    emotional_result = await emotional_agent.execute(user_input)
    
    # Step 2: Goal Analysis (with emotional context)
    goal_result = await goal_agent.execute(
        user_input, 
        context={"emotional_analysis": emotional_result}
    )
    
    # Step 3: Action Planning (with full context)
    action_result = await planning_agent.execute(
        user_input,
        context={
            "emotional_analysis": emotional_result,
            "goal_analysis": goal_result
        }
    )
    
    return {
        "emotional": emotional_result,
        "goals": goal_result,
        "action_plan": action_result
    }
```

### Parallel Cognitive Processing

```python
# Parallel processing with cognitive synthesis
async def parallel_cognitive_workflow():
    # Process in parallel
    tasks = [
        emotional_agent.execute(user_input),
        goal_agent.execute(user_input),
        attention_agent.execute(user_input)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Synthesize results with cognitive coordination
    synthesis = await synthesis_agent.execute(
        user_input,
        context={
            "emotional_analysis": results[0],
            "goal_analysis": results[1], 
            "attention_analysis": results[2]
        }
    )
    
    return synthesis
```

## 🛡️ Safety and Monitoring

### Cognitive Safety Integration

```python
# Create safety-aware cognitive crew
safety_crew = adapter.create_cognitive_crew(
    agents=[main_agent, safety_agent],
    tasks=[main_task, safety_task],
    
    # Safety settings
    enable_safety_monitoring=True,
    safety_confidence_threshold=0.8,
    auto_safety_intervention=True,
    
    # Safety cognitive systems
    safety_cognitive_systems=[
        "safety_guardrails",
        "confidence_tracking", 
        "cultural_knowledge"
    ]
)
```

### Performance Monitoring

```python
# Monitor crew performance
async def monitor_crew_performance(crew):
    """Monitor cognitive crew performance metrics."""
    
    performance = await crew.get_performance_metrics()
    
    return {
        "processing_time": performance["total_processing_time_ms"],
        "cognitive_efficiency": performance["cognitive_efficiency_score"],
        "agent_coordination": performance["coordination_effectiveness"],
        "confidence_levels": performance["average_confidence"],
        "safety_score": performance["safety_assessment_score"]
    }
```

## 🔄 Integration Patterns

### With Existing CrewAI Code

```python
# Enhance existing CrewAI agents with cognitive capabilities
from crewai import Agent, Task, Crew

# Traditional CrewAI agent
traditional_agent = Agent(
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="Expert data analyst"
)

# Enhanced with AI Brain cognition
cognitive_agent = adapter.enhance_existing_agent(
    agent=traditional_agent,
    cognitive_systems=["confidence_tracking", "attention_management"],
    enable_cognitive_enhancement=True
)

# Use in existing crew
crew = Crew(
    agents=[cognitive_agent],
    tasks=[existing_task],
    verbose=True
)
```

### Gradual Migration

```python
# Gradually add cognitive capabilities to existing crews
class CognitiveCrewMigration:
    def __init__(self, existing_crew, ai_brain):
        self.existing_crew = existing_crew
        self.adapter = CrewAIAdapter(ai_brain=ai_brain)
    
    async def add_cognitive_agent(self, role, cognitive_systems):
        """Add a cognitive agent to existing crew."""
        cognitive_agent = self.adapter.create_cognitive_agent(
            role=role,
            goal=f"Provide cognitive enhancement for {role}",
            cognitive_systems=cognitive_systems
        )
        
        self.existing_crew.agents.append(cognitive_agent)
        return cognitive_agent
    
    async def enable_cognitive_coordination(self):
        """Enable cognitive coordination for the crew."""
        return self.adapter.enhance_crew_with_cognition(self.existing_crew)
```

## 📈 Best Practices

### 1. Cognitive System Selection

```python
# Choose cognitive systems based on use case
use_case_mappings = {
    "customer_support": ["emotional_intelligence", "empathy_response", "communication_protocol"],
    "goal_coaching": ["goal_hierarchy", "temporal_planning", "self_improvement"],
    "content_moderation": ["safety_guardrails", "cultural_knowledge", "confidence_tracking"],
    "strategic_planning": ["strategic_thinking", "temporal_planning", "goal_hierarchy"],
    "learning_assistance": ["attention_management", "skill_capability", "self_improvement"]
}
```

### 2. Performance Optimization

```python
# Optimize cognitive processing
optimized_agent = adapter.create_cognitive_agent(
    role="Optimized Agent",
    goal="High-performance cognitive processing",
    
    # Optimize cognitive systems
    cognitive_systems=["emotional_intelligence", "goal_hierarchy"],  # Limit to essential systems
    cognitive_config={
        "processing_priority": 8,           # Higher priority
        "enable_caching": True,             # Cache cognitive results
        "batch_processing": True,           # Process multiple inputs together
        "confidence_threshold": 0.7         # Skip low-confidence processing
    }
)
```

### 3. Error Handling

```python
# Robust error handling for cognitive crews
async def robust_cognitive_execution(crew, inputs):
    """Execute crew with comprehensive error handling."""
    try:
        result = await crew.cognitive_kickoff(inputs)
        return {"success": True, "result": result}
        
    except CognitiveProcessingError as e:
        # Handle cognitive system errors
        fallback_result = await crew.fallback_execution(inputs)
        return {"success": False, "fallback": fallback_result, "error": str(e)}
        
    except SafetyViolationError as e:
        # Handle safety violations
        return {"success": False, "safety_violation": True, "error": str(e)}
        
    except Exception as e:
        # Handle general errors
        return {"success": False, "error": str(e)}
```

---

## 📚 Additional Resources

- **[CrewAI Documentation](https://docs.crewai.com/)**
- **[AI Brain Core Documentation](../API_REFERENCE.md)**
- **[Cognitive Systems Guide](../cognitive_systems/)**
- **[Examples](../../examples/crewai_example.py)**

For more advanced CrewAI integration patterns, see the [examples directory](../../examples/) and [API reference](../API_REFERENCE.md).
