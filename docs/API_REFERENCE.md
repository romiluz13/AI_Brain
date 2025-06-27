# AI Brain Python - API Reference

## Overview

The AI Brain Python package provides a comprehensive cognitive AI system with 16 specialized cognitive systems, framework adapters, and safety monitoring.

## Core Classes

### UniversalAIBrain

The main orchestrator class that coordinates all cognitive systems.

```python
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig

# Initialize with default configuration
brain = UniversalAIBrain()
await brain.initialize()

# Initialize with custom configuration
config = UniversalAIBrainConfig(
    mongodb_uri="mongodb://localhost:27017",
    database_name="ai_brain_custom",
    enable_safety_systems=True,
    cognitive_systems_config={
        "emotional_intelligence": {"sensitivity": 0.8},
        "goal_hierarchy": {"max_goals": 10}
    }
)
brain = UniversalAIBrain(config)
await brain.initialize()
```

#### Methods

- `async initialize() -> None`: Initialize the AI Brain and all cognitive systems
- `async process_input(input_data: CognitiveInputData) -> CognitiveResponse`: Process input through cognitive systems
- `async shutdown() -> None`: Gracefully shutdown the AI Brain
- `get_system_status() -> Dict[str, Any]`: Get current system status
- `get_cognitive_insights() -> Dict[str, Any]`: Get insights from cognitive systems

### CognitiveInputData

Input data model for cognitive processing.

```python
from ai_brain_python import CognitiveInputData, CognitiveContext

input_data = CognitiveInputData(
    text="I'm excited about this new AI project!",
    input_type="user_message",
    context=CognitiveContext(
        user_id="user123",
        session_id="session456"
    ),
    requested_systems=["emotional_intelligence", "goal_hierarchy"],
    processing_priority=7
)
```

#### Fields

- `text: str`: The input text to process
- `input_type: str`: Type of input (e.g., "user_message", "system_prompt")
- `context: CognitiveContext`: Processing context
- `requested_systems: Optional[List[str]]`: Specific cognitive systems to use
- `processing_priority: int`: Priority level (1-10)
- `metadata: Dict[str, Any]`: Additional metadata

### CognitiveResponse

Response from cognitive processing.

```python
response = await brain.process_input(input_data)

print(f"Confidence: {response.confidence}")
print(f"Processing time: {response.processing_time_ms}ms")
print(f"Emotional state: {response.emotional_state}")
print(f"Goals detected: {response.goal_hierarchy}")
```

#### Fields

- `confidence: float`: Overall confidence score (0.0-1.0)
- `processing_time_ms: float`: Processing time in milliseconds
- `cognitive_results: Dict[str, Any]`: Results from each cognitive system
- `emotional_state: EmotionalState`: Detected emotional state
- `goal_hierarchy: GoalHierarchy`: Extracted goal hierarchy
- `safety_assessment: Dict[str, Any]`: Safety analysis results

## Framework Adapters

### CrewAI Adapter

```python
from ai_brain_python.adapters import CrewAIAdapter

# Initialize adapter
adapter = CrewAIAdapter(ai_brain_config)

# Create cognitive agent
agent = adapter.create_cognitive_agent(
    role="Data Analyst",
    goal="Analyze data with emotional intelligence",
    backstory="An AI agent with advanced cognitive capabilities",
    tools=[data_analysis_tool],
    cognitive_systems=["emotional_intelligence", "confidence_tracking"]
)

# Create cognitive crew
crew = adapter.create_cognitive_crew(
    agents=[agent],
    tasks=[analysis_task],
    enable_cognitive_coordination=True
)

# Execute with cognitive enhancement
result = await crew.cognitive_kickoff()
```

### Pydantic AI Adapter

```python
from ai_brain_python.adapters import PydanticAIAdapter
from pydantic_ai.models.openai import OpenAIModel

# Initialize adapter
adapter = PydanticAIAdapter(ai_brain_config)

# Create cognitive agent
agent = adapter.create_cognitive_agent(
    model=OpenAIModel('gpt-4'),
    cognitive_systems=["emotional_intelligence", "safety_guardrails"],
    instructions="You are an empathetic AI assistant"
)

# Run with cognitive enhancement
response = await agent.run("How can I improve my productivity?")
print(response.content)
print(f"Confidence: {response.confidence}")
```

### LangChain Adapter

```python
from ai_brain_python.adapters import LangChainAdapter
from langchain_core.prompts import PromptTemplate

# Initialize adapter
adapter = LangChainAdapter(ai_brain_config)

# Create cognitive memory
memory = adapter.create_cognitive_memory(
    ai_brain=brain,
    memory_key="chat_history"
)

# Create cognitive tool
tool = adapter.create_cognitive_tool(
    name="emotional_analysis",
    description="Analyze emotional content",
    ai_brain=brain,
    cognitive_systems=["emotional_intelligence"]
)

# Create cognitive chain
prompt = PromptTemplate(
    template="Analyze this text: {input}",
    input_variables=["input"]
)

chain = adapter.create_cognitive_chain(
    llm=llm,
    prompt=prompt,
    ai_brain=brain,
    memory=memory
)
```

### LangGraph Adapter

```python
from ai_brain_python.adapters import LangGraphAdapter

# Initialize adapter
adapter = LangGraphAdapter(ai_brain_config)

# Create cognitive graph
graph = adapter.create_cognitive_graph(
    ai_brain=brain,
    state_schema=CustomState
)

# Add cognitive nodes
graph.add_cognitive_node(
    name="emotional_analysis",
    cognitive_systems=["emotional_intelligence"]
)

graph.add_cognitive_node(
    name="goal_extraction", 
    cognitive_systems=["goal_hierarchy"]
)

# Add edges
graph.add_cognitive_edge("emotional_analysis", "goal_extraction")
graph.set_entry_point("emotional_analysis")
graph.set_finish_point("goal_extraction")

# Compile and run
compiled_graph = graph.compile()
result = await compiled_graph.invoke({"input": "I want to learn Python"})
```

## Safety Systems

### Integrated Safety System

```python
from ai_brain_python.safety import IntegratedSafetySystem, SafetySystemConfig

# Initialize safety system
safety_config = SafetySystemConfig()
safety_system = IntegratedSafetySystem(safety_config)
await safety_system.initialize()

# Perform comprehensive safety check
result = await safety_system.comprehensive_safety_check(
    text="This is some user input to check",
    user_id="user123",
    session_id="session456"
)

print(f"Is safe: {result['overall_safe']}")
print(f"Violations: {result['safety_guardrails']['violations']}")
print(f"Hallucinations: {result['hallucination_detection']['has_hallucinations']}")
```

### Safety Guardrails

```python
from ai_brain_python.safety import SafetyGuardrails, SafetyConfig

# Configure safety guardrails
config = SafetyConfig(
    safety_level="strict",
    enable_pii_detection=True,
    enable_harmful_content_detection=True
)

guardrails = SafetyGuardrails(config)

# Check safety
result = await guardrails.check_safety("Some text to check")
print(f"Is safe: {result['is_safe']}")
print(f"Masked text: {result['masked_text']}")
```

### Compliance Logger

```python
from ai_brain_python.safety import ComplianceLogger, ComplianceConfig, EventType

# Configure compliance logging
config = ComplianceConfig(
    enabled_standards=["GDPR", "CCPA"],
    mongodb_uri="mongodb://localhost:27017"
)

logger = ComplianceLogger(config)
await logger.initialize()

# Log compliance event
event_id = await logger.log_event(
    event_type=EventType.DATA_PROCESSING,
    description="User data processed",
    user_id="user123",
    metadata={"processing_type": "cognitive_analysis"}
)
```

## Cognitive Systems

### Emotional Intelligence Engine

```python
from ai_brain_python.cognitive_systems import EmotionalIntelligenceEngine

engine = EmotionalIntelligenceEngine(config)
await engine.initialize()

result = await engine.process_input(
    text="I'm so frustrated with this bug!",
    context=context
)

print(f"Primary emotion: {result['emotional_state']['primary_emotion']}")
print(f"Intensity: {result['emotional_state']['emotion_intensity']}")
```

### Goal Hierarchy Manager

```python
from ai_brain_python.cognitive_systems import GoalHierarchyManager

manager = GoalHierarchyManager(config)
await manager.initialize()

result = await manager.process_input(
    text="I want to learn Python to build AI applications",
    context=context
)

print(f"Goals extracted: {result['extracted_goals']}")
print(f"Goal hierarchy: {result['goal_hierarchy']}")
```

## Database Integration

### MongoDB Client

```python
from ai_brain_python.database import MongoDBClient

# Initialize MongoDB client
client = MongoDBClient("mongodb://localhost:27017")
await client.initialize()

# Store memory
await client.store_memory(
    user_id="user123",
    content="Important conversation",
    memory_type="conversation",
    metadata={"topic": "AI development"}
)

# Search memories
results = await client.search_memories(
    user_id="user123",
    query="AI development",
    limit=10
)
```

## Configuration

### UniversalAIBrainConfig

```python
from ai_brain_python import UniversalAIBrainConfig

config = UniversalAIBrainConfig(
    # MongoDB settings
    mongodb_uri="mongodb://localhost:27017",
    database_name="ai_brain_prod",
    
    # Safety settings
    enable_safety_systems=True,
    safety_level="strict",
    
    # Performance settings
    max_concurrent_requests=100,
    request_timeout_seconds=30,
    
    # Cognitive systems configuration
    cognitive_systems_config={
        "emotional_intelligence": {
            "sensitivity": 0.8,
            "enable_empathy_responses": True
        },
        "goal_hierarchy": {
            "max_goals": 10,
            "enable_goal_prioritization": True
        }
    }
)
```

## Error Handling

```python
from ai_brain_python.core.exceptions import (
    AIBrainError,
    CognitiveSystemError,
    SafetyViolationError,
    DatabaseConnectionError
)

try:
    response = await brain.process_input(input_data)
except SafetyViolationError as e:
    print(f"Safety violation: {e}")
except CognitiveSystemError as e:
    print(f"Cognitive system error: {e}")
except AIBrainError as e:
    print(f"General AI Brain error: {e}")
```

## Best Practices

1. **Always initialize systems before use**:
   ```python
   brain = UniversalAIBrain()
   await brain.initialize()  # Required
   ```

2. **Use context managers for resource management**:
   ```python
   async with UniversalAIBrain() as brain:
       response = await brain.process_input(input_data)
   ```

3. **Handle safety violations appropriately**:
   ```python
   result = await safety_system.comprehensive_safety_check(text)
   if not result['overall_safe']:
       # Handle unsafe content
       return filtered_response
   ```

4. **Monitor system performance**:
   ```python
   dashboard = await safety_system.get_safety_dashboard()
   print(f"System health: {dashboard['health_status']['status']}")
   ```

5. **Use appropriate cognitive systems for your use case**:
   ```python
   # For emotional content
   systems = ["emotional_intelligence", "empathy_response"]
   
   # For goal-oriented tasks
   systems = ["goal_hierarchy", "temporal_planning"]
   
   # For safety-critical applications
   systems = ["safety_guardrails", "confidence_tracking"]
   ```
