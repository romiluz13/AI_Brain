# Framework Integration Guide

Universal AI Brain provides seamless integration with popular AI frameworks, enhancing them with advanced cognitive capabilities.

## Supported Frameworks

### 🤖 CrewAI Integration
**Multi-agent orchestration with cognitive enhancements**

```python
from ai_brain_python import UniversalAIBrainConfig
from ai_brain_python.adapters import CrewAIAdapter

# Initialize with cognitive capabilities
config = UniversalAIBrainConfig(
    mongodb_uri="your_mongodb_uri",
    voyage_api_key="your_voyage_key"
)

adapter = CrewAIAdapter(config)

# Create cognitive-enhanced agents
agent = adapter.create_cognitive_agent(
    role="Data Analyst",
    goal="Analyze data with emotional intelligence",
    backstory="An AI agent with advanced cognitive capabilities"
)

# Agents now have:
# - Emotional intelligence
# - Goal hierarchy management
# - Confidence tracking
# - Safety guardrails
# - Cultural awareness
```

### 🦜 LangChain Integration
**LLM applications with cognitive memory**

```python
from ai_brain_python.adapters import LangChainAdapter

adapter = LangChainAdapter(config)

# Create cognitive memory
memory = adapter.create_cognitive_memory()

# Enhanced with:
# - Semantic memory with vector embeddings
# - Working memory management
# - Memory decay and optimization
# - Cross-session memory sharing
```

### 🕸️ LangGraph Integration
**Stateful applications with cognitive workflows**

```python
from ai_brain_python.adapters import LangGraphAdapter

adapter = LangGraphAdapter(config)

# Create cognitive graph
graph = adapter.create_cognitive_graph()

# Enhanced with:
# - Temporal planning
# - Attention management
# - Workflow orchestration
# - Real-time monitoring
```

### 🔧 Pydantic AI Integration
**Type-safe agents with cognitive validation**

```python
from ai_brain_python.adapters import PydanticAIAdapter

adapter = PydanticAIAdapter(config)

# Enhanced with:
# - Skill capability assessment
# - Communication protocol optimization
# - Self-improvement tracking
# - Performance analytics
```

### ⚡ Agno Integration
**Advanced agents with metacognitive awareness**

```python
from ai_brain_python.adapters import AgnoAdapter

adapter = AgnoAdapter(config)

# Enhanced with:
# - Metacognitive awareness
# - Adaptive learning
# - Cognitive load monitoring
# - Strategy optimization
```

## Quick Start

### 1. Installation

```bash
# Install Universal AI Brain
pip install ai-brain-python

# Install your preferred framework
pip install crewai  # For CrewAI
pip install langchain  # For LangChain
pip install langgraph  # For LangGraph
pip install pydantic-ai  # For Pydantic AI
pip install agno  # For Agno
```

### 2. Basic Setup

```python
from ai_brain_python import UniversalAIBrainConfig
from ai_brain_python.adapters import get_available_frameworks, create_adapter

# Configure AI Brain
config = UniversalAIBrainConfig(
    mongodb_uri="mongodb://localhost:27017/ai_brain",
    voyage_api_key="your_voyage_api_key",
    cognitive_systems=[
        "emotional_intelligence",
        "goal_hierarchy", 
        "confidence_tracking",
        "safety_guardrails"
    ]
)

# Check available frameworks
frameworks = get_available_frameworks()
print(f"Available frameworks: {frameworks}")

# Create adapter for your framework
adapter = create_adapter("crewai", config)  # or "langchain", "langgraph", etc.
```

### 3. Framework-Specific Usage

#### CrewAI Example
```python
from ai_brain_python.adapters import CrewAIAdapter

adapter = CrewAIAdapter(config)

# Create cognitive agents
researcher = adapter.create_cognitive_agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="Expert in AI research with emotional intelligence"
)

writer = adapter.create_cognitive_agent(
    role="Tech Content Strategist", 
    goal="Craft compelling content on tech advancements",
    backstory="Creative writer with cultural awareness"
)

# Create cognitive crew
crew = adapter.create_cognitive_crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task]
)

# Execute with cognitive enhancements
result = crew.kickoff()
```

#### LangChain Example
```python
from ai_brain_python.adapters import LangChainAdapter

adapter = LangChainAdapter(config)

# Create cognitive memory
memory = adapter.create_cognitive_memory()

# Create cognitive chain
chain = adapter.create_cognitive_chain(
    llm=your_llm,
    memory=memory,
    cognitive_systems=["emotional_intelligence", "safety_guardrails"]
)

# Use with cognitive enhancements
response = chain.run("Analyze this emotional text with safety checks")
```

## Cognitive Enhancements

### 🧠 Core Cognitive Systems
- **Emotional Intelligence**: Sentiment analysis, emotion recognition, empathy modeling
- **Goal Hierarchy**: Multi-level goal management, priority optimization
- **Confidence Tracking**: Dynamic confidence assessment, uncertainty quantification
- **Safety Guardrails**: Risk assessment, content filtering, compliance monitoring

### 🎯 Advanced Capabilities
- **Cultural Knowledge**: Cross-cultural communication, cultural adaptation
- **Skill Capability**: Dynamic skill assessment, capability optimization
- **Temporal Planning**: Time-aware planning, scheduling optimization
- **Attention Management**: Focus allocation, attention optimization

### 🔄 Meta-Cognitive Features
- **Self-Improvement**: Continuous learning, performance optimization
- **Metacognitive Awareness**: Self-monitoring, strategy adaptation
- **Memory Management**: Intelligent memory decay, working memory optimization
- **Monitoring**: Real-time performance tracking, anomaly detection

## Configuration Options

### Basic Configuration
```python
config = UniversalAIBrainConfig(
    mongodb_uri="mongodb://localhost:27017/ai_brain",
    voyage_api_key="your_voyage_api_key"
)
```

### Advanced Configuration
```python
config = UniversalAIBrainConfig(
    mongodb_uri="mongodb://localhost:27017/ai_brain",
    voyage_api_key="your_voyage_api_key",
    cognitive_systems=[
        "emotional_intelligence",
        "goal_hierarchy",
        "confidence_tracking", 
        "attention_management",
        "cultural_knowledge",
        "skill_capability",
        "communication_protocol",
        "temporal_planning",
        "semantic_memory",
        "safety_guardrails",
        "self_improvement",
        "monitoring"
    ],
    enable_hybrid_search=True,
    enable_real_time_monitoring=True,
    memory_decay_enabled=True,
    metacognitive_awareness=True
)
```

## Best Practices

### 1. Framework Selection
- **CrewAI**: Best for multi-agent systems requiring coordination
- **LangChain**: Ideal for LLM applications with complex memory needs
- **LangGraph**: Perfect for stateful, workflow-based applications
- **Pydantic AI**: Great for type-safe, validated AI applications
- **Agno**: Excellent for advanced metacognitive capabilities

### 2. Cognitive System Selection
Choose cognitive systems based on your use case:
- **Customer Service**: emotional_intelligence, cultural_knowledge, safety_guardrails
- **Research**: goal_hierarchy, attention_management, semantic_memory
- **Creative Tasks**: emotional_intelligence, self_improvement, metacognitive_awareness
- **Critical Systems**: safety_guardrails, confidence_tracking, monitoring

### 3. Performance Optimization
- Enable only needed cognitive systems for better performance
- Use hybrid search for large-scale memory operations
- Monitor cognitive load and adjust accordingly
- Implement proper error handling and fallbacks

## Troubleshooting

### Common Issues

1. **Framework Not Available**
   ```python
   from ai_brain_python.adapters import get_available_frameworks
   print(get_available_frameworks())  # Check what's installed
   ```

2. **MongoDB Connection Issues**
   - Verify MongoDB URI is correct
   - Ensure MongoDB is running
   - Check network connectivity

3. **API Key Issues**
   - Verify Voyage API key is valid
   - Check API rate limits
   - Ensure proper environment setup

### Getting Help
- Check the [documentation](https://github.com/romiluz13/ai_brain_js)
- Review [examples](./examples/)
- Open an [issue](https://github.com/romiluz13/ai_brain_js/issues)

## Examples

See the [examples directory](./examples/) for complete working examples with each framework.
