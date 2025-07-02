# AI Brain Python 🧠

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)

**Universal AI Brain** - A sophisticated cognitive architecture system with 16 specialized cognitive systems and seamless integration across 5 major AI frameworks.

## 🌟 Features

### 16 Cognitive Systems
- 🎭 **Emotional Intelligence Engine** - Emotion detection, empathy modeling, mood tracking
- 🎯 **Goal Hierarchy Manager** - Hierarchical goal planning and achievement tracking  
- 🤔 **Confidence Tracking Engine** - Real-time uncertainty assessment and reliability tracking
- 👁️ **Attention Management System** - Dynamic attention allocation and focus control
- 🌍 **Cultural Knowledge Engine** - Cross-cultural intelligence and adaptation
- 🛠️ **Skill Capability Manager** - Dynamic skill acquisition and proficiency tracking
- 📡 **Communication Protocol Manager** - Multi-protocol communication management
- ⏰ **Temporal Planning Engine** - Time-aware task management and scheduling
- 🧠 **Semantic Memory Engine** - Perfect recall with MongoDB vector search
- 🛡️ **Safety Guardrails Engine** - Multi-layer safety and compliance systems
- 🚀 **Self-Improvement Engine** - Continuous learning and optimization
- 📊 **Real-time Monitoring Engine** - Live metrics and performance analytics
- 🔧 **Advanced Tool Interface** - Tool recovery and validation systems
- 🔄 **Workflow Orchestration Engine** - Intelligent routing and parallel processing
- 🎭 **Multi-Modal Processing Engine** - Image/audio/video processing
- 👥 **Human Feedback Integration Engine** - Approval workflows and learning

### Multi-Framework Support
- **CrewAI** - Multi-agent coordination with specialized cognitive agents
- **Pydantic AI** - Type-safe validation with structured outputs  
- **Agno** - High-performance reasoning with multi-modal support
- **LangChain** - LCEL chains with extensive tool ecosystem
- **LangGraph** - State machine workflows with human-in-the-loop

## 🚀 Quick Start

### Installation

```bash
# Install with all framework support
pip install ai-brain-python[all]

# Or install with specific frameworks
pip install ai-brain-python[crewai,langchain]
```

### Basic Usage

```python
import asyncio
from ai_brain_python import UniversalAIBrain
from ai_brain_python.models import CognitiveInputData

async def main():
    # Initialize the AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Process input through cognitive systems
    input_data = CognitiveInputData(
        text="I'm feeling excited about this new project!",
        context={"user_id": "user123"}
    )
    
    response = await brain.process_input(input_data)
    
    print(f"Emotional State: {response.emotional_state.primary_emotion}")
    print(f"Confidence: {response.confidence}")
    print(f"Goals Identified: {len(response.goal_hierarchy.primary_goals)}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Framework Integration

#### CrewAI Integration
```python
from ai_brain_python.adapters import CrewAIAdapter

adapter = CrewAIAdapter()
await adapter.initialize()

# Create cognitive crew with specialized agents
crew = await adapter.create_cognitive_crew()
result = await crew.kickoff({"input": "Analyze this complex problem"})
```

#### LangGraph Integration
```python
from ai_brain_python.adapters import LangGraphAdapter

adapter = LangGraphAdapter()
workflow = await adapter.create_cognitive_workflow()

# Run cognitive processing workflow
result = await workflow.ainvoke({
    "input_data": input_data,
    "requested_systems": ["emotional", "attention", "confidence"]
})
```

## 📖 Documentation

- [**Planning Document**](AI_BRAIN_PYTHON_PLANNING.md) - Complete project architecture and design
- [**Task Breakdown**](AI_BRAIN_PYTHON_TASKS.md) - Detailed implementation roadmap
- [**System Prompt**](AI_BRAIN_SYSTEM_PROMPT.md) - Development guidelines and standards
- [**API Reference**](docs/api/) - Complete API documentation
- [**Framework Guides**](docs/frameworks/) - Framework-specific tutorials
- [**Examples**](examples/) - Working code examples

## 🏗️ Architecture

```
ai_brain_python/
├── core/                          # Framework-agnostic core
│   ├── universal_ai_brain.py      # Main orchestrator
│   ├── cognitive_systems/         # 16 cognitive systems
│   ├── models/                    # Pydantic data models
│   └── interfaces/                # Abstract base classes
├── adapters/                      # Framework-specific adapters
│   ├── crewai_adapter.py
│   ├── pydantic_ai_adapter.py
│   ├── agno_adapter.py
│   ├── langchain_adapter.py
│   └── langgraph_adapter.py
├── storage/                       # MongoDB and vector storage
├── safety/                        # Safety and compliance
├── monitoring/                    # Real-time monitoring
└── examples/                      # Framework examples
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/ai-brain/ai-brain-python.git
cd ai-brain-python

# Install with development dependencies
poetry install --with dev

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Type checking
mypy ai_brain_python/

# Code formatting
black ai_brain_python/
isort ai_brain_python/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m performance           # Performance tests only

# Run framework-specific tests
pytest -m crewai                # CrewAI tests
pytest -m langchain             # LangChain tests
pytest -m langgraph             # LangGraph tests

# Run with coverage
pytest --cov=ai_brain_python --cov-report=html
```

## 📊 Performance

- **Response Time**: < 200ms for simple cognitive operations
- **Concurrency**: Supports 1000+ concurrent users
- **Memory Usage**: < 512MB per instance
- **Uptime**: 99.9% availability target

## 🛡️ Security

- **Input Validation**: Comprehensive safety guardrails
- **PII Detection**: Automatic detection and handling
- **Rate Limiting**: Built-in protection against abuse
- **Compliance**: GDPR and privacy-focused design

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original JavaScript AI Brain architecture
- CrewAI, Pydantic AI, Agno, LangChain, and LangGraph communities
- MongoDB for vector search capabilities
- Python async/await ecosystem

---

**Built with ❤️ by the AI Brain Team**
