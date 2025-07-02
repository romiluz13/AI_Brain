"""
AI Brain Python - Universal AI Brain with Multi-Framework Support

A sophisticated cognitive architecture system that provides 16 specialized cognitive systems
with seamless integration across CrewAI, Pydantic AI, Agno, LangChain, and LangGraph frameworks.

Core Features:
- 16 Cognitive Systems (Emotional Intelligence, Goal Hierarchy, Attention Management, etc.)
- Multi-Framework Adapters (CrewAI, Pydantic AI, Agno, LangChain, LangGraph)
- MongoDB Vector Search Integration
- Real-time Monitoring and Safety Guardrails
- Async/Await Performance Optimization
- Type-Safe Pydantic Models

Example Usage:
    ```python
    from ai_brain_python import UniversalAIBrain
    from ai_brain_python.models import CognitiveInputData
    
    # Initialize the AI Brain
    brain = UniversalAIBrain()
    await brain.initialize()
    
    # Process input through cognitive systems
    input_data = CognitiveInputData(
        text="I'm feeling excited about this new project!",
        context={"user_id": "user123"}
    )
    
    response = await brain.process_input(input_data)
    print(f"Emotional State: {response.emotional_state}")
    print(f"Confidence: {response.confidence}")
    ```

Framework Integration:
    ```python
    # Use with CrewAI
    from ai_brain_python.adapters import CrewAIAdapter
    adapter = CrewAIAdapter()
    await adapter.initialize()
    
    # Use with LangGraph
    from ai_brain_python.adapters import LangGraphAdapter
    adapter = LangGraphAdapter()
    workflow = await adapter.create_workflow()
    ```
"""

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models import (
    CognitiveInputData,
    CognitiveResponse,
    CognitiveState,
    EmotionalState,
    GoalHierarchy,
)

# Version information
__version__ = "0.1.0"
__author__ = "AI Brain Team"
__email__ = "team@aibrain.dev"
__license__ = "MIT"

# Public API
__all__ = [
    "UniversalAIBrain",
    "UniversalAIBrainConfig",
    "CognitiveInputData", 
    "CognitiveResponse",
    "CognitiveState",
    "EmotionalState",
    "GoalHierarchy",
    "__version__",
]

# Framework availability check
def check_framework_availability():
    """Check which AI frameworks are available in the current environment."""
    frameworks = {}
    
    try:
        import crewai
        frameworks["crewai"] = crewai.__version__
    except ImportError:
        frameworks["crewai"] = None
    
    try:
        import pydantic_ai
        frameworks["pydantic_ai"] = pydantic_ai.__version__
    except ImportError:
        frameworks["pydantic_ai"] = None
    
    try:
        import agno
        frameworks["agno"] = agno.__version__
    except ImportError:
        frameworks["agno"] = None
    
    try:
        import langchain
        frameworks["langchain"] = langchain.__version__
    except ImportError:
        frameworks["langchain"] = None
    
    try:
        import langgraph
        frameworks["langgraph"] = langgraph.__version__
    except ImportError:
        frameworks["langgraph"] = None
    
    return frameworks

# Lazy loading for framework adapters
def get_adapter(framework_name: str):
    """Get a framework adapter by name."""
    if framework_name == "crewai":
        from ai_brain_python.adapters.crewai_adapter import CrewAIAdapter
        return CrewAIAdapter
    elif framework_name == "pydantic_ai":
        from ai_brain_python.adapters.pydantic_ai_adapter import PydanticAIAdapter
        return PydanticAIAdapter
    elif framework_name == "agno":
        from ai_brain_python.adapters.agno_adapter import AgnoAdapter
        return AgnoAdapter
    elif framework_name == "langchain":
        from ai_brain_python.adapters.langchain_adapter import LangChainAdapter
        return LangChainAdapter
    elif framework_name == "langgraph":
        from ai_brain_python.adapters.langgraph_adapter import LangGraphAdapter
        return LangGraphAdapter
    else:
        raise ValueError(f"Unknown framework: {framework_name}")
