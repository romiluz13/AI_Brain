"""
AI Brain Framework Adapters

Provides seamless integration between the Universal AI Brain and major Python AI frameworks.

Supported Frameworks:
- CrewAI: Multi-agent orchestration framework
- Pydantic AI: Type-safe AI agent framework
- Agno: Advanced AI agent framework
- LangChain: LLM application framework
- LangGraph: Stateful multi-actor applications

Features:
- Native framework integration with cognitive capabilities
- Unified adapter interface across all frameworks
- Framework-specific optimizations and enhancements
- Automatic framework detection and initialization
- Comprehensive error handling and fallbacks
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from ai_brain_python.core.universal_ai_brain import UniversalAIBrainConfig
from ai_brain_python.adapters.base_adapter import BaseFrameworkAdapter

logger = logging.getLogger(__name__)

# Framework availability flags
FRAMEWORK_AVAILABILITY = {
    "crewai": False,
    "pydantic_ai": False,
    "agno": False,
    "langchain": False,
    "langgraph": False
}

# Adapter registry
ADAPTER_REGISTRY: Dict[str, Type[BaseFrameworkAdapter]] = {}


def _check_framework_availability():
    """Check which frameworks are available."""
    global FRAMEWORK_AVAILABILITY

    # Check CrewAI
    try:
        import crewai
        FRAMEWORK_AVAILABILITY["crewai"] = True
    except ImportError:
        pass

    # Check Pydantic AI
    try:
        import pydantic_ai
        FRAMEWORK_AVAILABILITY["pydantic_ai"] = True
    except ImportError:
        pass

    # Check Agno
    try:
        import agno
        FRAMEWORK_AVAILABILITY["agno"] = True
    except ImportError:
        pass

    # Check LangChain
    try:
        import langchain
        FRAMEWORK_AVAILABILITY["langchain"] = True
    except ImportError:
        pass

    # Check LangGraph
    try:
        import langgraph
        FRAMEWORK_AVAILABILITY["langgraph"] = True
    except ImportError:
        pass


def _register_adapters():
    """Register all available framework adapters."""
    global ADAPTER_REGISTRY

    # Register CrewAI adapter
    if FRAMEWORK_AVAILABILITY["crewai"]:
        try:
            from ai_brain_python.adapters.crewai_adapter import CrewAIAdapter
            ADAPTER_REGISTRY["crewai"] = CrewAIAdapter
            logger.info("CrewAI adapter registered")
        except ImportError as e:
            logger.warning(f"Failed to register CrewAI adapter: {e}")

    # Register Pydantic AI adapter
    if FRAMEWORK_AVAILABILITY["pydantic_ai"]:
        try:
            from ai_brain_python.adapters.pydantic_ai_adapter import PydanticAIAdapter
            ADAPTER_REGISTRY["pydantic_ai"] = PydanticAIAdapter
            logger.info("Pydantic AI adapter registered")
        except ImportError as e:
            logger.warning(f"Failed to register Pydantic AI adapter: {e}")

    # Register Agno adapter
    if FRAMEWORK_AVAILABILITY["agno"]:
        try:
            from ai_brain_python.adapters.agno_adapter import AgnoAdapter
            ADAPTER_REGISTRY["agno"] = AgnoAdapter
            logger.info("Agno adapter registered")
        except ImportError as e:
            logger.warning(f"Failed to register Agno adapter: {e}")

    # Register LangChain adapter
    if FRAMEWORK_AVAILABILITY["langchain"]:
        try:
            from ai_brain_python.adapters.langchain_adapter import LangChainAdapter
            ADAPTER_REGISTRY["langchain"] = LangChainAdapter
            logger.info("LangChain adapter registered")
        except ImportError as e:
            logger.warning(f"Failed to register LangChain adapter: {e}")

    # Register LangGraph adapter
    if FRAMEWORK_AVAILABILITY["langgraph"]:
        try:
            from ai_brain_python.adapters.langgraph_adapter import LangGraphAdapter
            ADAPTER_REGISTRY["langgraph"] = LangGraphAdapter
            logger.info("LangGraph adapter registered")
        except ImportError as e:
            logger.warning(f"Failed to register LangGraph adapter: {e}")


def get_available_frameworks() -> List[str]:
    """Get list of available frameworks."""
    return [framework for framework, available in FRAMEWORK_AVAILABILITY.items() if available]


def get_registered_adapters() -> List[str]:
    """Get list of registered adapters."""
    return list(ADAPTER_REGISTRY.keys())


def create_adapter(framework: str, ai_brain_config: UniversalAIBrainConfig) -> BaseFrameworkAdapter:
    """
    Create an adapter for the specified framework.

    Args:
        framework: Name of the framework ('crewai', 'pydantic_ai', 'agno', 'langchain', 'langgraph')
        ai_brain_config: Configuration for the AI Brain

    Returns:
        Framework adapter instance

    Raises:
        ValueError: If framework is not supported or not available
        ImportError: If framework dependencies are not installed
    """
    if framework not in FRAMEWORK_AVAILABILITY:
        raise ValueError(f"Unknown framework: {framework}. Supported frameworks: {list(FRAMEWORK_AVAILABILITY.keys())}")

    if not FRAMEWORK_AVAILABILITY[framework]:
        raise ImportError(f"Framework '{framework}' is not available. Please install it first.")

    if framework not in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter for '{framework}' is not registered.")

    adapter_class = ADAPTER_REGISTRY[framework]
    return adapter_class(ai_brain_config)


async def initialize_adapter(framework: str, ai_brain_config: UniversalAIBrainConfig) -> BaseFrameworkAdapter:
    """
    Create and initialize an adapter for the specified framework.

    Args:
        framework: Name of the framework
        ai_brain_config: Configuration for the AI Brain

    Returns:
        Initialized framework adapter
    """
    adapter = create_adapter(framework, ai_brain_config)
    await adapter.initialize()
    return adapter


def get_framework_info(framework: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about frameworks and adapters.

    Args:
        framework: Specific framework to get info for, or None for all

    Returns:
        Framework information dictionary
    """
    if framework:
        if framework not in ADAPTER_REGISTRY:
            return {"error": f"Framework '{framework}' not available"}

        adapter_class = ADAPTER_REGISTRY[framework]
        # Create temporary instance to get info (without initialization)
        temp_config = UniversalAIBrainConfig()  # Minimal config for info only
        temp_adapter = adapter_class(temp_config)
        return temp_adapter.get_framework_info()

    # Return info for all frameworks
    all_info = {
        "available_frameworks": get_available_frameworks(),
        "registered_adapters": get_registered_adapters(),
        "framework_availability": FRAMEWORK_AVAILABILITY.copy(),
        "frameworks": {}
    }

    for framework_name in get_registered_adapters():
        try:
            adapter_class = ADAPTER_REGISTRY[framework_name]
            temp_config = UniversalAIBrainConfig()
            temp_adapter = adapter_class(temp_config)
            all_info["frameworks"][framework_name] = temp_adapter.get_framework_info()
        except Exception as e:
            all_info["frameworks"][framework_name] = {"error": str(e)}

    return all_info


def get_installation_instructions() -> Dict[str, str]:
    """Get installation instructions for missing frameworks."""
    instructions = {}

    for framework, available in FRAMEWORK_AVAILABILITY.items():
        if not available:
            if framework == "crewai":
                instructions[framework] = "pip install crewai"
            elif framework == "pydantic_ai":
                instructions[framework] = "pip install pydantic-ai"
            elif framework == "agno":
                instructions[framework] = "pip install agno"
            elif framework == "langchain":
                instructions[framework] = "pip install langchain"
            elif framework == "langgraph":
                instructions[framework] = "pip install langgraph"

    return instructions


class AdapterManager:
    """
    Manager for framework adapters.

    Provides centralized management of multiple framework adapters
    with automatic initialization and lifecycle management.
    """

    def __init__(self, ai_brain_config: UniversalAIBrainConfig):
        self.ai_brain_config = ai_brain_config
        self.adapters: Dict[str, BaseFrameworkAdapter] = {}
        self.initialized_adapters: List[str] = []

    async def initialize_adapter(self, framework: str) -> BaseFrameworkAdapter:
        """Initialize a specific framework adapter."""
        if framework in self.adapters:
            return self.adapters[framework]

        adapter = await initialize_adapter(framework, self.ai_brain_config)
        self.adapters[framework] = adapter
        self.initialized_adapters.append(framework)

        logger.info(f"Initialized {framework} adapter")
        return adapter

    async def initialize_all_available(self) -> Dict[str, BaseFrameworkAdapter]:
        """Initialize all available framework adapters."""
        available_frameworks = get_available_frameworks()

        for framework in available_frameworks:
            try:
                await self.initialize_adapter(framework)
            except Exception as e:
                logger.error(f"Failed to initialize {framework} adapter: {e}")

        return self.adapters.copy()

    def get_adapter(self, framework: str) -> Optional[BaseFrameworkAdapter]:
        """Get an initialized adapter."""
        return self.adapters.get(framework)

    async def shutdown_all(self) -> None:
        """Shutdown all initialized adapters."""
        for framework, adapter in self.adapters.items():
            try:
                await adapter.shutdown()
                logger.info(f"Shutdown {framework} adapter")
            except Exception as e:
                logger.error(f"Error shutting down {framework} adapter: {e}")

        self.adapters.clear()
        self.initialized_adapters.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get status of all adapters."""
        return {
            "initialized_adapters": self.initialized_adapters.copy(),
            "available_frameworks": get_available_frameworks(),
            "adapter_count": len(self.adapters),
            "adapters": {
                framework: adapter.get_usage_stats()
                for framework, adapter in self.adapters.items()
            }
        }


# Initialize framework detection and adapter registration
_check_framework_availability()
_register_adapters()

# Export main classes and functions
__all__ = [
    "BaseFrameworkAdapter",
    "AdapterManager",
    "create_adapter",
    "initialize_adapter",
    "get_available_frameworks",
    "get_registered_adapters",
    "get_framework_info",
    "get_installation_instructions",
    "FRAMEWORK_AVAILABILITY",
    "ADAPTER_REGISTRY"
]

# Framework-specific exports (only if available)
if FRAMEWORK_AVAILABILITY["crewai"]:
    try:
        from ai_brain_python.adapters.crewai_adapter import CrewAIAdapter, CognitiveAgent, CognitiveCrew
        __all__.extend(["CrewAIAdapter", "CognitiveAgent", "CognitiveCrew"])
    except ImportError:
        pass

if FRAMEWORK_AVAILABILITY["pydantic_ai"]:
    try:
        from ai_brain_python.adapters.pydantic_ai_adapter import PydanticAIAdapter, CognitiveAgent as PydanticCognitiveAgent, CognitiveResponse
        __all__.extend(["PydanticAIAdapter", "PydanticCognitiveAgent", "CognitiveResponse"])
    except ImportError:
        pass

if FRAMEWORK_AVAILABILITY["agno"]:
    try:
        from ai_brain_python.adapters.agno_adapter import AgnoAdapter, CognitiveAgnoAgent, CognitiveAgnoWorkflow
        __all__.extend(["AgnoAdapter", "CognitiveAgnoAgent", "CognitiveAgnoWorkflow"])
    except ImportError:
        pass

if FRAMEWORK_AVAILABILITY["langchain"]:
    try:
        from ai_brain_python.adapters.langchain_adapter import LangChainAdapter, CognitiveMemory, CognitiveTool, CognitiveChain
        __all__.extend(["LangChainAdapter", "CognitiveMemory", "CognitiveTool", "CognitiveChain"])
    except ImportError:
        pass

if FRAMEWORK_AVAILABILITY["langgraph"]:
    try:
        from ai_brain_python.adapters.langgraph_adapter import LangGraphAdapter, CognitiveGraph, CognitiveNode, CognitiveState
        __all__.extend(["LangGraphAdapter", "CognitiveGraph", "CognitiveNode", "CognitiveState"])
    except ImportError:
        pass

logger.info(f"AI Brain adapters initialized. Available frameworks: {get_available_frameworks()}")
