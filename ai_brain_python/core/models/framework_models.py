"""
Framework Configuration Models for AI Brain Python

Models for configuring different AI frameworks:
- CrewAI configuration
- Pydantic AI configuration  
- Agno configuration
- LangChain configuration
- LangGraph configuration
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import Field, validator

from ai_brain_python.core.models.base_models import BaseAIBrainModel


class FrameworkType(str, Enum):
    """Supported AI frameworks."""
    CREWAI = "crewai"
    PYDANTIC_AI = "pydantic_ai"
    AGNO = "agno"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"


class ModelProvider(str, Enum):
    """LLM model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"


class FrameworkConfig(BaseAIBrainModel):
    """Base framework configuration."""
    
    framework_type: FrameworkType = Field(description="Type of AI framework")
    enabled: bool = Field(default=True, description="Whether framework is enabled")
    
    # Model configuration
    model_provider: ModelProvider = Field(description="LLM model provider")
    model_name: str = Field(description="Model name/identifier")
    api_key: Optional[str] = Field(default=None, description="API key for model provider")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    
    # Performance settings
    max_tokens: int = Field(default=4096, ge=1, le=32768, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    
    # Framework-specific settings
    framework_settings: Dict[str, Any] = Field(default_factory=dict, description="Framework-specific configuration")


class CrewAIConfig(FrameworkConfig):
    """CrewAI framework configuration."""

    framework_type: Literal[FrameworkType.CREWAI] = Field(default=FrameworkType.CREWAI)
    
    # CrewAI specific settings
    crew_name: str = Field(default="AI Brain Crew", description="Name of the CrewAI crew")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    memory: bool = Field(default=True, description="Enable crew memory")
    
    # Agent configuration
    max_agents: int = Field(default=16, ge=1, le=50, description="Maximum number of agents")
    agent_execution_timeout: int = Field(default=60, ge=10, le=600, description="Agent execution timeout")
    
    # Task management
    max_concurrent_tasks: int = Field(default=5, ge=1, le=20, description="Maximum concurrent tasks")
    task_delegation: bool = Field(default=True, description="Enable task delegation between agents")
    
    # Cognitive agent roles
    cognitive_agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "emotional_intelligence_agent": {
                "role": "Emotional Intelligence Specialist",
                "goal": "Analyze and respond to emotional content with empathy",
                "backstory": "Expert in emotional intelligence and empathetic communication"
            },
            "goal_hierarchy_agent": {
                "role": "Goal Planning Specialist", 
                "goal": "Manage and optimize goal hierarchies",
                "backstory": "Strategic planner with expertise in goal decomposition and prioritization"
            },
            "attention_management_agent": {
                "role": "Attention Management Specialist",
                "goal": "Optimize attention allocation and focus",
                "backstory": "Cognitive scientist specializing in attention and focus management"
            },
            "safety_guardrails_agent": {
                "role": "Safety and Compliance Officer",
                "goal": "Ensure all outputs meet safety and ethical standards",
                "backstory": "AI safety expert with deep knowledge of ethical AI principles"
            }
        },
        description="Configuration for cognitive system agents"
    )
    
    # Tools and capabilities
    enable_tools: bool = Field(default=True, description="Enable tool usage for agents")
    tool_timeout: int = Field(default=30, ge=5, le=120, description="Tool execution timeout")
    
    @validator('cognitive_agents')
    def validate_cognitive_agents(cls, v):
        """Validate cognitive agent configurations."""
        required_fields = ['role', 'goal', 'backstory']
        for agent_name, config in v.items():
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Agent {agent_name} missing required field: {field}")
        return v


class PydanticAIConfig(FrameworkConfig):
    """Pydantic AI framework configuration."""

    framework_type: Literal[FrameworkType.PYDANTIC_AI] = Field(default=FrameworkType.PYDANTIC_AI)
    
    # Pydantic AI specific settings
    strict_validation: bool = Field(default=True, description="Enable strict Pydantic validation")
    response_model_validation: bool = Field(default=True, description="Validate response models")
    
    # Agent configuration
    agent_name: str = Field(default="AI Brain Agent", description="Name of the Pydantic AI agent")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the agent")
    
    # Response models
    response_models: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Pydantic models for structured responses"
    )
    
    # Validation settings
    allow_partial_responses: bool = Field(default=False, description="Allow partial response validation")
    validation_retries: int = Field(default=2, ge=0, le=5, description="Validation retry attempts")
    
    # Type safety
    enforce_types: bool = Field(default=True, description="Enforce strict type checking")
    coerce_types: bool = Field(default=False, description="Allow type coercion")


class AgnoConfig(FrameworkConfig):
    """Agno framework configuration."""

    framework_type: Literal[FrameworkType.AGNO] = Field(default=FrameworkType.AGNO)
    
    # Agno specific settings
    reasoning_mode: str = Field(default="advanced", description="Reasoning mode (basic, advanced, expert)")
    multi_modal: bool = Field(default=True, description="Enable multi-modal processing")
    
    # Agent configuration
    agent_pool_size: int = Field(default=8, ge=1, le=32, description="Size of agent pool")
    reasoning_depth: int = Field(default=3, ge=1, le=10, description="Maximum reasoning depth")
    
    # Performance optimization
    parallel_reasoning: bool = Field(default=True, description="Enable parallel reasoning")
    reasoning_cache: bool = Field(default=True, description="Enable reasoning result caching")
    
    # Multi-modal settings
    vision_model: Optional[str] = Field(default=None, description="Vision model for image processing")
    audio_model: Optional[str] = Field(default=None, description="Audio model for speech processing")
    
    # Reasoning configuration
    reasoning_strategies: List[str] = Field(
        default_factory=lambda: ["chain_of_thought", "tree_of_thought", "reflection"],
        description="Enabled reasoning strategies"
    )
    
    @validator('reasoning_mode')
    def validate_reasoning_mode(cls, v):
        """Validate reasoning mode."""
        valid_modes = ["basic", "advanced", "expert"]
        if v not in valid_modes:
            raise ValueError(f"Reasoning mode must be one of: {valid_modes}")
        return v


class LangChainConfig(FrameworkConfig):
    """LangChain framework configuration."""

    framework_type: Literal[FrameworkType.LANGCHAIN] = Field(default=FrameworkType.LANGCHAIN)
    
    # LangChain specific settings
    chain_type: str = Field(default="sequential", description="Type of chain (sequential, parallel, conditional)")
    enable_memory: bool = Field(default=True, description="Enable conversation memory")
    
    # Chain configuration
    max_chain_length: int = Field(default=10, ge=1, le=50, description="Maximum chain length")
    chain_timeout: int = Field(default=120, ge=30, le=600, description="Chain execution timeout")
    
    # Memory configuration
    memory_type: str = Field(default="buffer", description="Type of memory (buffer, summary, vector)")
    memory_size: int = Field(default=1000, ge=100, le=10000, description="Memory buffer size")
    
    # Tool integration
    enable_tools: bool = Field(default=True, description="Enable LangChain tools")
    tool_selection_strategy: str = Field(default="auto", description="Tool selection strategy")
    
    # LCEL (LangChain Expression Language) settings
    enable_lcel: bool = Field(default=True, description="Enable LCEL chains")
    streaming: bool = Field(default=False, description="Enable streaming responses")
    
    # Cognitive chains
    cognitive_chains: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "emotional_analysis_chain": {
                "type": "sequential",
                "components": ["emotion_detector", "empathy_generator", "response_formatter"]
            },
            "goal_planning_chain": {
                "type": "conditional", 
                "components": ["goal_analyzer", "priority_assessor", "plan_generator"]
            },
            "safety_check_chain": {
                "type": "parallel",
                "components": ["content_filter", "bias_detector", "harm_assessor"]
            }
        },
        description="Configuration for cognitive processing chains"
    )
    
    @validator('chain_type')
    def validate_chain_type(cls, v):
        """Validate chain type."""
        valid_types = ["sequential", "parallel", "conditional", "custom"]
        if v not in valid_types:
            raise ValueError(f"Chain type must be one of: {valid_types}")
        return v


class LangGraphConfig(FrameworkConfig):
    """LangGraph framework configuration."""

    framework_type: Literal[FrameworkType.LANGGRAPH] = Field(default=FrameworkType.LANGGRAPH)
    
    # LangGraph specific settings
    graph_type: str = Field(default="state_machine", description="Type of graph (state_machine, workflow, dag)")
    enable_persistence: bool = Field(default=True, description="Enable state persistence")
    
    # State management
    state_backend: str = Field(default="memory", description="State backend (memory, redis, postgres)")
    checkpoint_frequency: int = Field(default=5, ge=1, le=100, description="Checkpoint frequency")
    
    # Workflow configuration
    max_workflow_steps: int = Field(default=50, ge=5, le=200, description="Maximum workflow steps")
    workflow_timeout: int = Field(default=300, ge=60, le=1800, description="Workflow timeout in seconds")
    
    # Human-in-the-loop
    enable_human_feedback: bool = Field(default=True, description="Enable human feedback integration")
    approval_required: bool = Field(default=False, description="Require human approval for actions")
    
    # Parallel execution
    enable_parallel_execution: bool = Field(default=True, description="Enable parallel node execution")
    max_parallel_nodes: int = Field(default=5, ge=1, le=20, description="Maximum parallel nodes")
    
    # Cognitive workflow nodes
    cognitive_nodes: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "input_processing": {
                "type": "processor",
                "systems": ["multimodal_processing", "safety_guardrails"],
                "parallel": True
            },
            "cognitive_analysis": {
                "type": "analyzer", 
                "systems": ["emotional_intelligence", "attention_management", "cultural_knowledge"],
                "parallel": True
            },
            "goal_planning": {
                "type": "planner",
                "systems": ["goal_hierarchy", "temporal_planning"],
                "parallel": False
            },
            "response_generation": {
                "type": "generator",
                "systems": ["communication_protocol", "confidence_tracking"],
                "parallel": False
            },
            "output_validation": {
                "type": "validator",
                "systems": ["safety_guardrails", "monitoring"],
                "parallel": True
            }
        },
        description="Configuration for cognitive workflow nodes"
    )
    
    # State transitions
    state_transitions: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "input_processing": ["cognitive_analysis"],
            "cognitive_analysis": ["goal_planning"],
            "goal_planning": ["response_generation"],
            "response_generation": ["output_validation"],
            "output_validation": ["END"]
        },
        description="State transition mappings"
    )
    
    @validator('graph_type')
    def validate_graph_type(cls, v):
        """Validate graph type."""
        valid_types = ["state_machine", "workflow", "dag", "custom"]
        if v not in valid_types:
            raise ValueError(f"Graph type must be one of: {valid_types}")
        return v
    
    @validator('state_backend')
    def validate_state_backend(cls, v):
        """Validate state backend."""
        valid_backends = ["memory", "redis", "postgres", "mongodb"]
        if v not in valid_backends:
            raise ValueError(f"State backend must be one of: {valid_backends}")
        return v


# Framework configuration factory
def create_framework_config(
    framework_type: FrameworkType,
    model_provider: ModelProvider,
    model_name: str,
    **kwargs
) -> FrameworkConfig:
    """Factory function to create framework configurations."""
    
    base_config = {
        "model_provider": model_provider,
        "model_name": model_name,
        **kwargs
    }
    
    if framework_type == FrameworkType.CREWAI:
        return CrewAIConfig(**base_config)
    elif framework_type == FrameworkType.PYDANTIC_AI:
        return PydanticAIConfig(**base_config)
    elif framework_type == FrameworkType.AGNO:
        return AgnoConfig(**base_config)
    elif framework_type == FrameworkType.LANGCHAIN:
        return LangChainConfig(**base_config)
    elif framework_type == FrameworkType.LANGGRAPH:
        return LangGraphConfig(**base_config)
    else:
        raise ValueError(f"Unsupported framework type: {framework_type}")


# Default configurations for each framework
DEFAULT_CONFIGS = {
    FrameworkType.CREWAI: {
        "model_provider": ModelProvider.OPENAI,
        "model_name": "gpt-4",
        "crew_name": "AI Brain Cognitive Crew",
        "verbose": False,
        "memory": True,
        "max_agents": 16
    },
    FrameworkType.PYDANTIC_AI: {
        "model_provider": ModelProvider.OPENAI,
        "model_name": "gpt-4",
        "strict_validation": True,
        "response_model_validation": True
    },
    FrameworkType.AGNO: {
        "model_provider": ModelProvider.ANTHROPIC,
        "model_name": "claude-3-sonnet",
        "reasoning_mode": "advanced",
        "multi_modal": True,
        "parallel_reasoning": True
    },
    FrameworkType.LANGCHAIN: {
        "model_provider": ModelProvider.OPENAI,
        "model_name": "gpt-4",
        "chain_type": "sequential",
        "enable_memory": True,
        "enable_lcel": True
    },
    FrameworkType.LANGGRAPH: {
        "model_provider": ModelProvider.OPENAI,
        "model_name": "gpt-4",
        "graph_type": "state_machine",
        "enable_persistence": True,
        "enable_human_feedback": True
    }
}
