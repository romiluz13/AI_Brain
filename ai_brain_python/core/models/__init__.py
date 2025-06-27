"""
AI Brain Python Core Models

This module contains all Pydantic models for the AI Brain system:
- Base models for cognitive data structures
- Input/Output models for system interfaces
- State models for each cognitive system
- Validation and serialization models

All models provide:
- Type safety with runtime validation
- JSON serialization/deserialization
- Field validation and constraints
- Documentation and examples
"""

from ai_brain_python.core.models.base_models import (
    BaseAIBrainModel,
    CognitiveInputData,
    CognitiveResponse,
    CognitiveRequest,
    CognitiveContext,
    ProcessingMetadata,
    ValidationResult,
)

from ai_brain_python.core.models.cognitive_states import (
    CognitiveState,
    EmotionalState,
    AttentionState,
    ConfidenceState,
    CulturalContext,
    SkillAssessment,
    CommunicationProtocol,
    TemporalPlan,
    SemanticMemory,
    SafetyAssessment,
    MonitoringMetrics,
    ToolValidation,
    WorkflowState,
    MultiModalData,
    HumanFeedback,
)

from ai_brain_python.core.models.goal_hierarchy import (
    Goal,
    GoalHierarchy,
    GoalStatus,
    GoalPriority,
    GoalRelationship,
)

from ai_brain_python.core.models.framework_models import (
    FrameworkConfig,
    CrewAIConfig,
    PydanticAIConfig,
    AgnoConfig,
    LangChainConfig,
    LangGraphConfig,
)

__all__ = [
    # Base models
    "BaseAIBrainModel",
    "CognitiveInputData",
    "CognitiveResponse",
    "CognitiveRequest",
    "CognitiveContext",
    "ProcessingMetadata",
    "ValidationResult",

    # Cognitive state models
    "CognitiveState",
    "EmotionalState",
    "AttentionState",
    "ConfidenceState",
    "CulturalContext",
    "SkillAssessment",
    "CommunicationProtocol",
    "TemporalPlan",
    "SemanticMemory",
    "SafetyAssessment",
    "MonitoringMetrics",
    "ToolValidation",
    "WorkflowState",
    "MultiModalData",
    "HumanFeedback",

    # Goal hierarchy models
    "Goal",
    "GoalHierarchy",
    "GoalStatus",
    "GoalPriority",
    "GoalRelationship",

    # Framework configuration models
    "FrameworkConfig",
    "CrewAIConfig",
    "PydanticAIConfig",
    "AgnoConfig",
    "LangChainConfig",
    "LangGraphConfig",
]