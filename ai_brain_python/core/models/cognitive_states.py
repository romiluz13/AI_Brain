"""
Cognitive State Models for AI Brain Python

Models for each of the 24 cognitive systems (matching JavaScript version exactly):

Memory Systems (4):
- Working Memory Manager
- Episodic Memory Engine
- Semantic Memory Engine
- Memory Decay Engine

Reasoning Systems (6):
- Analogical Mapping System
- Causal Reasoning Engine
- Attention Management System
- Confidence Tracking Engine
- Context Injection Engine
- Vector Search Engine

Emotional Systems (3):
- Emotional Intelligence Engine
- Social Intelligence Engine
- Cultural Knowledge Engine

Social Systems (3):
- Goal Hierarchy Manager
- Human Feedback Integration Engine
- Safety Guardrails Engine

Temporal Systems (2):
- Temporal Planning Engine
- Skill Capability Manager

Meta Systems (6):
- Self-Improvement Engine
- Multi-Modal Processing Engine
- Advanced Tool Interface
- Workflow Orchestration Engine
- Hybrid Search Engine
- Real-time Monitoring Engine
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator

from ai_brain_python.core.models.base_models import BaseAIBrainModel, ConfidenceLevel


class CognitiveSystemType(str, Enum):
    """Types of cognitive systems - All 24 systems matching JavaScript version."""
    # Memory Systems (4)
    WORKING_MEMORY = "working_memory"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    MEMORY_DECAY = "memory_decay"

    # Reasoning Systems (6)
    ANALOGICAL_MAPPING = "analogical_mapping"
    CAUSAL_REASONING = "causal_reasoning"
    ATTENTION_MANAGEMENT = "attention_management"
    CONFIDENCE_TRACKING = "confidence_tracking"
    CONTEXT_INJECTION = "context_injection"
    VECTOR_SEARCH = "vector_search"

    # Emotional Systems (3)
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    SOCIAL_INTELLIGENCE = "social_intelligence"
    CULTURAL_KNOWLEDGE = "cultural_knowledge"

    # Social Systems (3)
    GOAL_HIERARCHY = "goal_hierarchy"
    HUMAN_FEEDBACK = "human_feedback"
    SAFETY_GUARDRAILS = "safety_guardrails"

    # Temporal Systems (2)
    TEMPORAL_PLANNING = "temporal_planning"
    SKILL_CAPABILITY = "skill_capability"

    # Meta Systems (6)
    SELF_IMPROVEMENT = "self_improvement"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    TOOL_INTERFACE = "tool_interface"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    HYBRID_SEARCH = "hybrid_search"
    REALTIME_MONITORING = "realtime_monitoring"


class CognitiveState(BaseAIBrainModel):
    """Base cognitive system state."""
    
    system_type: CognitiveSystemType = Field(description="Type of cognitive system")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    
    # State information
    is_active: bool = Field(default=True, description="Whether the system is active")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in current state")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last state update")
    
    # Performance metrics
    processing_count: int = Field(default=0, ge=0, description="Number of processing operations")
    average_processing_time: float = Field(default=0.0, ge=0.0, description="Average processing time in ms")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate")
    
    # System-specific state data
    state_data: Dict[str, Any] = Field(default_factory=dict, description="System-specific state data")


# 1. Emotional Intelligence Engine
class EmotionType(str, Enum):
    """Types of emotions."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class EmotionalState(BaseAIBrainModel):
    """Emotional intelligence state model."""
    
    # Primary emotion detection
    primary_emotion: EmotionType = Field(description="Primary detected emotion")
    emotion_intensity: float = Field(ge=0.0, le=1.0, description="Intensity of the emotion")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in emotion detection")
    
    # Emotional dimensions
    valence: float = Field(ge=-1.0, le=1.0, description="Emotional valence (negative to positive)")
    arousal: float = Field(ge=0.0, le=1.0, description="Emotional arousal level")
    dominance: float = Field(ge=0.0, le=1.0, description="Emotional dominance level")
    
    # Secondary emotions
    secondary_emotions: Dict[EmotionType, float] = Field(default_factory=dict, description="Secondary emotions with intensities")
    
    # Empathy modeling
    empathy_response: Optional[str] = Field(default=None, description="Appropriate empathetic response")
    emotional_context: Dict[str, Any] = Field(default_factory=dict, description="Emotional context information")
    
    # Mood tracking
    mood_trend: Optional[str] = Field(default=None, description="Mood trend over time")
    mood_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Mood stability score")


# 2. Attention Management System
class AttentionType(str, Enum):
    """Types of attention."""
    FOCUSED = "focused"
    DIVIDED = "divided"
    SELECTIVE = "selective"
    SUSTAINED = "sustained"
    ALTERNATING = "alternating"


class AttentionState(BaseAIBrainModel):
    """Attention management state model."""
    
    # Attention allocation
    attention_type: AttentionType = Field(description="Current attention type")
    focus_level: float = Field(ge=0.0, le=1.0, description="Current focus level")
    attention_span: float = Field(ge=0.0, description="Attention span in seconds")
    
    # Attention distribution
    attention_allocation: Dict[str, float] = Field(default_factory=dict, description="Attention allocation across tasks")
    priority_queue: List[Dict[str, Any]] = Field(default_factory=list, description="Priority queue for attention")
    
    # Distraction management
    distraction_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Current distraction level")
    distraction_sources: List[str] = Field(default_factory=list, description="Sources of distraction")
    
    # Attention control
    attention_control_strategy: Optional[str] = Field(default=None, description="Current attention control strategy")
    cognitive_load: float = Field(default=0.5, ge=0.0, le=1.0, description="Current cognitive load")
    
    @validator('attention_allocation')
    def validate_attention_sum(cls, v):
        """Validate attention allocation sums to 1.0."""
        if v and abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("Attention allocation must sum to 1.0")
        return v


# 3. Confidence Tracking Engine
class ConfidenceState(BaseAIBrainModel):
    """Confidence tracking state model."""
    
    # Overall confidence
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall system confidence")
    confidence_level: ConfidenceLevel = Field(description="Categorical confidence level")
    
    # System-specific confidence
    system_confidence: Dict[str, float] = Field(default_factory=dict, description="Confidence per cognitive system")
    
    # Uncertainty quantification
    epistemic_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0, description="Knowledge uncertainty")
    aleatoric_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0, description="Data uncertainty")
    
    # Reliability metrics
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0, description="System reliability score")
    calibration_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence calibration score")
    
    # Historical tracking
    confidence_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historical confidence data")
    confidence_trend: Optional[str] = Field(default=None, description="Confidence trend direction")


# 4. Cultural Knowledge Engine
class CulturalDimension(str, Enum):
    """Cultural dimensions."""
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM = "individualism"
    MASCULINITY = "masculinity"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE = "indulgence"


class CulturalContext(BaseAIBrainModel):
    """Cultural knowledge state model."""
    
    # Cultural identification
    primary_culture: Optional[str] = Field(default=None, description="Primary cultural identifier")
    cultural_dimensions: Dict[CulturalDimension, float] = Field(default_factory=dict, description="Cultural dimension scores")
    
    # Language and communication
    primary_language: str = Field(default="en", description="Primary language")
    communication_style: Optional[str] = Field(default=None, description="Communication style preference")
    
    # Cultural adaptation
    adaptation_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Cultural adaptation level")
    cultural_sensitivity: float = Field(default=0.8, ge=0.0, le=1.0, description="Cultural sensitivity score")
    
    # Context awareness
    cultural_context: Dict[str, Any] = Field(default_factory=dict, description="Cultural context information")
    cultural_norms: List[str] = Field(default_factory=list, description="Relevant cultural norms")


# 5. Skill Capability Manager
class SkillLevel(str, Enum):
    """Skill proficiency levels."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SkillAssessment(BaseAIBrainModel):
    """Skill capability state model."""
    
    # Skill inventory
    skills: Dict[str, SkillLevel] = Field(default_factory=dict, description="Skill proficiency mapping")
    skill_confidence: Dict[str, float] = Field(default_factory=dict, description="Confidence in skill assessments")
    
    # Learning progress
    learning_goals: List[str] = Field(default_factory=list, description="Current learning goals")
    skill_gaps: List[str] = Field(default_factory=list, description="Identified skill gaps")
    
    # Performance tracking
    skill_usage_frequency: Dict[str, int] = Field(default_factory=dict, description="Skill usage frequency")
    skill_improvement_rate: Dict[str, float] = Field(default_factory=dict, description="Skill improvement rates")
    
    # Recommendations
    recommended_skills: List[str] = Field(default_factory=list, description="Recommended skills to develop")
    learning_path: List[Dict[str, Any]] = Field(default_factory=list, description="Suggested learning path")


# 6. Communication Protocol Manager
class CommunicationProtocol(BaseAIBrainModel):
    """Communication protocol state model."""
    
    # Protocol selection
    active_protocol: str = Field(description="Currently active communication protocol")
    available_protocols: List[str] = Field(default_factory=list, description="Available communication protocols")
    
    # Communication style
    formality_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Communication formality level")
    verbosity_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Communication verbosity level")
    
    # Adaptation
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User communication preferences")
    context_adaptation: Dict[str, Any] = Field(default_factory=dict, description="Context-based adaptations")
    
    # Performance metrics
    communication_effectiveness: float = Field(default=0.8, ge=0.0, le=1.0, description="Communication effectiveness score")
    user_satisfaction: float = Field(default=0.8, ge=0.0, le=1.0, description="User satisfaction with communication")


# 7. Temporal Planning Engine
class TemporalPlan(BaseAIBrainModel):
    """Temporal planning state model."""
    
    # Time awareness
    current_time: datetime = Field(default_factory=datetime.utcnow, description="Current system time")
    time_horizon: timedelta = Field(default=timedelta(hours=24), description="Planning time horizon")
    
    # Task scheduling
    scheduled_tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Scheduled tasks")
    task_priorities: Dict[str, int] = Field(default_factory=dict, description="Task priorities")
    
    # Deadline management
    deadlines: Dict[str, datetime] = Field(default_factory=dict, description="Task deadlines")
    deadline_pressure: float = Field(default=0.0, ge=0.0, le=1.0, description="Current deadline pressure")
    
    # Time optimization
    time_allocation: Dict[str, float] = Field(default_factory=dict, description="Time allocation per task")
    efficiency_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Time management efficiency")


# 8. Semantic Memory Engine
class SemanticMemory(BaseAIBrainModel):
    """Semantic memory state model."""
    
    # Memory content
    content: str = Field(description="Memory content")
    embedding: List[float] = Field(description="Vector embedding of content")
    
    # Memory metadata
    memory_type: str = Field(default="episodic", description="Type of memory")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Memory importance score")
    
    # Access patterns
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last access time")
    
    # Memory consolidation
    consolidation_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Memory consolidation level")
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Memory decay rate")
    
    # Associations
    related_memories: List[str] = Field(default_factory=list, description="Related memory IDs")
    semantic_tags: List[str] = Field(default_factory=list, description="Semantic tags")


# 9. Safety Guardrails Engine
class SafetyLevel(str, Enum):
    """Safety assessment levels."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class SafetyAssessment(BaseAIBrainModel):
    """Safety guardrails state model."""
    
    # Safety status
    safety_level: SafetyLevel = Field(description="Current safety level")
    is_safe: bool = Field(description="Whether content/action is safe")
    
    # Risk assessment
    risk_score: float = Field(ge=0.0, le=1.0, description="Overall risk score")
    risk_categories: Dict[str, float] = Field(default_factory=dict, description="Risk scores by category")
    
    # Violations and warnings
    violations: List[str] = Field(default_factory=list, description="Safety violations detected")
    warnings: List[str] = Field(default_factory=list, description="Safety warnings")
    
    # Compliance
    compliance_checks: Dict[str, bool] = Field(default_factory=dict, description="Compliance check results")
    policy_violations: List[str] = Field(default_factory=list, description="Policy violations")
    
    # Mitigation
    mitigation_actions: List[str] = Field(default_factory=list, description="Recommended mitigation actions")
    safety_recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")


# 10. Real-time Monitoring Engine
class MonitoringMetrics(BaseAIBrainModel):
    """Real-time monitoring state model."""
    
    # Performance metrics
    response_time_ms: float = Field(ge=0.0, description="Response time in milliseconds")
    throughput: float = Field(ge=0.0, description="Requests per second")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate")
    
    # Resource utilization
    cpu_usage: float = Field(ge=0.0, le=1.0, description="CPU usage percentage")
    memory_usage: float = Field(ge=0.0, le=1.0, description="Memory usage percentage")
    
    # System health
    health_score: float = Field(ge=0.0, le=1.0, description="Overall system health score")
    availability: float = Field(ge=0.0, le=1.0, description="System availability")
    
    # Alerts and notifications
    active_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active system alerts")
    alert_history: List[Dict[str, Any]] = Field(default_factory=list, description="Alert history")


# 11. Advanced Tool Interface
class ToolValidation(BaseAIBrainModel):
    """Tool interface state model."""
    
    # Tool information
    tool_name: str = Field(description="Name of the tool")
    tool_version: str = Field(description="Tool version")
    
    # Validation status
    is_validated: bool = Field(description="Whether tool is validated")
    validation_score: float = Field(ge=0.0, le=1.0, description="Tool validation score")
    
    # Performance metrics
    success_rate: float = Field(ge=0.0, le=1.0, description="Tool success rate")
    average_execution_time: float = Field(ge=0.0, description="Average execution time in ms")
    
    # Error handling
    error_count: int = Field(default=0, ge=0, description="Number of errors encountered")
    recovery_attempts: int = Field(default=0, ge=0, description="Number of recovery attempts")
    
    # Tool capabilities
    capabilities: List[str] = Field(default_factory=list, description="Tool capabilities")
    limitations: List[str] = Field(default_factory=list, description="Tool limitations")


# 12. Workflow Orchestration Engine
class WorkflowState(BaseAIBrainModel):
    """Workflow orchestration state model."""
    
    # Workflow information
    workflow_id: str = Field(description="Workflow identifier")
    workflow_name: str = Field(description="Workflow name")
    
    # Execution state
    current_step: str = Field(description="Current workflow step")
    completed_steps: List[str] = Field(default_factory=list, description="Completed workflow steps")
    
    # Routing and coordination
    routing_decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Routing decisions made")
    parallel_executions: Dict[str, Any] = Field(default_factory=dict, description="Parallel execution states")
    
    # Performance
    execution_time: float = Field(default=0.0, ge=0.0, description="Total execution time in ms")
    efficiency_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Workflow efficiency score")


# 13. Multi-Modal Processing Engine
class MultiModalData(BaseAIBrainModel):
    """Multi-modal processing state model."""
    
    # Input modalities
    text_data: Optional[str] = Field(default=None, description="Text input data")
    image_data: Optional[bytes] = Field(default=None, description="Image input data")
    audio_data: Optional[bytes] = Field(default=None, description="Audio input data")
    video_data: Optional[bytes] = Field(default=None, description="Video input data")
    
    # Processing results
    modality_confidence: Dict[str, float] = Field(default_factory=dict, description="Confidence per modality")
    cross_modal_alignment: float = Field(default=0.0, ge=0.0, le=1.0, description="Cross-modal alignment score")
    
    # Feature extraction
    extracted_features: Dict[str, Any] = Field(default_factory=dict, description="Extracted features per modality")
    fusion_strategy: Optional[str] = Field(default=None, description="Multi-modal fusion strategy used")
    
    # Output generation
    generated_content: Dict[str, Any] = Field(default_factory=dict, description="Generated content per modality")
    output_quality: float = Field(default=0.8, ge=0.0, le=1.0, description="Output quality score")


# 14. Human Feedback Integration Engine
class FeedbackType(str, Enum):
    """Types of human feedback."""
    APPROVAL = "approval"
    REJECTION = "rejection"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    RATING = "rating"


class HumanFeedback(BaseAIBrainModel):
    """Human feedback integration state model."""
    
    # Feedback information
    feedback_type: FeedbackType = Field(description="Type of feedback")
    feedback_content: str = Field(description="Feedback content")
    
    # Feedback source
    user_id: Optional[str] = Field(default=None, description="User providing feedback")
    feedback_context: Dict[str, Any] = Field(default_factory=dict, description="Context of feedback")
    
    # Feedback processing
    is_processed: bool = Field(default=False, description="Whether feedback has been processed")
    processing_result: Optional[Dict[str, Any]] = Field(default=None, description="Feedback processing result")
    
    # Learning integration
    learning_impact: float = Field(default=0.0, ge=0.0, le=1.0, description="Impact on learning")
    model_updates: List[str] = Field(default_factory=list, description="Model updates triggered by feedback")
    
    # Approval workflows
    requires_approval: bool = Field(default=False, description="Whether action requires approval")
    approval_status: Optional[str] = Field(default=None, description="Current approval status")
    approval_chain: List[Dict[str, Any]] = Field(default_factory=list, description="Approval workflow chain")
