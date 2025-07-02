"""
Cognitive System Interfaces - Python Dataclasses (converted from JavaScript TypeScript)

This module contains all the dataclass definitions that exactly match the TypeScript
interfaces from the JavaScript version. These ensure 1:1 compatibility between
JavaScript and Python implementations.

All interfaces maintain:
- Exact field names (converted from camelCase to snake_case)
- Identical data types and structures
- Same optional/required field patterns
- Compatible MongoDB serialization
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum


# ========================================
# EMOTIONAL INTELLIGENCE INTERFACES
# ========================================

@dataclass
class TaskContext:
    """Task context information."""
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    task_type: Optional[str] = None
    progress: Optional[float] = None


@dataclass
class UserContext:
    """User context information."""
    mood: Optional[str] = None
    urgency: Optional[float] = None
    satisfaction: Optional[float] = None


@dataclass
class ConversationMessage:
    """Single conversation message."""
    role: str
    content: str


@dataclass
class EmotionalContext:
    """Context for emotional intelligence processing."""
    agent_id: str
    input: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[ConversationMessage]] = None
    task_context: Optional[TaskContext] = None
    user_context: Optional[UserContext] = None


@dataclass
class EmotionDetectionResult:
    """Result of emotion detection analysis."""
    primary: str
    intensity: float
    valence: float
    arousal: float
    dominance: float
    confidence: float
    reasoning: str
    secondary: Optional[List[str]] = None


@dataclass
class EmotionalGuidance:
    """Emotional guidance for response generation."""
    response_style: str
    tone_adjustment: str
    empathy_level: float
    assertiveness_level: float
    support_level: float


@dataclass
class CognitiveImpact:
    """Cognitive impact of emotional state."""
    attention_focus: List[str]
    memory_priority: float
    decision_bias: str
    risk_tolerance: float


@dataclass
class EmotionalResponse:
    """Complete emotional response with guidance."""
    current_emotion: Dict[str, Any]  # EmotionalState from collection
    emotional_guidance: EmotionalGuidance
    cognitive_impact: CognitiveImpact
    recommendations: List[str]


@dataclass
class EmotionalPattern:
    """Emotional pattern for learning."""
    trigger: str
    emotional_response: str
    effectiveness: float
    frequency: int


@dataclass
class EmotionalImprovement:
    """Emotional improvement suggestion."""
    area: str
    suggestion: str
    priority: float


@dataclass
class EmotionalCalibration:
    """Emotional calibration metrics."""
    accuracy: float
    bias: float
    consistency: float


@dataclass
class EmotionalLearning:
    """Emotional learning data."""
    patterns: List[EmotionalPattern]
    improvements: List[EmotionalImprovement]
    calibration: EmotionalCalibration


# ========================================
# WORKING MEMORY INTERFACES
# ========================================

@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory management."""
    max_memories_per_session: int = 50
    max_total_working_memories: int = 1000
    default_ttl_minutes: int = 30
    max_ttl_minutes: int = 240
    min_ttl_minutes: int = 5
    high_priority_threshold: float = 0.8
    low_priority_threshold: float = 0.3
    cleanup_interval_minutes: int = 15
    pressure_cleanup_threshold: float = 0.8
    access_count_for_promotion: int = 3
    importance_boost_on_access: float = 0.1


@dataclass
class WorkingMemoryMetadata:
    """Metadata for working memory items."""
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_type: Optional[str] = None
    related_memories: Optional[List[str]] = None
    promotion_candidate: Optional[bool] = None
    additional_fields: Optional[Dict[str, Any]] = None


@dataclass
class WorkingMemoryItem:
    """Working memory item with all metadata."""
    id: str
    content: str
    session_id: str
    framework: str
    priority: Literal['low', 'medium', 'high', 'critical']
    importance: float
    confidence: float
    created: datetime
    expires: datetime
    last_accessed: datetime
    access_count: int
    tags: List[str]
    metadata: WorkingMemoryMetadata


@dataclass
class MemoryPressureStats:
    """Memory pressure statistics and recommendations."""
    total_working_memories: int
    memories_per_session: Dict[str, int]
    average_age: float
    expired_memories: int
    high_priority_memories: int
    memory_pressure: float  # 0-1 scale
    recommended_action: Literal['none', 'cleanup', 'aggressive_cleanup', 'promote_memories']


# ========================================
# SAFETY GUARDRAILS INTERFACES
# ========================================

@dataclass
class SafetyRule:
    """Individual safety rule definition."""
    id: str
    type: Literal['content_filter', 'prompt_injection', 'output_validation', 'rate_limit', 'compliance']
    pattern: str  # String representation of pattern/regex
    action: Literal['block', 'warn', 'log', 'modify']
    description: str
    enabled: bool
    threshold: Optional[float] = None


@dataclass
class SafetyPolicy:
    """Safety policy containing multiple rules."""
    id: str
    name: str
    description: str
    enabled: bool
    severity: Literal['low', 'medium', 'high', 'critical']
    rules: List[SafetyRule]
    frameworks: List[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class SafetyContent:
    """Content involved in safety violation."""
    input: Optional[str] = None
    output: Optional[str] = None
    context: Optional[str] = None


@dataclass
class SafetyViolation:
    """Safety violation record."""
    id: str
    timestamp: datetime
    session_id: str
    violation_type: Literal['harmful_content', 'prompt_injection', 'policy_violation', 'rate_limit', 'compliance']
    severity: Literal['low', 'medium', 'high', 'critical']
    content: SafetyContent
    policy_id: str
    rule_id: str
    action: Literal['blocked', 'warned', 'logged', 'modified']
    framework: str
    metadata: Dict[str, Any]
    trace_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class SafetyViolationSummary:
    """Summary of safety violations."""
    type: str
    severity: str
    description: str
    confidence: float


@dataclass
class ContentAnalysisResult:
    """Result of content safety analysis."""
    is_safe: bool
    confidence: float
    violations: List[SafetyViolationSummary]
    suggested_action: Literal['allow', 'block', 'modify', 'review']
    modified_content: Optional[str] = None


@dataclass
class SafetyTrend:
    """Safety trend data point."""
    date: datetime
    violation_count: int
    blocked_count: int


@dataclass
class TopViolatedPolicy:
    """Top violated policy information."""
    policy_id: str
    policy_name: str
    violation_count: int


@dataclass
class SafetyTimeRange:
    """Time range for safety analytics."""
    start: datetime
    end: datetime


@dataclass
class SafetyAnalytics:
    """Comprehensive safety analytics."""
    time_range: SafetyTimeRange
    total_violations: int
    violations_by_type: Dict[str, int]
    violations_by_severity: Dict[str, int]
    violations_by_framework: Dict[str, int]
    top_violated_policies: List[TopViolatedPolicy]
    safety_trends: List[SafetyTrend]
    compliance_score: float  # 0-100


# ========================================
# GOAL HIERARCHY INTERFACES
# ========================================

@dataclass
class GoalRequest:
    """Request to create or update a goal."""
    title: str
    description: str
    priority: Literal['low', 'medium', 'high', 'critical']
    deadline: Optional[datetime] = None
    parent_goal_id: Optional[str] = None
    agent_id: Optional[str] = None
    framework: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GoalProgress:
    """Progress tracking for a goal."""
    completion_percentage: float
    milestones_completed: int
    milestones_total: int
    last_updated: datetime
    notes: Optional[str] = None


@dataclass
class Goal:
    """Complete goal definition."""
    id: str
    title: str
    description: str
    priority: Literal['low', 'medium', 'high', 'critical']
    status: Literal['not_started', 'in_progress', 'completed', 'cancelled', 'on_hold']
    created: datetime
    updated: datetime
    deadline: Optional[datetime] = None
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    progress: Optional[GoalProgress] = None
    agent_id: Optional[str] = None
    framework: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ========================================
# CONFIDENCE TRACKING INTERFACES
# ========================================

@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence assessment."""
    data_quality: float
    model_certainty: float
    historical_accuracy: float
    context_relevance: float
    complexity_penalty: float


@dataclass
class ConfidenceAssessment:
    """Complete confidence assessment."""
    overall_confidence: float
    confidence_level: Literal['very_low', 'low', 'medium', 'high', 'very_high']
    factors: ConfidenceFactors
    reasoning: str
    recommendations: List[str]
    uncertainty_sources: List[str]


# ========================================
# ANALOGICAL MAPPING INTERFACES
# ========================================

@dataclass
class AnalogyMapping:
    """Mapping between source and target domains."""
    source_domain: str
    target_domain: str
    mappings: Dict[str, str]
    confidence: float
    reasoning: str


@dataclass
class AnalogyResult:
    """Result of analogical reasoning."""
    analogies: List[AnalogyMapping]
    best_analogy: Optional[AnalogyMapping]
    insights: List[str]
    confidence: float
    reasoning: str


# ========================================
# CAUSAL REASONING INTERFACES
# ========================================

@dataclass
class CausalRelationship:
    """Causal relationship between events/concepts."""
    cause: str
    effect: str
    strength: float
    confidence: float
    evidence: List[str]
    mechanism: Optional[str] = None


@dataclass
class CausalChain:
    """Chain of causal relationships."""
    relationships: List[CausalRelationship]
    overall_confidence: float
    alternative_explanations: List[str]


@dataclass
class CausalAnalysis:
    """Complete causal analysis result."""
    primary_causes: List[str]
    causal_chains: List[CausalChain]
    confidence: float
    reasoning: str
    recommendations: List[str]


# ========================================
# TEMPORAL PLANNING INTERFACES
# ========================================

@dataclass
class TimeConstraint:
    """Time constraint for planning."""
    type: Literal['deadline', 'duration', 'dependency', 'availability']
    value: Union[datetime, int, str]
    description: str
    flexibility: float  # 0-1 scale


@dataclass
class PlanStep:
    """Individual step in a plan."""
    id: str
    title: str
    description: str
    estimated_duration: int  # minutes
    dependencies: List[str]
    resources_required: List[str]
    priority: float
    constraints: List[TimeConstraint]


@dataclass
class ExecutionPlan:
    """Complete execution plan."""
    id: str
    title: str
    description: str
    steps: List[PlanStep]
    total_duration: int  # minutes
    start_time: datetime
    end_time: datetime
    confidence: float
    alternatives: List[str]


# ========================================
# SKILL CAPABILITY INTERFACES
# ========================================

@dataclass
class SkillLevel:
    """Skill proficiency level."""
    skill_name: str
    current_level: float  # 0-1 scale
    confidence: float
    evidence: List[str]
    last_assessed: datetime


@dataclass
class SkillGap:
    """Identified skill gap."""
    skill_name: str
    required_level: float
    current_level: float
    gap_size: float
    priority: float
    development_suggestions: List[str]


@dataclass
class SkillAssessment:
    """Complete skill assessment."""
    agent_id: str
    skills: List[SkillLevel]
    gaps: List[SkillGap]
    overall_competency: float
    recommendations: List[str]
    assessment_date: datetime


# ========================================
# UTILITY FUNCTIONS
# ========================================

def to_dict(dataclass_instance) -> Dict[str, Any]:
    """Convert dataclass to dictionary for MongoDB storage."""
    from dataclasses import asdict
    result = asdict(dataclass_instance)
    
    # Convert datetime objects to ISO strings for MongoDB
    def convert_datetime(obj):
        if isinstance(obj, dict):
            return {k: convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetime(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    return convert_datetime(result)


def from_dict(data: Dict[str, Any], dataclass_type):
    """Convert dictionary from MongoDB to dataclass."""
    # Convert ISO strings back to datetime objects
    def convert_iso_strings(obj):
        if isinstance(obj, dict):
            return {k: convert_iso_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_iso_strings(item) for item in obj]
        elif isinstance(obj, str):
            try:
                # Try to parse as datetime
                return datetime.fromisoformat(obj.replace('Z', '+00:00'))
            except ValueError:
                return obj
        return obj
    
    converted_data = convert_iso_strings(data)
    return dataclass_type(**converted_data)
