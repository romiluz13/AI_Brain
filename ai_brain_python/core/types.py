"""
Core Types and Interfaces for Universal AI Brain Python

Exact Python equivalents of JavaScript TypeScript interfaces with:
- Identical field names, types, and validation rules
- Same optional fields and default values
- Matching enum values and constraints
- Precise Pydantic model equivalents
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId


# ============================================================================
# CORE BRAIN TYPES - Exact TypeScript Equivalents
# ============================================================================

class AgentStatus(str, Enum):
    """Exact equivalent of JavaScript AgentStatus enum."""
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    PAUSED = 'paused'
    ERROR = 'error'


class MemoryType(str, Enum):
    """Exact equivalent of JavaScript MemoryType enum."""
    CONVERSATION = 'conversation'
    FACT = 'fact'
    PROCEDURE = 'procedure'
    EXPERIENCE = 'experience'


class MemoryImportance(str, Enum):
    """Exact equivalent of JavaScript MemoryImportance enum."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class ToolStatus(str, Enum):
    """Exact equivalent of JavaScript ToolStatus enum."""
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    DEPRECATED = 'deprecated'
    ERROR = 'error'


class WorkflowStatus(str, Enum):
    """Exact equivalent of JavaScript WorkflowStatus enum."""
    DRAFT = 'draft'
    ACTIVE = 'active'
    PAUSED = 'paused'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class AgentConfiguration(BaseModel):
    """Exact equivalent of JavaScript AgentConfiguration interface."""
    model: str
    temperature: float
    max_tokens: int = Field(..., alias='maxTokens')
    system_prompt: Optional[str] = Field(None, alias='systemPrompt')
    tools: Optional[List[str]] = None
    
    class Config:
        allow_population_by_field_name = True


class Agent(BaseModel):
    """Exact equivalent of JavaScript Agent interface."""
    id_: Optional[str] = Field(None, alias='_id')
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    status: AgentStatus
    configuration: AgentConfiguration
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    last_active_at: Optional[datetime] = Field(None, alias='lastActiveAt')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    
    class Config:
        allow_population_by_field_name = True


class AgentMemory(BaseModel):
    """Exact equivalent of JavaScript AgentMemory interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    agent_id: str = Field(..., alias='agentId')
    conversation_id: Optional[str] = Field(None, alias='conversationId')
    memory_type: str = Field(..., alias='memoryType')
    content: str
    importance: float
    metadata: Dict[str, Any]
    expires_at: Optional[datetime] = Field(None, alias='expiresAt')
    type: Optional[MemoryType] = None
    access_count: Optional[int] = Field(None, alias='accessCount')
    last_accessed: Optional[datetime] = Field(None, alias='lastAccessed')
    
    class Config:
        allow_population_by_field_name = True


class AgentPerformanceMetrics(BaseModel):
    """Exact equivalent of JavaScript AgentPerformanceMetrics interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    agent_id: str = Field(..., alias='agentId')
    timestamp: datetime
    recorded_at: Optional[datetime] = Field(None, alias='recordedAt')
    response_time: float = Field(..., alias='responseTime')
    accuracy: float
    user_satisfaction: float = Field(..., alias='userSatisfaction')
    cost_per_operation: float = Field(..., alias='costPerOperation')
    error_count: int = Field(..., alias='errorCount')
    success_rate: float = Field(..., alias='successRate')
    value: Optional[Any] = None
    
    class Config:
        allow_population_by_field_name = True


class AgentTool(BaseModel):
    """Exact equivalent of JavaScript AgentTool interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    agent_id: str = Field(..., alias='agentId')
    name: str
    description: str
    category: str
    status: ToolStatus
    configuration: Dict[str, Any]
    metadata: Dict[str, Any]
    execution_count: Optional[int] = Field(None, alias='executionCount')
    total_cost: Optional[float] = Field(None, alias='totalCost')
    rate_limits: Optional[Dict[str, Any]] = Field(None, alias='rateLimits')
    value: Optional[Any] = None
    
    class Config:
        allow_population_by_field_name = True


class ToolExecution(BaseModel):
    """Exact equivalent of JavaScript ToolExecution interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    tool_id: str = Field(..., alias='toolId')
    agent_id: str = Field(..., alias='agentId')
    execution_id: str = Field(..., alias='executionId')
    status: str  # 'pending' | 'running' | 'completed' | 'failed'
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: datetime = Field(..., alias='startTime')
    end_time: Optional[datetime] = Field(None, alias='endTime')
    duration: Optional[float] = None
    cost: Optional[float] = None
    
    class Config:
        allow_population_by_field_name = True


class WorkflowStep(BaseModel):
    """Exact equivalent of JavaScript WorkflowStep interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    workflow_id: str = Field(..., alias='workflowId')
    step_id: str = Field(..., alias='stepId')
    name: str
    type: str
    configuration: Dict[str, Any]
    dependencies: List[str]
    status: str  # 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
    order: int
    metadata: Dict[str, Any]
    
    class Config:
        allow_population_by_field_name = True


class AgentWorkflow(BaseModel):
    """Exact equivalent of JavaScript AgentWorkflow interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    agent_id: str = Field(..., alias='agentId')
    name: str
    description: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    configuration: Dict[str, Any]
    metadata: Dict[str, Any]
    start_time: Optional[datetime] = Field(None, alias='startTime')
    end_time: Optional[datetime] = Field(None, alias='endTime')
    execution_count: Optional[int] = Field(None, alias='executionCount')
    variables: Optional[Dict[str, Any]] = None
    current_step_index: Optional[int] = Field(None, alias='currentStepIndex')
    value: Optional[Any] = None
    started_at: Optional[datetime] = Field(None, alias='startedAt')
    completed_at: Optional[datetime] = Field(None, alias='completedAt')
    
    class Config:
        allow_population_by_field_name = True


# ============================================================================
# BRAIN CONFIG TYPES - Exact TypeScript Equivalents
# ============================================================================

class EmbeddingConfig(BaseModel):
    """Exact equivalent of JavaScript BrainConfig.embeddingConfig."""
    provider: str  # 'openai' | 'cohere' | 'huggingface' | 'azure-openai'
    model: str
    api_key: str = Field(..., alias='apiKey')
    dimensions: int
    base_url: Optional[str] = Field(None, alias='baseUrl')

    class Config:
        allow_population_by_field_name = True


class VectorSearchConfig(BaseModel):
    """Exact equivalent of JavaScript BrainConfig.vectorSearchConfig."""
    index_name: str = Field(..., alias='indexName')
    collection_name: str = Field(..., alias='collectionName')
    min_score: float = Field(..., alias='minScore')
    max_results: int = Field(..., alias='maxResults')

    class Config:
        allow_population_by_field_name = True


class BrainFeatures(BaseModel):
    """Exact equivalent of JavaScript BrainConfig.features."""
    enable_hybrid_search: Optional[bool] = Field(None, alias='enableHybridSearch')
    enable_conversation_memory: Optional[bool] = Field(None, alias='enableConversationMemory')
    enable_knowledge_graph: Optional[bool] = Field(None, alias='enableKnowledgeGraph')
    enable_real_time_updates: Optional[bool] = Field(None, alias='enableRealTimeUpdates')

    class Config:
        allow_population_by_field_name = True


class MongoConfig(BaseModel):
    """Exact equivalent of JavaScript BrainConfig.mongoConfig."""
    uri: str
    db_name: str = Field(..., alias='dbName')

    class Config:
        allow_population_by_field_name = True


class BrainConfig(BaseModel):
    """Exact equivalent of JavaScript BrainConfig interface."""
    mongo_config: MongoConfig = Field(..., alias='mongoConfig')
    embedding_config: EmbeddingConfig = Field(..., alias='embeddingConfig')
    vector_search_config: VectorSearchConfig = Field(..., alias='vectorSearchConfig')
    features: Optional[BrainFeatures] = None

    class Config:
        allow_population_by_field_name = True


class Context(BaseModel):
    """Exact equivalent of JavaScript Context interface."""
    id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = Field(..., alias='relevanceScore')
    source: str
    timestamp: datetime
    type: Optional[str] = None  # 'semantic' | 'conversational' | 'knowledge_graph' | 'real_time'

    class Config:
        allow_population_by_field_name = True


class EnhancedPromptMetadata(BaseModel):
    """Exact equivalent of JavaScript EnhancedPrompt.metadata."""
    framework_type: str = Field(..., alias='frameworkType')
    enhancement_strategy: str = Field(..., alias='enhancementStrategy')
    context_sources: List[str] = Field(..., alias='contextSources')
    processing_time: float = Field(..., alias='processingTime')
    token_count: Optional[int] = Field(None, alias='tokenCount')

    class Config:
        allow_population_by_field_name = True


class EnhancedPrompt(BaseModel):
    """Exact equivalent of JavaScript EnhancedPrompt interface."""
    original_prompt: str = Field(..., alias='originalPrompt')
    enhanced_prompt: str = Field(..., alias='enhancedPrompt')
    injected_context: List[Context] = Field(..., alias='injectedContext')
    metadata: EnhancedPromptMetadata

    class Config:
        allow_population_by_field_name = True


class InteractionPerformance(BaseModel):
    """Exact equivalent of JavaScript Interaction.performance."""
    response_time: float = Field(..., alias='responseTime')
    context_retrieval_time: float = Field(..., alias='contextRetrievalTime')
    embedding_time: float = Field(..., alias='embeddingTime')

    class Config:
        allow_population_by_field_name = True


class Interaction(BaseModel):
    """Exact equivalent of JavaScript Interaction interface."""
    id: str
    conversation_id: str = Field(..., alias='conversationId')
    user_message: str = Field(..., alias='userMessage')
    assistant_response: str = Field(..., alias='assistantResponse')
    context: List[Context]
    metadata: Dict[str, Any]
    timestamp: datetime
    framework: str
    performance: Optional[InteractionPerformance] = None

    class Config:
        allow_population_by_field_name = True


class Conversation(BaseModel):
    """Exact equivalent of JavaScript Conversation interface."""
    id: str
    framework: str
    title: Optional[str] = None
    metadata: Dict[str, Any]
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')
    interaction_count: int = Field(..., alias='interactionCount')
    tags: Optional[List[str]] = None

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# FRAMEWORK ADAPTER TYPES - Exact TypeScript Equivalents
# ============================================================================

class FrameworkCapabilities(BaseModel):
    """Exact equivalent of JavaScript FrameworkCapabilities interface."""
    supports_streaming: bool = Field(..., alias='supportsStreaming')
    supports_tools: bool = Field(..., alias='supportsTools')
    supports_multi_modal: bool = Field(..., alias='supportsMultiModal')
    supports_memory: bool = Field(..., alias='supportsMemory')
    supported_models: List[str] = Field(..., alias='supportedModels')
    max_context_length: int = Field(..., alias='maxContextLength')

    class Config:
        allow_population_by_field_name = True


class FrameworkInfo(BaseModel):
    """Exact equivalent of JavaScript FrameworkInfo interface."""
    name: str
    version: str
    is_available: bool = Field(..., alias='isAvailable')
    is_compatible: bool = Field(..., alias='isCompatible')
    confidence: float
    capabilities: FrameworkCapabilities

    class Config:
        allow_population_by_field_name = True


class FrameworkDetectionResult(BaseModel):
    """Exact equivalent of JavaScript FrameworkDetectionResult interface."""
    detected_frameworks: List[FrameworkInfo] = Field(..., alias='detectedFrameworks')
    recommended_framework: Optional[str] = Field(None, alias='recommendedFramework')
    confidence: float
    suggestions: Optional[List[str]] = None
    recommended_adapter: Optional[str] = Field(None, alias='recommendedAdapter')

    class Config:
        allow_population_by_field_name = True


class AdapterConfig(BaseModel):
    """Exact equivalent of JavaScript AdapterConfig interface."""
    enable_memory_injection: bool = Field(..., alias='enableMemoryInjection')
    enable_context_enhancement: bool = Field(..., alias='enableContextEnhancement')
    enable_context_injection: Optional[bool] = Field(None, alias='enableContextInjection')
    enable_tool_integration: bool = Field(..., alias='enableToolIntegration')
    enable_learning: Optional[bool] = Field(None, alias='enableLearning')
    enable_safety_checks: Optional[bool] = Field(None, alias='enableSafetyChecks')
    enable_performance_monitoring: Optional[bool] = Field(None, alias='enablePerformanceMonitoring')
    max_context_items: int = Field(..., alias='maxContextItems')
    enhancement_strategy: str = Field(..., alias='enhancementStrategy')  # 'semantic' | 'hybrid' | 'conversational'

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# BASE DOCUMENT TYPE - Exact TypeScript Equivalent
# ============================================================================

class BaseDocument(BaseModel):
    """Exact equivalent of JavaScript BaseDocument interface."""
    id_: Optional[str] = Field(None, alias='_id')
    created_at: datetime = Field(..., alias='createdAt')
    updated_at: datetime = Field(..., alias='updatedAt')

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# COGNITIVE STATE MODELS - Exact JavaScript Equivalents
# ============================================================================

class EmotionalStateEmotions(BaseModel):
    """Exact equivalent of JavaScript EmotionalState.emotions."""
    primary: str = Field(..., description="Primary emotion: joy, sadness, anger, fear, surprise, disgust, trust, anticipation")
    secondary: Optional[List[str]] = Field(None, description="Complex emotions like frustration, excitement, etc.")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Emotion intensity 0.0 to 1.0")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence -1.0 (negative) to 1.0 (positive)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional arousal 0.0 (calm) to 1.0 (excited)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Emotional dominance 0.0 (submissive) to 1.0 (dominant)")


class EmotionalStateContext(BaseModel):
    """Exact equivalent of JavaScript EmotionalState.context."""
    trigger: str = Field(..., description="What caused this emotional state")
    trigger_type: str = Field(..., alias='triggerType', description="Type: user_input, task_completion, error, success, interaction, system_event")
    conversation_turn: int = Field(..., alias='conversationTurn', description="Conversation turn number")
    task_id: Optional[str] = Field(None, alias='taskId', description="Associated task ID")
    workflow_id: Optional[str] = Field(None, alias='workflowId', description="Associated workflow ID")
    previous_emotion: Optional[str] = Field(None, alias='previousEmotion', description="Previous emotional state")

    class Config:
        allow_population_by_field_name = True


class EmotionalStateCognitiveEffects(BaseModel):
    """Exact equivalent of JavaScript EmotionalState.cognitiveEffects."""
    attention_modification: float = Field(..., alias='attentionModification', ge=-1.0, le=1.0, description="How emotion affects attention")
    memory_strength: float = Field(..., alias='memoryStrength', ge=0.0, le=1.0, description="How memorable this emotional event is")
    decision_bias: float = Field(..., alias='decisionBias', ge=-1.0, le=1.0, description="How emotion biases decisions")
    response_style: str = Field(..., alias='responseStyle', description="Response style: analytical, empathetic, assertive, cautious, creative")

    class Config:
        allow_population_by_field_name = True


class EmotionalStateDecay(BaseModel):
    """Exact equivalent of JavaScript EmotionalState.decay."""
    half_life: float = Field(..., alias='halfLife', description="Minutes until emotion intensity halves")
    decay_function: str = Field(..., alias='decayFunction', description="Decay function: exponential, linear, logarithmic")
    baseline_return: float = Field(..., alias='baselineReturn', description="Minutes to return to emotional baseline")

    class Config:
        allow_population_by_field_name = True


class EmotionalStateMetadata(BaseModel):
    """Exact equivalent of JavaScript EmotionalState.metadata."""
    framework: str = Field(..., description="Framework used")
    model: str = Field(..., description="Model used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in emotion detection")
    source: str = Field(..., description="Source: detected, inferred, user_reported, system_generated")
    version: str = Field(..., description="Version")


class EmotionalState(BaseDocument):
    """Exact equivalent of JavaScript EmotionalState interface."""
    agent_id: str = Field(..., alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")
    timestamp: datetime = Field(..., description="Timestamp of emotional state")
    expires_at: Optional[datetime] = Field(None, alias='expiresAt', description="TTL field for automatic decay")

    # Core emotional dimensions
    emotions: EmotionalStateEmotions = Field(..., description="Core emotional dimensions")

    # Contextual information
    context: EmotionalStateContext = Field(..., description="Contextual information")

    # Cognitive impact
    cognitive_effects: EmotionalStateCognitiveEffects = Field(..., alias='cognitiveEffects', description="Cognitive impact")

    # Decay parameters
    decay: EmotionalStateDecay = Field(..., description="Decay parameters")

    # Metadata
    metadata: EmotionalStateMetadata = Field(..., description="Metadata")

    class Config:
        allow_population_by_field_name = True


class AttentionStatePrimary(BaseModel):
    """Exact equivalent of JavaScript AttentionState.attention.primary."""
    task_id: str = Field(..., alias='taskId', description="Task identifier")
    task_type: str = Field(..., alias='taskType', description="Task type: conversation, analysis, planning, execution, monitoring")
    focus: float = Field(..., ge=0.0, le=1.0, description="0-1 attention allocation to primary task")
    priority: str = Field(..., description="Priority: critical, high, medium, low")
    start_time: datetime = Field(..., alias='startTime', description="Task start time")
    estimated_duration: float = Field(..., alias='estimatedDuration', description="Estimated duration in minutes")

    class Config:
        allow_population_by_field_name = True


class AttentionStateSecondary(BaseModel):
    """Exact equivalent of JavaScript AttentionState.attention.secondary item."""
    task_id: str = Field(..., alias='taskId', description="Task identifier")
    task_type: str = Field(..., alias='taskType', description="Task type")
    focus: float = Field(..., ge=0.0, le=1.0, description="0-1 attention allocation")
    priority: str = Field(..., description="Priority level")
    background_processing: bool = Field(..., alias='backgroundProcessing', description="Background processing flag")

    class Config:
        allow_population_by_field_name = True


class AttentionStateEfficiency(BaseModel):
    """Exact equivalent of JavaScript AttentionState.attention.efficiency."""
    focus_quality: float = Field(..., alias='focusQuality', ge=0.0, le=1.0, description="0-1 how well attention is focused")
    task_switching_cost: float = Field(..., alias='taskSwitchingCost', ge=0.0, le=1.0, description="0-1 cost of switching between tasks")
    distraction_level: float = Field(..., alias='distractionLevel', ge=0.0, le=1.0, description="0-1 current distraction level")
    attention_stability: float = Field(..., alias='attentionStability', ge=0.0, le=1.0, description="0-1 stability of attention over time")

    class Config:
        allow_population_by_field_name = True


class AttentionStateAttention(BaseModel):
    """Exact equivalent of JavaScript AttentionState.attention."""
    primary: AttentionStatePrimary = Field(..., description="Primary attention allocation")
    secondary: List[AttentionStateSecondary] = Field(..., description="Secondary attention allocations")
    total_allocation: float = Field(..., alias='totalAllocation', description="Total attention allocation (should not exceed 1.0)")
    efficiency: AttentionStateEfficiency = Field(..., description="Attention efficiency metrics")

    class Config:
        allow_population_by_field_name = True


class CognitiveLoadBreakdown(BaseModel):
    """Exact equivalent of JavaScript AttentionState.cognitiveLoad.breakdown."""
    working_memory: float = Field(..., ge=0.0, le=1.0, description="0-1 working memory load")
    processing: float = Field(..., ge=0.0, le=1.0, description="0-1 processing load")
    decision_making: float = Field(..., ge=0.0, le=1.0, description="0-1 decision making load")
    communication: float = Field(..., ge=0.0, le=1.0, description="0-1 communication load")
    monitoring: float = Field(..., ge=0.0, le=1.0, description="0-1 monitoring load")


class CognitiveLoadManagement(BaseModel):
    """Exact equivalent of JavaScript AttentionState.cognitiveLoad.management."""
    load_shedding: bool = Field(..., alias='loadShedding', description="Actively reducing load")
    priority_filtering: bool = Field(..., alias='priorityFiltering', description="Filtering low priority items")
    batch_processing: bool = Field(..., alias='batchProcessing', description="Batching similar tasks")
    deferred_processing: List[str] = Field(..., alias='deferredProcessing', description="Tasks deferred due to load")

    class Config:
        allow_population_by_field_name = True


class CognitiveLoad(BaseModel):
    """Exact equivalent of JavaScript AttentionState.cognitiveLoad."""
    current: float = Field(..., ge=0.0, le=1.0, description="0-1 current cognitive load")
    capacity: float = Field(..., ge=0.0, le=1.0, description="0-1 maximum cognitive capacity")
    utilization: float = Field(..., description="current/capacity ratio")
    overload: bool = Field(..., description="True if exceeding safe capacity")
    breakdown: CognitiveLoadBreakdown = Field(..., description="Load breakdown by cognitive function")
    management: CognitiveLoadManagement = Field(..., description="Load management")


class AttentionState(BaseDocument):
    """Exact equivalent of JavaScript AttentionState interface."""
    agent_id: str = Field(..., alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")
    timestamp: datetime = Field(..., description="Timestamp")

    # Current attention allocation
    attention: AttentionStateAttention = Field(..., description="Current attention allocation")

    # Cognitive load monitoring
    cognitive_load: CognitiveLoad = Field(..., alias='cognitiveLoad', description="Cognitive load monitoring")

    class Config:
        allow_population_by_field_name = True


class AttentionFilter(BaseModel):
    """Exact equivalent of JavaScript AttentionFilter interface."""
    agent_id: Optional[str] = Field(None, alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")
    task_type: Optional[str] = Field(None, alias='taskType', description="Task type filter")
    priority: Optional[str] = Field(None, description="Priority filter")
    focus_threshold: Optional[float] = Field(None, alias='focusThreshold', description="Minimum focus level")
    created_after: Optional[datetime] = Field(None, alias='createdAfter', description="Created after date")
    created_before: Optional[datetime] = Field(None, alias='createdBefore', description="Created before date")

    class Config:
        allow_population_by_field_name = True


class AttentionAnalyticsOptions(BaseModel):
    """Exact equivalent of JavaScript AttentionAnalyticsOptions interface."""
    time_range: Optional[Dict[str, datetime]] = Field(None, alias='timeRange', description="Time range for analytics")
    agent_id: Optional[str] = Field(None, alias='agentId', description="Agent identifier filter")
    task_type: Optional[str] = Field(None, alias='taskType', description="Task type filter")
    include_efficiency_metrics: bool = Field(True, alias='includeEfficiencyMetrics', description="Include efficiency metrics")
    include_load_analysis: bool = Field(True, alias='includeLoadAnalysis', description="Include cognitive load analysis")

    class Config:
        allow_population_by_field_name = True


class ConfidenceAspects(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.confidence.aspects."""
    factual_accuracy: float = Field(..., alias='factualAccuracy', ge=0.0, le=1.0, description="Confidence in factual correctness")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Confidence in response completeness")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Confidence in response relevance")
    clarity: float = Field(..., ge=0.0, le=1.0, description="Confidence in response clarity")
    appropriateness: float = Field(..., ge=0.0, le=1.0, description="Confidence in response appropriateness")

    class Config:
        allow_population_by_field_name = True


class ConfidenceSources(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.confidence.sources."""
    model_intrinsic: float = Field(..., alias='modelIntrinsic', ge=0.0, le=1.0, description="Model's intrinsic confidence")
    retrieval_quality: float = Field(..., alias='retrievalQuality', ge=0.0, le=1.0, description="Quality of retrieved information")
    context_relevance: float = Field(..., alias='contextRelevance', ge=0.0, le=1.0, description="Relevance of context")
    historical_performance: float = Field(..., alias='historicalPerformance', ge=0.0, le=1.0, description="Historical performance in similar tasks")
    domain_expertise: float = Field(..., alias='domainExpertise', ge=0.0, le=1.0, description="Domain expertise level")

    class Config:
        allow_population_by_field_name = True


class ConfidenceRecord(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.confidence."""
    overall: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    epistemic: float = Field(..., ge=0.0, le=1.0, description="Knowledge-based uncertainty (what we don't know)")
    aleatoric: float = Field(..., ge=0.0, le=1.0, description="Data-based uncertainty (inherent randomness)")
    calibrated: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence (adjusted for historical accuracy)")
    aspects: ConfidenceAspects = Field(..., description="Confidence breakdown by aspect")
    sources: ConfidenceSources = Field(..., description="Confidence sources breakdown")


class ConfidenceContext(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.context."""
    task: str = Field(..., description="What task/decision this confidence relates to")
    task_type: str = Field(..., alias='taskType', description="Task type: prediction, classification, generation, reasoning, decision")
    domain: str = Field(..., description="Domain of expertise (e.g., 'customer_service', 'technical_support')")
    complexity: float = Field(..., ge=0.0, le=1.0, description="0-1 scale of task complexity")
    novelty: float = Field(..., ge=0.0, le=1.0, description="0-1 scale of how novel/unfamiliar the task is")
    stakes: str = Field(..., description="Importance of being correct: low, medium, high, critical")

    class Config:
        allow_population_by_field_name = True


class PredictionAlternative(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.prediction.alternatives item."""
    value: str = Field(..., description="Alternative prediction value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this alternative")
    reasoning: str = Field(..., description="Reasoning for this alternative")


class PredictionDistribution(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.prediction.distribution item."""
    value: str = Field(..., description="Prediction value")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of this value")


class Prediction(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.prediction."""
    type: str = Field(..., description="Prediction type: multiclass, binary, regression, etc.")
    value: str = Field(..., description="Primary prediction value")
    alternatives: List[PredictionAlternative] = Field(..., description="Alternative predictions")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of primary prediction")
    distribution: List[PredictionDistribution] = Field(..., description="Full probability distribution")


class TemporalConfidence(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.temporal."""
    decay_rate: float = Field(..., alias='decayRate', description="Rate of confidence decay")
    half_life: float = Field(..., alias='halfLife', description="Half-life of confidence in hours")
    expires_at: datetime = Field(..., alias='expiresAt', description="When confidence expires")
    seasonality: str = Field(..., description="Seasonality pattern: business_hours, daily, weekly, etc.")

    class Config:
        allow_population_by_field_name = True


class LearningMetrics(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.learning."""
    surprisal: float = Field(..., description="Surprisal value")
    information_gain: float = Field(..., alias='informationGain', description="Information gain from this prediction")
    model_update: bool = Field(..., alias='modelUpdate', description="Whether this should trigger model updates")
    confidence_adjustment: float = Field(..., alias='confidenceAdjustment', description="Suggested confidence adjustment")

    class Config:
        allow_population_by_field_name = True


class ConfidenceMetadata(BaseModel):
    """Exact equivalent of JavaScript ConfidenceRecord.metadata."""
    framework: str = Field(..., description="Framework used")
    model: str = Field(..., description="Model used")
    version: str = Field(..., description="Version")
    features: List[str] = Field(..., description="Features used for this prediction")
    computation_time: float = Field(..., alias='computationTime', description="Time taken to compute (ms)")
    memory_usage: Optional[float] = Field(None, alias='memoryUsage', description="Memory used (MB)")

    class Config:
        allow_population_by_field_name = True


class ConfidenceState(BaseDocument):
    """Exact equivalent of JavaScript ConfidenceRecord interface."""
    agent_id: str = Field(..., alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")
    timestamp: datetime = Field(..., description="Timestamp")

    # Context of the confidence measurement
    context: ConfidenceContext = Field(..., description="Context of the confidence measurement")

    # Multi-dimensional confidence measurements
    confidence: ConfidenceRecord = Field(..., description="Multi-dimensional confidence measurements")

    # Prediction details
    prediction: Prediction = Field(..., description="Prediction details")

    # Temporal aspects
    temporal: TemporalConfidence = Field(..., description="Temporal confidence aspects")

    # Learning metrics
    learning: LearningMetrics = Field(..., description="Learning metrics")

    # Metadata
    metadata: ConfidenceMetadata = Field(..., description="Metadata")

    class Config:
        allow_population_by_field_name = True


class ConfidenceAnalyticsOptions(BaseModel):
    """Exact equivalent of JavaScript ConfidenceAnalyticsOptions interface."""
    time_range: Optional[Dict[str, datetime]] = Field(None, alias='timeRange', description="Time range for analytics")
    agent_id: Optional[str] = Field(None, alias='agentId', description="Agent identifier filter")
    task_type: Optional[str] = Field(None, alias='taskType', description="Task type filter")
    domain: Optional[str] = Field(None, description="Domain filter")
    include_metadata: bool = Field(True, alias='includeMetadata', description="Include metadata in results")

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# EMOTIONAL INTELLIGENCE DATA MODELS - Exact JavaScript Equivalents
# ============================================================================

class EmotionDetectionResult(BaseModel):
    """Exact equivalent of JavaScript EmotionDetectionResult interface."""
    primary: str = Field(..., description="Primary emotion")
    secondary: Optional[List[str]] = Field(None, description="Secondary emotions")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Emotion intensity")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional arousal")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Emotional dominance")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    reasoning: str = Field(..., description="Reasoning for emotion detection")


class EmotionalContextTaskContext(BaseModel):
    """Exact equivalent of JavaScript EmotionalContext.taskContext."""
    task_id: Optional[str] = Field(None, alias='taskId', description="Task identifier")
    workflow_id: Optional[str] = Field(None, alias='workflowId', description="Workflow identifier")
    task_type: Optional[str] = Field(None, alias='taskType', description="Task type")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Task progress")

    class Config:
        allow_population_by_field_name = True


class EmotionalContextUserContext(BaseModel):
    """Exact equivalent of JavaScript EmotionalContext.userContext."""
    mood: Optional[str] = Field(None, description="User mood")
    urgency: Optional[float] = Field(None, ge=0.0, le=1.0, description="Urgency level")
    satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0, description="Satisfaction level")


class EmotionalContext(BaseModel):
    """Exact equivalent of JavaScript EmotionalContext interface."""
    agent_id: str = Field(..., alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")
    input: str = Field(..., description="Input text")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, alias='conversationHistory', description="Conversation history")
    task_context: Optional[EmotionalContextTaskContext] = Field(None, alias='taskContext', description="Task context")
    user_context: Optional[EmotionalContextUserContext] = Field(None, alias='userContext', description="User context")

    class Config:
        allow_population_by_field_name = True


class EmotionalGuidance(BaseModel):
    """Exact equivalent of JavaScript EmotionalResponse.emotionalGuidance."""
    response_style: str = Field(..., alias='responseStyle', description="Response style")
    tone_adjustment: str = Field(..., alias='toneAdjustment', description="Tone adjustment")
    empathy_level: float = Field(..., alias='empathyLevel', ge=0.0, le=1.0, description="Empathy level")
    assertiveness_level: float = Field(..., alias='assertivenessLevel', ge=0.0, le=1.0, description="Assertiveness level")
    support_level: float = Field(..., alias='supportLevel', ge=0.0, le=1.0, description="Support level")

    class Config:
        allow_population_by_field_name = True


class CognitiveImpact(BaseModel):
    """Exact equivalent of JavaScript EmotionalResponse.cognitiveImpact."""
    attention_focus: List[str] = Field(..., alias='attentionFocus', description="Attention focus areas")
    memory_priority: float = Field(..., alias='memoryPriority', ge=0.0, le=1.0, description="Memory priority")
    decision_bias: str = Field(..., alias='decisionBias', description="Decision bias")
    risk_tolerance: float = Field(..., alias='riskTolerance', ge=0.0, le=1.0, description="Risk tolerance")

    class Config:
        allow_population_by_field_name = True


class EmotionalResponse(BaseModel):
    """Exact equivalent of JavaScript EmotionalResponse interface."""
    current_emotion: EmotionalState = Field(..., alias='currentEmotion', description="Current emotional state")
    emotional_guidance: EmotionalGuidance = Field(..., alias='emotionalGuidance', description="Emotional guidance")
    cognitive_impact: CognitiveImpact = Field(..., alias='cognitiveImpact', description="Cognitive impact")
    recommendations: List[str] = Field(..., description="Recommendations")

    class Config:
        allow_population_by_field_name = True


class EmotionalPattern(BaseModel):
    """Exact equivalent of JavaScript EmotionalLearning.patterns item."""
    trigger: str = Field(..., description="Trigger")
    emotional_response: str = Field(..., alias='emotionalResponse', description="Emotional response")
    effectiveness: float = Field(..., ge=0.0, le=1.0, description="Effectiveness")
    frequency: int = Field(..., ge=0, description="Frequency")

    class Config:
        allow_population_by_field_name = True


class EmotionalImprovement(BaseModel):
    """Exact equivalent of JavaScript EmotionalLearning.improvements item."""
    area: str = Field(..., description="Improvement area")
    suggestion: str = Field(..., description="Suggestion")
    priority: float = Field(..., ge=0.0, le=1.0, description="Priority")


class EmotionalCalibration(BaseModel):
    """Exact equivalent of JavaScript EmotionalLearning.calibration."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy")
    bias: float = Field(..., ge=-1.0, le=1.0, description="Bias")
    consistency: float = Field(..., ge=0.0, le=1.0, description="Consistency")


class EmotionalLearning(BaseModel):
    """Exact equivalent of JavaScript EmotionalLearning interface."""
    patterns: List[EmotionalPattern] = Field(..., description="Emotional patterns")
    improvements: List[EmotionalImprovement] = Field(..., description="Improvements")
    calibration: EmotionalCalibration = Field(..., description="Calibration metrics")


# ============================================================================
# EMOTIONAL STATE FILTER AND UPDATE MODELS
# ============================================================================

class EmotionalStateFilter(BaseModel):
    """Exact equivalent of JavaScript EmotionalStateFilter interface."""
    agent_id: Optional[str] = Field(None, alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")
    emotions_primary: Optional[str] = Field(None, alias='emotions.primary', description="Primary emotion filter")
    emotions_intensity: Optional[Dict[str, float]] = Field(None, description="Intensity range filter")
    emotions_valence: Optional[Dict[str, float]] = Field(None, description="Valence range filter")
    context_trigger_type: Optional[str] = Field(None, alias='context.triggerType', description="Trigger type filter")
    timestamp: Optional[Dict[str, datetime]] = Field(None, description="Timestamp range filter")
    metadata_confidence: Optional[Dict[str, float]] = Field(None, description="Confidence range filter")

    class Config:
        allow_population_by_field_name = True


class EmotionalStateUpdateData(BaseModel):
    """Exact equivalent of JavaScript EmotionalStateUpdateData interface."""
    emotions_intensity: Optional[float] = Field(None, alias='emotions.intensity', ge=0.0, le=1.0)
    emotions_valence: Optional[float] = Field(None, alias='emotions.valence', ge=-1.0, le=1.0)
    emotions_arousal: Optional[float] = Field(None, alias='emotions.arousal', ge=0.0, le=1.0)
    emotions_dominance: Optional[float] = Field(None, alias='emotions.dominance', ge=0.0, le=1.0)
    cognitive_effects_attention_modification: Optional[float] = Field(None, alias='cognitiveEffects.attentionModification', ge=-1.0, le=1.0)
    cognitive_effects_memory_strength: Optional[float] = Field(None, alias='cognitiveEffects.memoryStrength', ge=0.0, le=1.0)
    cognitive_effects_decision_bias: Optional[float] = Field(None, alias='cognitiveEffects.decisionBias', ge=-1.0, le=1.0)
    decay_half_life: Optional[float] = Field(None, alias='decay.halfLife', description="Half life in minutes")
    expires_at: Optional[datetime] = Field(None, alias='expiresAt', description="Expiration time")

    class Config:
        allow_population_by_field_name = True


class EmotionalAnalyticsOptions(BaseModel):
    """Exact equivalent of JavaScript EmotionalAnalyticsOptions interface."""
    time_range: Optional[Dict[str, datetime]] = Field(None, alias='timeRange', description="Time range")
    group_by: Optional[str] = Field(None, alias='groupBy', description="Grouping: hour, day, week, month")
    include_decayed: Optional[bool] = Field(None, alias='includeDecayed', description="Include decayed emotions")
    emotion_types: Optional[List[str]] = Field(None, alias='emotionTypes', description="Emotion types to include")
    min_intensity: Optional[float] = Field(None, alias='minIntensity', ge=0.0, le=1.0, description="Minimum intensity")

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# GOAL HIERARCHY MODELS - Exact JavaScript Equivalents
# ============================================================================

class GoalDefinition(BaseModel):
    """Exact equivalent of JavaScript Goal.goal."""
    title: str = Field(..., description="Goal title")
    description: str = Field(..., description="Goal description")
    type: str = Field(..., description="Goal type: objective, task, milestone, action, constraint")
    priority: str = Field(..., description="Priority: critical, high, medium, low")
    category: str = Field(..., description="Goal category (e.g., 'customer_service', 'problem_solving', 'learning')")


class GoalProgress(BaseModel):
    """Exact equivalent of JavaScript Goal.progress."""
    percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage 0-100")
    completed_sub_goals: int = Field(..., alias='completedSubGoals', ge=0, description="Number of completed sub-goals")
    total_sub_goals: int = Field(..., alias='totalSubGoals', ge=0, description="Total number of sub-goals")
    last_updated: datetime = Field(..., alias='lastUpdated', description="Last progress update time")

    class Config:
        allow_population_by_field_name = True


class GoalTimeline(BaseModel):
    """Exact equivalent of JavaScript Goal.timeline."""
    estimated_duration: float = Field(..., alias='estimatedDuration', description="Estimated duration in minutes")
    actual_duration: Optional[float] = Field(None, alias='actualDuration', description="Actual duration in minutes")
    start_time: Optional[datetime] = Field(None, alias='startTime', description="Goal start time")
    end_time: Optional[datetime] = Field(None, alias='endTime', description="Goal end time")
    deadline: Optional[datetime] = Field(None, description="Goal deadline")

    class Config:
        allow_population_by_field_name = True


class GoalDependencies(BaseModel):
    """Exact equivalent of JavaScript Goal.dependencies."""
    required_goals: List[str] = Field(..., alias='requiredGoals', description="Goals that must be completed first")
    blocked_by: List[str] = Field(..., alias='blockedBy', description="Goals currently blocking this one")
    enables: List[str] = Field(..., description="Goals that this one enables")
    conflicts: List[str] = Field(..., description="Goals that conflict with this one")

    class Config:
        allow_population_by_field_name = True


class GoalSuccessCondition(BaseModel):
    """Exact equivalent of JavaScript Goal.successCriteria.conditions item."""
    type: str = Field(..., description="Condition type: metric, boolean, threshold, approval")
    description: str = Field(..., description="Condition description")
    target: Any = Field(..., description="Target value")
    current: Optional[Any] = Field(None, description="Current value")
    achieved: bool = Field(..., description="Whether condition is achieved")


class GoalSuccessCriteria(BaseModel):
    """Exact equivalent of JavaScript Goal.successCriteria."""
    conditions: List[GoalSuccessCondition] = Field(..., description="Success conditions")
    verification: str = Field(..., description="Verification method: automatic, manual, external")


class GoalRisk(BaseModel):
    """Exact equivalent of JavaScript Goal.context.risks item."""
    description: str = Field(..., description="Risk description")
    probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability 0-1")
    impact: float = Field(..., ge=0.0, le=1.0, description="Risk impact 0-1")
    mitigation: Optional[str] = Field(None, description="Risk mitigation strategy")


class GoalContext(BaseModel):
    """Exact equivalent of JavaScript Goal.context."""
    trigger: str = Field(..., description="What initiated this goal")
    reasoning: str = Field(..., description="Why this goal is important")
    assumptions: List[str] = Field(..., description="Assumptions made")
    risks: List[GoalRisk] = Field(..., description="Goal risks")


class GoalLearning(BaseModel):
    """Exact equivalent of JavaScript Goal.learning."""
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Goal difficulty 0-1")
    satisfaction: float = Field(..., ge=0.0, le=1.0, description="Satisfaction with outcome 0-1")
    lessons: List[str] = Field(..., description="Lessons learned")
    improvements: List[str] = Field(..., description="Suggested improvements")


class GoalMetadata(BaseModel):
    """Exact equivalent of JavaScript Goal.metadata."""
    framework: str = Field(..., description="Framework used")
    created_by: str = Field(..., alias='createdBy', description="Creator: agent, user, system")
    tags: List[str] = Field(..., description="Goal tags")
    version: str = Field(..., description="Version")

    class Config:
        allow_population_by_field_name = True


class Goal(BaseDocument):
    """Exact equivalent of JavaScript Goal interface."""
    agent_id: str = Field(..., alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")

    # Hierarchical structure using materialized paths
    path: str = Field(..., description="Materialized path (e.g., '/root/project1/task1/subtask1')")
    parent_id: Optional[str] = Field(None, alias='parentId', description="Parent goal ID")
    level: int = Field(..., ge=0, description="Hierarchy level (0 = root, 1 = top-level, etc.)")

    # Goal definition
    goal: GoalDefinition = Field(..., description="Goal definition")

    # Status and progress
    status: str = Field(..., description="Status: not_started, in_progress, blocked, completed, failed, cancelled")
    progress: GoalProgress = Field(..., description="Progress tracking")

    # Temporal aspects
    timeline: GoalTimeline = Field(..., description="Timeline information")

    # Dependencies and constraints
    dependencies: GoalDependencies = Field(..., description="Goal dependencies")

    # Success criteria
    success_criteria: GoalSuccessCriteria = Field(..., alias='successCriteria', description="Success criteria")

    # Context and reasoning
    context: GoalContext = Field(..., description="Goal context")

    # Learning and adaptation
    learning: GoalLearning = Field(..., description="Learning metrics")

    # Metadata
    metadata: GoalMetadata = Field(..., description="Goal metadata")

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# GOAL FILTER AND UPDATE MODELS
# ============================================================================

class GoalFilter(BaseModel):
    """Exact equivalent of JavaScript GoalFilter interface."""
    agent_id: Optional[str] = Field(None, alias='agentId', description="Agent identifier")
    session_id: Optional[str] = Field(None, alias='sessionId', description="Session identifier")
    path: Optional[Dict[str, str]] = Field(None, description="Path regex filter")
    level: Optional[int] = Field(None, description="Hierarchy level")
    goal_type: Optional[str] = Field(None, alias='goal.type', description="Goal type filter")
    goal_priority: Optional[str] = Field(None, alias='goal.priority', description="Goal priority filter")
    status: Optional[Union[str, Dict[str, List[str]]]] = Field(None, description="Status filter")
    progress_percentage: Optional[Dict[str, float]] = Field(None, alias='progress.percentage', description="Progress range filter")
    timeline_deadline: Optional[Dict[str, datetime]] = Field(None, alias='timeline.deadline', description="Deadline range filter")

    class Config:
        allow_population_by_field_name = True


class GoalUpdateData(BaseModel):
    """Exact equivalent of JavaScript GoalUpdateData interface."""
    status: Optional[str] = Field(None, description="Goal status")
    progress_percentage: Optional[float] = Field(None, alias='progress.percentage', ge=0.0, le=100.0, description="Progress percentage")
    progress_last_updated: Optional[datetime] = Field(None, alias='progress.lastUpdated', description="Progress update time")
    timeline_actual_duration: Optional[float] = Field(None, alias='timeline.actualDuration', description="Actual duration")
    timeline_end_time: Optional[datetime] = Field(None, alias='timeline.endTime', description="End time")
    success_criteria_conditions: Optional[List[Any]] = Field(None, alias='successCriteria.conditions', description="Success conditions")
    learning_difficulty: Optional[float] = Field(None, alias='learning.difficulty', ge=0.0, le=1.0, description="Difficulty rating")
    learning_satisfaction: Optional[float] = Field(None, alias='learning.satisfaction', ge=0.0, le=1.0, description="Satisfaction rating")
    learning_lessons: Optional[List[str]] = Field(None, alias='learning.lessons', description="Lessons learned")

    class Config:
        allow_population_by_field_name = True


class GoalAnalyticsOptions(BaseModel):
    """Exact equivalent of JavaScript GoalAnalyticsOptions interface."""
    time_range: Optional[Dict[str, datetime]] = Field(None, alias='timeRange', description="Time range filter")
    include_completed: Optional[bool] = Field(None, alias='includeCompleted', description="Include completed goals")
    group_by: Optional[str] = Field(None, alias='groupBy', description="Grouping: type, priority, category, level")
    min_progress: Optional[float] = Field(None, alias='minProgress', ge=0.0, le=100.0, description="Minimum progress")

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# GOAL DEPENDENCY ANALYSIS MODELS
# ============================================================================

class GoalDependencyAnalysis(BaseModel):
    """Exact equivalent of JavaScript getGoalDependencies return type."""
    required: List[Goal] = Field(..., description="Required goals")
    blockers: List[Goal] = Field(..., description="Blocking goals")
    enabled: List[Goal] = Field(..., description="Goals enabled by this one")
    conflicts: List[Goal] = Field(..., description="Conflicting goals")
    can_start: bool = Field(..., alias='canStart', description="Whether goal can start")

    class Config:
        allow_population_by_field_name = True


class GoalPriorityDistribution(BaseModel):
    """Goal priority distribution analysis."""
    priority: str = Field(..., description="Priority level")
    count: int = Field(..., description="Number of goals")
    avg_completion: float = Field(..., alias='avgCompletion', description="Average completion rate")

    class Config:
        allow_population_by_field_name = True


class GoalTypeAnalysis(BaseModel):
    """Goal type analysis."""
    type: str = Field(..., description="Goal type")
    count: int = Field(..., description="Number of goals")
    success_rate: float = Field(..., alias='successRate', description="Success rate")

    class Config:
        allow_population_by_field_name = True


class GoalDifficultyAnalysis(BaseModel):
    """Goal difficulty analysis."""
    avg_difficulty: float = Field(..., alias='avgDifficulty', description="Average difficulty")
    satisfaction_correlation: float = Field(..., alias='satisfactionCorrelation', description="Satisfaction correlation")

    class Config:
        allow_population_by_field_name = True


class GoalTimelineAccuracy(BaseModel):
    """Goal timeline accuracy analysis."""
    on_time_rate: float = Field(..., alias='onTimeRate', description="On-time completion rate")
    avg_delay: float = Field(..., alias='avgDelay', description="Average delay in days")

    class Config:
        allow_population_by_field_name = True


class GoalPatternAnalysis(BaseModel):
    """Exact equivalent of JavaScript analyzeGoalPatterns return type."""
    completion_rate: float = Field(..., alias='completionRate', description="Goal completion rate")
    avg_duration: float = Field(..., alias='avgDuration', description="Average duration")
    priority_distribution: List[GoalPriorityDistribution] = Field(..., alias='priorityDistribution', description="Priority distribution")
    type_analysis: List[GoalTypeAnalysis] = Field(..., alias='typeAnalysis', description="Type analysis")
    difficulty_analysis: GoalDifficultyAnalysis = Field(..., alias='difficultyAnalysis', description="Difficulty analysis")
    timeline_accuracy: GoalTimelineAccuracy = Field(..., alias='timelineAccuracy', description="Timeline accuracy")

    class Config:
        allow_population_by_field_name = True


class GoalLevelStats(BaseModel):
    """Goal statistics by level."""
    level: int = Field(..., description="Hierarchy level")
    count: int = Field(..., description="Number of goals at this level")


class GoalStats(BaseModel):
    """Exact equivalent of JavaScript getGoalStats return type."""
    total_goals: int = Field(..., alias='totalGoals', description="Total number of goals")
    active_goals: int = Field(..., alias='activeGoals', description="Number of active goals")
    completed_goals: int = Field(..., alias='completedGoals', description="Number of completed goals")
    avg_progress: float = Field(..., alias='avgProgress', description="Average progress")
    goals_by_level: List[GoalLevelStats] = Field(..., alias='goalsByLevel', description="Goals by hierarchy level")

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# MEMORY & CONTEXT MODELS - Exact JavaScript Equivalents
# ============================================================================

class MemoryFilter(BaseModel):
    """Exact equivalent of JavaScript MemoryFilter interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    agent_id: Optional[Union[str, ObjectId]] = Field(None, alias='agentId', description="Agent identifier")
    conversation_id: Optional[str] = Field(None, alias='conversationId', description="Conversation identifier")
    memory_type: Optional[MemoryType] = Field(None, alias='memoryType', description="Memory type")
    importance: Optional[MemoryImportance] = Field(None, description="Memory importance")
    tags: Optional[List[str]] = Field(None, description="Memory tags")
    created_after: Optional[datetime] = Field(None, alias='createdAfter', description="Created after date")
    created_before: Optional[datetime] = Field(None, alias='createdBefore', description="Created before date")
    expires_after: Optional[datetime] = Field(None, alias='expiresAfter', description="Expires after date")
    expires_before: Optional[datetime] = Field(None, alias='expiresBefore', description="Expires before date")


class MemoryUpdateData(BaseModel):
    """Exact equivalent of JavaScript MemoryUpdateData interface."""
    content: Optional[str] = Field(None, description="Memory content")
    importance: Optional[MemoryImportance] = Field(None, description="Memory importance")
    tags: Optional[List[str]] = Field(None, description="Memory tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Memory metadata")
    expires_at: Optional[datetime] = Field(None, alias='expiresAt', description="Expiration time")
    access_count: Optional[int] = Field(None, alias='accessCount', description="Access count")
    last_accessed_at: Optional[datetime] = Field(None, alias='lastAccessedAt', description="Last access time")

    class Config:
        allow_population_by_field_name = True


class MemorySearchOptions(BaseModel):
    """Exact equivalent of JavaScript MemorySearchOptions interface."""
    limit: Optional[int] = Field(None, description="Maximum number of results")
    time_range: Optional[Dict[str, datetime]] = Field(None, alias='timeRange', description="Time range filter")
    conversation_id: Optional[str] = Field(None, alias='conversationId', description="Conversation identifier")
    framework: Optional[str] = Field(None, description="Framework filter")
    min_relevance_score: Optional[float] = Field(None, alias='minRelevanceScore', description="Minimum relevance score")
    search_type: Optional[str] = Field(None, alias='searchType', description="Search type: semantic, hybrid, text")
    include_metadata: Optional[bool] = Field(None, alias='includeMetadata', description="Include metadata")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    min_importance: Optional[MemoryImportance] = Field(None, alias='minImportance', description="Minimum importance")
    include_expired: Optional[bool] = Field(None, alias='includeExpired', description="Include expired memories")
    sort_by: Optional[str] = Field(None, alias='sortBy', description="Sort by: relevance, importance, recency, access_count")

    class Config:
        allow_population_by_field_name = True


class VectorSearchOptions(BaseModel):
    """Exact equivalent of JavaScript VectorSearchOptions interface."""
    limit: Optional[int] = Field(None, description="Maximum number of results")
    num_candidates: Optional[int] = Field(None, alias='numCandidates', description="Number of candidates")
    filter: Optional[Dict[str, Any]] = Field(None, description="Search filter")
    min_score: Optional[float] = Field(None, alias='minScore', description="Minimum score")
    index: Optional[str] = Field(None, description="Index name")
    include_embeddings: Optional[bool] = Field(None, alias='includeEmbeddings', description="Include embeddings")

    class Config:
        allow_population_by_field_name = True


class VectorDocument(BaseModel):
    """Exact equivalent of JavaScript VectorDocument interface."""
    id: str = Field(..., description="Document identifier")
    text: str = Field(..., description="Document text")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    source: str = Field(..., description="Document source")
    timestamp: datetime = Field(..., description="Document timestamp")
    chunk_index: Optional[int] = Field(None, alias='chunkIndex', description="Chunk index")
    parent_document_id: Optional[str] = Field(None, alias='parentDocumentId', description="Parent document ID")

    class Config:
        allow_population_by_field_name = True


class EmbeddingRequest(BaseModel):
    """Exact equivalent of JavaScript EmbeddingRequest interface."""
    text: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Embedding model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Request metadata")


class EmbeddingResponse(BaseModel):
    """Exact equivalent of JavaScript EmbeddingResponse interface."""
    embedding: List[float] = Field(..., description="Vector embedding")
    model: str = Field(..., description="Model used")
    dimensions: int = Field(..., description="Embedding dimensions")
    token_count: Optional[int] = Field(None, alias='tokenCount', description="Token count")
    processing_time: float = Field(..., alias='processingTime', description="Processing time")

    class Config:
        allow_population_by_field_name = True


class KnowledgeNode(BaseModel):
    """Exact equivalent of JavaScript KnowledgeNode interface."""
    id: str = Field(..., description="Node identifier")
    type: str = Field(..., description="Node type")
    content: str = Field(..., description="Node content")
    metadata: Dict[str, Any] = Field(..., description="Node metadata")
    embedding: Optional[List[float]] = Field(None, description="Node embedding")
    created_at: datetime = Field(..., alias='createdAt', description="Creation time")
    updated_at: datetime = Field(..., alias='updatedAt', description="Update time")

    class Config:
        allow_population_by_field_name = True


class KnowledgeRelation(BaseModel):
    """Exact equivalent of JavaScript KnowledgeRelation interface."""
    id: str = Field(..., description="Relation identifier")
    source_node_id: str = Field(..., alias='sourceNodeId', description="Source node ID")
    target_node_id: str = Field(..., alias='targetNodeId', description="Target node ID")
    relation_type: str = Field(..., alias='relationType', description="Relation type")
    strength: float = Field(..., ge=0.0, le=1.0, description="Relation strength")
    metadata: Dict[str, Any] = Field(..., description="Relation metadata")
    created_at: datetime = Field(..., alias='createdAt', description="Creation time")

    class Config:
        allow_population_by_field_name = True


class KnowledgeGraphMetadata(BaseModel):
    """Exact equivalent of JavaScript KnowledgeGraph.metadata."""
    total_nodes: int = Field(..., alias='totalNodes', description="Total number of nodes")
    total_relations: int = Field(..., alias='totalRelations', description="Total number of relations")
    last_updated: datetime = Field(..., alias='lastUpdated', description="Last update time")

    class Config:
        allow_population_by_field_name = True


class KnowledgeGraph(BaseModel):
    """Exact equivalent of JavaScript KnowledgeGraph interface."""
    nodes: List[KnowledgeNode] = Field(..., description="Graph nodes")
    relations: List[KnowledgeRelation] = Field(..., description="Graph relations")
    metadata: KnowledgeGraphMetadata = Field(..., description="Graph metadata")


class ContextItem(BaseModel):
    """Exact equivalent of JavaScript ContextItem interface."""
    context_id: str = Field(..., alias='contextId', description="Context identifier")
    source: str = Field(..., description="Context source")
    content: str = Field(..., description="Context content")
    relevance_score: float = Field(..., alias='relevanceScore', description="Relevance score")
    retrieval_time: float = Field(..., alias='retrievalTime', description="Retrieval time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Context metadata")

    class Config:
        allow_population_by_field_name = True


class TokenUsageFramework(BaseModel):
    """Exact equivalent of JavaScript TokenUsage.frameworkTokens."""
    input_tokens: int = Field(..., alias='inputTokens', description="Input tokens")
    output_tokens: int = Field(..., alias='outputTokens', description="Output tokens")
    reasoning_tokens: Optional[int] = Field(None, alias='reasoningTokens', description="Reasoning tokens (for models like o1)")

    class Config:
        allow_population_by_field_name = True


class TokenUsage(BaseModel):
    """Exact equivalent of JavaScript TokenUsage interface."""
    prompt_tokens: int = Field(..., alias='promptTokens', description="Prompt tokens")
    completion_tokens: int = Field(..., alias='completionTokens', description="Completion tokens")
    total_tokens: int = Field(..., alias='totalTokens', description="Total tokens")
    embedding_tokens: Optional[int] = Field(None, alias='embeddingTokens', description="Embedding tokens")
    framework_tokens: Optional[TokenUsageFramework] = Field(None, alias='frameworkTokens', description="Framework-specific tokens")

    class Config:
        allow_population_by_field_name = True


class EnhancedPromptMetadata(BaseModel):
    """Exact equivalent of JavaScript EnhancedPrompt.metadata."""
    framework_type: str = Field(..., alias='frameworkType', description="Framework type")
    enhancement_strategy: str = Field(..., alias='enhancementStrategy', description="Enhancement strategy")
    context_sources: List[str] = Field(..., alias='contextSources', description="Context sources")
    processing_time: float = Field(..., alias='processingTime', description="Processing time")
    token_count: Optional[int] = Field(None, alias='tokenCount', description="Token count")

    class Config:
        allow_population_by_field_name = True


class EnhancedPrompt(BaseModel):
    """Exact equivalent of JavaScript EnhancedPrompt interface."""
    original_prompt: str = Field(..., alias='originalPrompt', description="Original prompt")
    enhanced_prompt: str = Field(..., alias='enhancedPrompt', description="Enhanced prompt")
    injected_context: List[Context] = Field(..., alias='injectedContext', description="Injected context")
    metadata: EnhancedPromptMetadata = Field(..., description="Enhancement metadata")

    class Config:
        allow_population_by_field_name = True


class InteractionPerformance(BaseModel):
    """Exact equivalent of JavaScript Interaction.performance."""
    response_time: float = Field(..., alias='responseTime', description="Response time")
    context_retrieval_time: float = Field(..., alias='contextRetrievalTime', description="Context retrieval time")
    embedding_time: float = Field(..., alias='embeddingTime', description="Embedding time")

    class Config:
        allow_population_by_field_name = True


class Interaction(BaseModel):
    """Exact equivalent of JavaScript Interaction interface."""
    id: str = Field(..., description="Interaction identifier")
    conversation_id: str = Field(..., alias='conversationId', description="Conversation identifier")
    user_message: str = Field(..., alias='userMessage', description="User message")
    assistant_response: str = Field(..., alias='assistantResponse', description="Assistant response")
    context: List[Context] = Field(..., description="Interaction context")
    metadata: Dict[str, Any] = Field(..., description="Interaction metadata")
    timestamp: datetime = Field(..., description="Interaction timestamp")
    framework: str = Field(..., description="Framework used")
    performance: Optional[InteractionPerformance] = Field(None, description="Performance metrics")

    class Config:
        allow_population_by_field_name = True


class Conversation(BaseModel):
    """Exact equivalent of JavaScript Conversation interface."""
    id: str = Field(..., description="Conversation identifier")
    framework: str = Field(..., description="Framework used")
    title: Optional[str] = Field(None, description="Conversation title")
    metadata: Dict[str, Any] = Field(..., description="Conversation metadata")
    created_at: datetime = Field(..., alias='createdAt', description="Creation time")
    updated_at: datetime = Field(..., alias='updatedAt', description="Update time")
    interaction_count: int = Field(..., alias='interactionCount', description="Number of interactions")
    tags: Optional[List[str]] = Field(None, description="Conversation tags")

    class Config:
        allow_population_by_field_name = True


class SearchResult(BaseModel):
    """Exact equivalent of JavaScript SearchResult interface."""
    id: str = Field(..., description="Result identifier")
    content: str = Field(..., description="Result content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Result metadata")
    source: str = Field(..., description="Result source")
    timestamp: datetime = Field(..., description="Result timestamp")
    embedding: Optional[List[float]] = Field(None, description="Result embedding")


class BrainStatsCollections(BaseModel):
    """Exact equivalent of JavaScript BrainStats.collections."""
    embeddings: Any = Field(..., description="Embeddings collection stats")
    interactions: int = Field(..., description="Number of interactions")
    conversations: int = Field(..., description="Number of conversations")
    knowledge_nodes: Optional[int] = Field(None, alias='knowledgeNodes', description="Number of knowledge nodes")

    class Config:
        allow_population_by_field_name = True


class BrainStatsPerformance(BaseModel):
    """Exact equivalent of JavaScript BrainStats.performance."""
    average_response_time: float = Field(..., alias='averageResponseTime', description="Average response time")
    average_context_retrieval_time: float = Field(..., alias='averageContextRetrievalTime', description="Average context retrieval time")
    average_embedding_time: float = Field(..., alias='averageEmbeddingTime', description="Average embedding time")
    total_requests: int = Field(..., alias='totalRequests', description="Total requests")

    class Config:
        allow_population_by_field_name = True


class BrainStats(BaseModel):
    """Exact equivalent of JavaScript BrainStats interface."""
    is_healthy: bool = Field(..., alias='isHealthy', description="System health status")
    collections: BrainStatsCollections = Field(..., description="Collection statistics")
    performance: BrainStatsPerformance = Field(..., description="Performance statistics")
    last_updated: datetime = Field(..., alias='lastUpdated', description="Last update time")
    version: str = Field(..., description="System version")

    class Config:
        allow_population_by_field_name = True


class PerformanceMetrics(BaseModel):
    """Exact equivalent of JavaScript PerformanceMetrics interface."""
    request_id: str = Field(..., alias='requestId', description="Request identifier")
    operation: str = Field(..., description="Operation name")
    start_time: datetime = Field(..., alias='startTime', description="Start time")
    end_time: datetime = Field(..., alias='endTime', description="End time")
    duration: float = Field(..., description="Duration in milliseconds")
    success: bool = Field(..., description="Success status")
    error: Optional[str] = Field(None, description="Error message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Performance metadata")

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# FRAMEWORK CONFIGURATION MODELS - Exact JavaScript Equivalents
# ============================================================================

class FrameworkInfo(BaseModel):
    """Exact equivalent of JavaScript FrameworkInfo interface."""
    name: str = Field(..., description="Framework name")
    version: str = Field(..., description="Framework version")
    is_available: bool = Field(..., alias='isAvailable', description="Framework availability")
    is_compatible: bool = Field(..., alias='isCompatible', description="Framework compatibility")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in framework")
    capabilities: FrameworkCapabilities = Field(..., description="Framework capabilities")

    class Config:
        allow_population_by_field_name = True


class AdapterConfig(BaseModel):
    """Exact equivalent of JavaScript AdapterConfig interface."""
    enable_memory_injection: bool = Field(..., alias='enableMemoryInjection', description="Enable memory injection")
    enable_context_enhancement: bool = Field(..., alias='enableContextEnhancement', description="Enable context enhancement")
    enable_context_injection: Optional[bool] = Field(None, alias='enableContextInjection', description="Enable context injection")
    enable_tool_integration: bool = Field(..., alias='enableToolIntegration', description="Enable tool integration")
    enable_learning: Optional[bool] = Field(None, alias='enableLearning', description="Enable learning")
    enable_safety_checks: Optional[bool] = Field(None, alias='enableSafetyChecks', description="Enable safety checks")
    enable_performance_monitoring: Optional[bool] = Field(None, alias='enablePerformanceMonitoring', description="Enable performance monitoring")
    max_context_items: int = Field(..., alias='maxContextItems', description="Maximum context items")
    enhancement_strategy: str = Field(..., alias='enhancementStrategy', description="Enhancement strategy: semantic, hybrid, conversational")

    class Config:
        allow_population_by_field_name = True


class OpenAIAgentsAdapterConfig(AdapterConfig):
    """Exact equivalent of JavaScript OpenAIAgentsAdapterConfig interface."""
    enable_agent_enhancement: Optional[bool] = Field(None, alias='enableAgentEnhancement', description="Enable agent enhancement")
    enable_memory_persistence: Optional[bool] = Field(None, alias='enableMemoryPersistence', description="Enable memory persistence")

    class Config:
        allow_population_by_field_name = True


class VercelAIAdapterConfig(AdapterConfig):
    """Exact equivalent of JavaScript VercelAIAdapterConfig interface."""
    enable_stream_enhancement: Optional[bool] = Field(None, alias='enableStreamEnhancement', description="Enable stream enhancement")
    enable_chat_memory: Optional[bool] = Field(None, alias='enableChatMemory', description="Enable chat memory")

    class Config:
        allow_population_by_field_name = True


class MastraAdapterConfig(AdapterConfig):
    """Exact equivalent of JavaScript MastraAdapterConfig interface."""
    enable_workflow_integration: Optional[bool] = Field(None, alias='enableWorkflowIntegration', description="Enable workflow integration")
    enable_memory_replacement: Optional[bool] = Field(None, alias='enableMemoryReplacement', description="Enable memory replacement")
    enable_tool_enhancement: Optional[bool] = Field(None, alias='enableToolEnhancement', description="Enable tool enhancement")

    class Config:
        allow_population_by_field_name = True


class LangChainJSAdapterConfig(AdapterConfig):
    """Exact equivalent of JavaScript LangChainJSAdapterConfig interface."""
    enable_vector_store_replacement: Optional[bool] = Field(None, alias='enableVectorStoreReplacement', description="Enable vector store replacement")
    enable_memory_replacement: Optional[bool] = Field(None, alias='enableMemoryReplacement', description="Enable memory replacement")
    enable_chain_enhancement: Optional[bool] = Field(None, alias='enableChainEnhancement', description="Enable chain enhancement")

    class Config:
        allow_population_by_field_name = True


class CrewAIAdapterConfig(AdapterConfig):
    """CrewAI-specific adapter configuration."""
    enable_crew_enhancement: Optional[bool] = Field(None, alias='enableCrewEnhancement', description="Enable crew enhancement")
    enable_agent_memory: Optional[bool] = Field(None, alias='enableAgentMemory', description="Enable agent memory")
    enable_task_enhancement: Optional[bool] = Field(None, alias='enableTaskEnhancement', description="Enable task enhancement")
    max_agents: Optional[int] = Field(None, alias='maxAgents', description="Maximum number of agents")

    class Config:
        allow_population_by_field_name = True


class LangChainAdapterConfig(AdapterConfig):
    """LangChain-specific adapter configuration."""
    enable_chain_memory: Optional[bool] = Field(None, alias='enableChainMemory', description="Enable chain memory")
    enable_vector_store: Optional[bool] = Field(None, alias='enableVectorStore', description="Enable vector store")
    enable_retrieval_qa: Optional[bool] = Field(None, alias='enableRetrievalQA', description="Enable retrieval QA")
    chain_type: Optional[str] = Field(None, alias='chainType', description="Chain type")

    class Config:
        allow_population_by_field_name = True


class LangGraphAdapterConfig(AdapterConfig):
    """LangGraph-specific adapter configuration."""
    enable_graph_memory: Optional[bool] = Field(None, alias='enableGraphMemory', description="Enable graph memory")
    enable_state_persistence: Optional[bool] = Field(None, alias='enableStatePersistence', description="Enable state persistence")
    enable_workflow_enhancement: Optional[bool] = Field(None, alias='enableWorkflowEnhancement', description="Enable workflow enhancement")
    max_iterations: Optional[int] = Field(None, alias='maxIterations', description="Maximum iterations")

    class Config:
        allow_population_by_field_name = True


class PydanticAIAdapterConfig(AdapterConfig):
    """Pydantic AI-specific adapter configuration."""
    enable_model_validation: Optional[bool] = Field(None, alias='enableModelValidation', description="Enable model validation")
    enable_structured_output: Optional[bool] = Field(None, alias='enableStructuredOutput', description="Enable structured output")
    enable_type_safety: Optional[bool] = Field(None, alias='enableTypeSafety', description="Enable type safety")
    strict_validation: Optional[bool] = Field(None, alias='strictValidation', description="Enable strict validation")

    class Config:
        allow_population_by_field_name = True


class AgnoAdapterConfig(AdapterConfig):
    """Agno-specific adapter configuration."""
    enable_reasoning_enhancement: Optional[bool] = Field(None, alias='enableReasoningEnhancement', description="Enable reasoning enhancement")
    enable_multi_agent: Optional[bool] = Field(None, alias='enableMultiAgent', description="Enable multi-agent")
    enable_workflow_integration: Optional[bool] = Field(None, alias='enableWorkflowIntegration', description="Enable workflow integration")
    reasoning_depth: Optional[int] = Field(None, alias='reasoningDepth', description="Reasoning depth")

    class Config:
        allow_population_by_field_name = True


# ============================================================================
# FRAMEWORK CONFIGURATION VALIDATION AND DEFAULTS
# ============================================================================

class FrameworkConfigDefaults(BaseModel):
    """Default configurations for each framework."""

    @staticmethod
    def get_default_adapter_config() -> AdapterConfig:
        """Get default adapter configuration."""
        return AdapterConfig(
            enableMemoryInjection=True,
            enableContextEnhancement=True,
            enableContextInjection=True,
            enableToolIntegration=True,
            enableLearning=True,
            enableSafetyChecks=True,
            enablePerformanceMonitoring=True,
            maxContextItems=5,
            enhancementStrategy="hybrid"
        )

    @staticmethod
    def get_crewai_config() -> CrewAIAdapterConfig:
        """Get default CrewAI adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return CrewAIAdapterConfig(
            **base_config.dict(),
            enableCrewEnhancement=True,
            enableAgentMemory=True,
            enableTaskEnhancement=True,
            maxAgents=16
        )

    @staticmethod
    def get_langchain_config() -> LangChainAdapterConfig:
        """Get default LangChain adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return LangChainAdapterConfig(
            **base_config.dict(),
            enableChainMemory=True,
            enableVectorStore=True,
            enableRetrievalQA=True,
            chainType="sequential"
        )

    @staticmethod
    def get_langgraph_config() -> LangGraphAdapterConfig:
        """Get default LangGraph adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return LangGraphAdapterConfig(
            **base_config.dict(),
            enableGraphMemory=True,
            enableStatePersistence=True,
            enableWorkflowEnhancement=True,
            maxIterations=50
        )

    @staticmethod
    def get_pydantic_ai_config() -> PydanticAIAdapterConfig:
        """Get default Pydantic AI adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return PydanticAIAdapterConfig(
            **base_config.dict(),
            enableModelValidation=True,
            enableStructuredOutput=True,
            enableTypeSafety=True,
            strictValidation=True
        )

    @staticmethod
    def get_agno_config() -> AgnoAdapterConfig:
        """Get default Agno adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return AgnoAdapterConfig(
            **base_config.dict(),
            enableReasoningEnhancement=True,
            enableMultiAgent=True,
            enableWorkflowIntegration=True,
            reasoningDepth=3
        )

    @staticmethod
    def get_openai_agents_config() -> OpenAIAgentsAdapterConfig:
        """Get default OpenAI Agents adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return OpenAIAgentsAdapterConfig(
            **base_config.dict(),
            enableAgentEnhancement=True,
            enableMemoryPersistence=True
        )

    @staticmethod
    def get_vercel_ai_config() -> VercelAIAdapterConfig:
        """Get default Vercel AI adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return VercelAIAdapterConfig(
            **base_config.dict(),
            enableStreamEnhancement=True,
            enableChatMemory=True
        )

    @staticmethod
    def get_mastra_config() -> MastraAdapterConfig:
        """Get default Mastra adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return MastraAdapterConfig(
            **base_config.dict(),
            enableWorkflowIntegration=True,
            enableMemoryReplacement=True,
            enableToolEnhancement=True
        )

    @staticmethod
    def get_langchain_js_config() -> LangChainJSAdapterConfig:
        """Get default LangChain JS adapter configuration."""
        base_config = FrameworkConfigDefaults.get_default_adapter_config()
        return LangChainJSAdapterConfig(
            **base_config.dict(),
            enableVectorStoreReplacement=True,
            enableMemoryReplacement=True,
            enableChainEnhancement=True
        )


class FrameworkConfigValidator:
    """Validator for framework configurations."""

    @staticmethod
    def validate_enhancement_strategy(strategy: str) -> bool:
        """Validate enhancement strategy."""
        valid_strategies = ["semantic", "hybrid", "conversational"]
        return strategy in valid_strategies

    @staticmethod
    def validate_max_context_items(max_items: int) -> bool:
        """Validate max context items."""
        return 1 <= max_items <= 100

    @staticmethod
    def validate_adapter_config(config: AdapterConfig) -> List[str]:
        """Validate adapter configuration and return list of errors."""
        errors = []

        if not FrameworkConfigValidator.validate_enhancement_strategy(config.enhancement_strategy):
            errors.append(f"Invalid enhancement strategy: {config.enhancement_strategy}")

        if not FrameworkConfigValidator.validate_max_context_items(config.max_context_items):
            errors.append(f"Invalid max context items: {config.max_context_items}")

        return errors

    @staticmethod
    def validate_framework_specific_config(config: Union[
        CrewAIAdapterConfig,
        LangChainAdapterConfig,
        LangGraphAdapterConfig,
        PydanticAIAdapterConfig,
        AgnoAdapterConfig,
        OpenAIAgentsAdapterConfig,
        VercelAIAdapterConfig,
        MastraAdapterConfig,
        LangChainJSAdapterConfig
    ]) -> List[str]:
        """Validate framework-specific configuration."""
        errors = FrameworkConfigValidator.validate_adapter_config(config)

        # Framework-specific validations
        if isinstance(config, CrewAIAdapterConfig) and config.max_agents:
            if not (1 <= config.max_agents <= 50):
                errors.append(f"Invalid max agents for CrewAI: {config.max_agents}")

        if isinstance(config, LangGraphAdapterConfig) and config.max_iterations:
            if not (1 <= config.max_iterations <= 1000):
                errors.append(f"Invalid max iterations for LangGraph: {config.max_iterations}")

        if isinstance(config, AgnoAdapterConfig) and config.reasoning_depth:
            if not (1 <= config.reasoning_depth <= 10):
                errors.append(f"Invalid reasoning depth for Agno: {config.reasoning_depth}")

        return errors
