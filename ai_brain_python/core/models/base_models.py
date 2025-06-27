"""
Base Pydantic Models for AI Brain Python

Foundational models that provide:
- Type safety with runtime validation
- JSON serialization/deserialization
- Field validation and constraints
- Common patterns for all cognitive systems
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, model_validator


class BaseAIBrainModel(BaseModel):
    """Base model for all AI Brain data structures."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class ProcessingStatus(str, Enum):
    """Status of cognitive processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConfidenceLevel(str, Enum):
    """Confidence levels for cognitive assessments."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ProcessingMetadata(BaseAIBrainModel):
    """Metadata for cognitive processing operations."""
    
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")
    model_version: str = Field(default="1.0.0", description="Model version used")
    framework_used: Optional[str] = Field(default=None, description="AI framework used for processing")
    system_id: str = Field(description="Cognitive system identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    
    @validator('processing_time_ms')
    def validate_processing_time(cls, v):
        """Validate processing time is non-negative."""
        if v is not None and v < 0:
            raise ValueError("Processing time must be non-negative")
        return v


class ValidationResult(BaseAIBrainModel):
    """Result of validation operations."""
    
    is_valid: bool = Field(description="Whether the validation passed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in validation result")
    violations: List[str] = Field(default_factory=list, description="List of validation violations")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional validation details")


class CognitiveContext(BaseAIBrainModel):
    """Context information for cognitive processing."""
    
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    cultural_context: Dict[str, Any] = Field(default_factory=dict, description="Cultural context information")
    temporal_context: Dict[str, Any] = Field(default_factory=dict, description="Temporal context information")
    environmental_context: Dict[str, Any] = Field(default_factory=dict, description="Environmental context")
    
    # Context state
    attention_focus: Optional[str] = Field(default=None, description="Current attention focus")
    emotional_state: Optional[str] = Field(default=None, description="Current emotional state")
    cognitive_load: float = Field(default=0.5, ge=0.0, le=1.0, description="Current cognitive load")
    
    @validator('conversation_history')
    def validate_conversation_history(cls, v):
        """Validate conversation history format."""
        for entry in v:
            if not isinstance(entry, dict) or 'timestamp' not in entry:
                raise ValueError("Conversation history entries must be dicts with timestamp")
        return v


class CognitiveInputData(BaseAIBrainModel):
    """Input data for cognitive processing."""
    
    # Primary input
    text: Optional[str] = Field(default=None, description="Text input for processing")
    audio_data: Optional[bytes] = Field(default=None, description="Audio data for processing")
    image_data: Optional[bytes] = Field(default=None, description="Image data for processing")
    video_data: Optional[bytes] = Field(default=None, description="Video data for processing")
    
    # Input metadata
    input_type: str = Field(default="text", description="Type of input (text, audio, image, video, multimodal)")
    language: str = Field(default="en", description="Language of the input")
    encoding: str = Field(default="utf-8", description="Text encoding")
    
    # Processing context
    context: CognitiveContext = Field(default_factory=CognitiveContext, description="Processing context")
    requested_systems: List[str] = Field(default_factory=list, description="Specific cognitive systems to engage")
    processing_priority: int = Field(default=5, ge=1, le=10, description="Processing priority (1=low, 10=high)")
    
    # Quality and safety
    content_filter_level: str = Field(default="standard", description="Content filtering level")
    safety_check_required: bool = Field(default=True, description="Whether safety checking is required")
    
    @validator('text')
    def validate_text_length(cls, v):
        """Validate text input length."""
        if v is not None and len(v) > 100000:  # 100k character limit
            raise ValueError("Text input too long (max 100,000 characters)")
        return v
    
    @model_validator(mode='after')
    def validate_input_data(self):
        """Validate that at least one input type is provided."""
        input_fields = [self.text, self.audio_data, self.image_data, self.video_data]
        if not any(input_fields):
            raise ValueError("At least one input type must be provided")
        return self


class CognitiveResponse(BaseAIBrainModel):
    """Response from cognitive processing."""
    
    # Processing status
    status: ProcessingStatus = Field(description="Processing status")
    success: bool = Field(description="Whether processing was successful")
    
    # Core cognitive results
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in response")
    emotional_state: Optional[Dict[str, Any]] = Field(default=None, description="Detected emotional state")
    attention_allocation: Optional[Dict[str, float]] = Field(default=None, description="Attention allocation")
    goal_hierarchy: Optional[Dict[str, Any]] = Field(default=None, description="Goal hierarchy updates")
    
    # System-specific results
    cognitive_results: Dict[str, Any] = Field(default_factory=dict, description="Results from each cognitive system")
    
    # Response content
    response_text: Optional[str] = Field(default=None, description="Generated response text")
    response_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured response data")
    
    # Processing information
    processing_metadata: ProcessingMetadata = Field(description="Processing metadata")
    validation_result: Optional[ValidationResult] = Field(default=None, description="Validation results")
    
    # Safety and monitoring
    safety_assessment: Optional[Dict[str, Any]] = Field(default=None, description="Safety assessment results")
    monitoring_metrics: Optional[Dict[str, float]] = Field(default=None, description="Performance metrics")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    @validator('confidence')
    def validate_confidence_range(cls, v):
        """Validate confidence is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    @validator('attention_allocation')
    def validate_attention_allocation(cls, v):
        """Validate attention allocation sums to 1.0."""
        if v is not None:
            total = sum(v.values())
            if not 0.95 <= total <= 1.05:  # Allow small floating point errors
                raise ValueError("Attention allocation must sum to approximately 1.0")
        return v


class CognitiveRequest(BaseAIBrainModel):
    """Request for cognitive processing."""
    
    input_data: CognitiveInputData = Field(description="Input data for processing")
    requested_systems: List[str] = Field(default_factory=list, description="Specific cognitive systems to engage")
    processing_options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    
    # Request configuration
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    priority: int = Field(default=5, ge=1, le=10, description="Request priority")
    
    # Framework preferences
    preferred_framework: Optional[str] = Field(default=None, description="Preferred AI framework")
    framework_config: Dict[str, Any] = Field(default_factory=dict, description="Framework-specific configuration")
    
    # Response requirements
    require_explanation: bool = Field(default=False, description="Whether to include explanations")
    require_confidence_scores: bool = Field(default=True, description="Whether to include confidence scores")
    response_format: str = Field(default="json", description="Desired response format")
    
    @validator('requested_systems')
    def validate_requested_systems(cls, v):
        """Validate requested cognitive systems."""
        valid_systems = {
            "emotional_intelligence", "goal_hierarchy", "confidence_tracking",
            "attention_management", "cultural_knowledge", "skill_capability",
            "communication_protocol", "temporal_planning", "semantic_memory",
            "safety_guardrails", "self_improvement", "monitoring",
            "tool_interface", "workflow_orchestration", "multimodal_processing",
            "human_feedback"
        }
        
        for system in v:
            if system not in valid_systems:
                raise ValueError(f"Unknown cognitive system: {system}")
        
        return v


# Type aliases for common use cases
CognitiveData = Union[CognitiveInputData, CognitiveResponse]
ProcessingResult = Dict[str, Any]
SystemState = Dict[str, Any]
