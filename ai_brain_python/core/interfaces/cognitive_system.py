"""
Cognitive System Interface

Abstract base classes for implementing cognitive systems in the AI Brain.
All 16 cognitive systems must implement these interfaces to ensure
consistency and interoperability.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ai_brain_python.core.models.base_models import (
    CognitiveInputData,
    CognitiveResponse,
    ValidationResult,
)
from ai_brain_python.core.models.cognitive_states import CognitiveState


class SystemCapability(str, Enum):
    """Capabilities that cognitive systems can provide."""
    EMOTION_DETECTION = "emotion_detection"
    EMOTION_GENERATION = "emotion_generation"
    GOAL_PLANNING = "goal_planning"
    GOAL_TRACKING = "goal_tracking"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    ATTENTION_ALLOCATION = "attention_allocation"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    SKILL_ASSESSMENT = "skill_assessment"
    COMMUNICATION_OPTIMIZATION = "communication_optimization"
    TEMPORAL_PLANNING = "temporal_planning"
    MEMORY_STORAGE = "memory_storage"
    MEMORY_RETRIEVAL = "memory_retrieval"
    SAFETY_VALIDATION = "safety_validation"
    PERFORMANCE_MONITORING = "performance_monitoring"
    TOOL_VALIDATION = "tool_validation"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    HUMAN_FEEDBACK_INTEGRATION = "human_feedback_integration"


class CognitiveProcessor(ABC):
    """Base class for cognitive processing components."""
    
    @abstractmethod
    async def process(
        self, 
        input_data: CognitiveInputData,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data to process
            context: Additional processing context
            
        Returns:
            Processing results as a dictionary
        """
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """
        Validate input data for processing.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Validation result
        """
        pass


class CognitiveSystemInterface(ABC):
    """
    Abstract base class for all cognitive systems.
    
    Each of the 16 cognitive systems must implement this interface
    to ensure consistent behavior and integration with the Universal AI Brain.
    """
    
    def __init__(self, system_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive system.
        
        Args:
            system_id: Unique identifier for the system
            config: System-specific configuration
        """
        self.system_id = system_id
        self.config = config or {}
        self._is_initialized = False
        self._capabilities: Set[SystemCapability] = set()
        self._processors: Dict[str, CognitiveProcessor] = {}
    
    @property
    @abstractmethod
    def system_name(self) -> str:
        """Human-readable name of the cognitive system."""
        pass
    
    @property
    @abstractmethod
    def system_description(self) -> str:
        """Description of what this cognitive system does."""
        pass
    
    @property
    @abstractmethod
    def required_capabilities(self) -> Set[SystemCapability]:
        """Set of capabilities this system requires."""
        pass
    
    @property
    @abstractmethod
    def provided_capabilities(self) -> Set[SystemCapability]:
        """Set of capabilities this system provides."""
        pass
    
    @property
    def capabilities(self) -> Set[SystemCapability]:
        """Get all capabilities (required and provided)."""
        return self.required_capabilities.union(self.provided_capabilities)
    
    @property
    def is_initialized(self) -> bool:
        """Check if the system is initialized."""
        return self._is_initialized
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the cognitive system.
        
        This method should:
        - Set up any required resources
        - Initialize processors
        - Load any persistent state
        - Validate configuration
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the cognitive system.
        
        This method should:
        - Clean up resources
        - Save any persistent state
        - Stop any background tasks
        """
        pass
    
    @abstractmethod
    async def process(
        self, 
        input_data: CognitiveInputData,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through this cognitive system.
        
        Args:
            input_data: Input data to process
            context: Additional processing context
            
        Returns:
            Processing results specific to this system
        """
        pass
    
    @abstractmethod
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """
        Get the current state of the cognitive system.
        
        Args:
            user_id: User ID for user-specific state
            
        Returns:
            Current cognitive state
        """
        pass
    
    @abstractmethod
    async def update_state(
        self, 
        state: CognitiveState, 
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update the state of the cognitive system.
        
        Args:
            state: New cognitive state
            user_id: User ID for user-specific state
            
        Returns:
            True if update was successful
        """
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """
        Validate input data for this cognitive system.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Validation result
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the cognitive system.
        
        Returns:
            Health status information
        """
        return {
            "system_id": self.system_id,
            "system_name": self.system_name,
            "initialized": self._is_initialized,
            "capabilities": [cap.value for cap in self.capabilities],
            "processors": list(self._processors.keys()),
            "status": "healthy" if self._is_initialized else "not_initialized"
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the cognitive system.
        
        Returns:
            Performance metrics
        """
        return {
            "system_id": self.system_id,
            "processors": len(self._processors),
            "capabilities": len(self.capabilities),
            "initialized": self._is_initialized
        }
    
    def add_processor(self, name: str, processor: CognitiveProcessor) -> None:
        """
        Add a cognitive processor to this system.
        
        Args:
            name: Name of the processor
            processor: Processor instance
        """
        self._processors[name] = processor
    
    def get_processor(self, name: str) -> Optional[CognitiveProcessor]:
        """
        Get a cognitive processor by name.
        
        Args:
            name: Name of the processor
            
        Returns:
            Processor instance or None if not found
        """
        return self._processors.get(name)
    
    def has_capability(self, capability: SystemCapability) -> bool:
        """
        Check if this system has a specific capability.
        
        Args:
            capability: Capability to check
            
        Returns:
            True if system has the capability
        """
        return capability in self.capabilities
    
    async def can_process(self, input_data: CognitiveInputData) -> bool:
        """
        Check if this system can process the given input.
        
        Args:
            input_data: Input data to check
            
        Returns:
            True if system can process the input
        """
        try:
            validation_result = await self.validate_input(input_data)
            return validation_result.is_valid
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of the cognitive system."""
        return f"{self.system_name} ({self.system_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"CognitiveSystem(id='{self.system_id}', "
            f"name='{self.system_name}', "
            f"initialized={self._is_initialized}, "
            f"capabilities={len(self.capabilities)})"
        )
