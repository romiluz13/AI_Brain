"""
Universal AI Brain - Main Orchestrator

The core orchestrator that coordinates all 24 cognitive systems (matching JavaScript version exactly):

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

Features:
- Async processing with high performance
- Framework-agnostic design
- MongoDB Atlas native integration
- Real-time state management
- Comprehensive error handling
- Performance monitoring
- Direct access patterns: brain.system.method()
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union, Literal
from dataclasses import dataclass, field

from ai_brain_python.core.models.base_models import (
    CognitiveInputData,
    CognitiveResponse,
    CognitiveRequest,
    ProcessingStatus,
    ProcessingMetadata,
    ValidationResult,
)
from ai_brain_python.core.models.cognitive_states import CognitiveSystemType
from ai_brain_python.storage.storage_manager import StorageManager, StorageConfig

# Import all 24 cognitive systems
# Memory Systems (4)
from ai_brain_python.core.cognitive_systems.working_memory_manager import WorkingMemoryManager
from ai_brain_python.core.cognitive_systems.episodic_memory import EpisodicMemoryEngine
from ai_brain_python.core.cognitive_systems.semantic_memory import SemanticMemoryEngine
from ai_brain_python.core.cognitive_systems.memory_decay_engine import MemoryDecayEngine

# Reasoning Systems (6)
from ai_brain_python.core.cognitive_systems.analogical_mapping import AnalogicalMappingSystem
from ai_brain_python.core.cognitive_systems.causal_reasoning import CausalReasoningEngine
from ai_brain_python.core.cognitive_systems.attention_management import AttentionManagementSystem
from ai_brain_python.core.cognitive_systems.confidence_tracking import ConfidenceTrackingEngine
from ai_brain_python.core.cognitive_systems.context_injection import ContextInjectionEngine
from ai_brain_python.core.cognitive_systems.vector_search import VectorSearchEngine

# Emotional Systems (3)
from ai_brain_python.core.cognitive_systems.emotional_intelligence import EmotionalIntelligenceEngine
from ai_brain_python.core.cognitive_systems.social_intelligence import SocialIntelligenceEngine
from ai_brain_python.core.cognitive_systems.cultural_knowledge import CulturalKnowledgeEngine

# Social Systems (3)
from ai_brain_python.core.cognitive_systems.goal_hierarchy import GoalHierarchyManager
from ai_brain_python.core.cognitive_systems.human_feedback import HumanFeedbackIntegrationEngine
from ai_brain_python.core.cognitive_systems.safety_guardrails import SafetyGuardrailsEngine

# Temporal Systems (2)
from ai_brain_python.core.cognitive_systems.temporal_planning import TemporalPlanningEngine
from ai_brain_python.core.cognitive_systems.skill_capability import SkillCapabilityManager

# Meta Systems (6)
from ai_brain_python.core.cognitive_systems.self_improvement import SelfImprovementEngine
from ai_brain_python.core.cognitive_systems.multimodal_processing import MultiModalProcessingEngine
from ai_brain_python.core.cognitive_systems.tool_interface import AdvancedToolInterface
from ai_brain_python.core.cognitive_systems.workflow_orchestration import WorkflowOrchestrationEngine
from ai_brain_python.features.hybrid_search import HybridSearchEngine
from ai_brain_python.core.cognitive_systems.monitoring import MonitoringEngine

logger = logging.getLogger(__name__)


# 🎯 CONFIGURATION STRUCTURES (matching JavaScript exactly)

@dataclass
class MongoDBCollections:
    """MongoDB collection configuration."""
    tracing: str = "agent_traces"
    memory: str = "agent_memory"
    context: str = "agent_context"
    metrics: str = "agent_metrics"
    audit: str = "agent_safety_logs"


@dataclass
class MongoDBConfig:
    """MongoDB configuration."""
    connection_string: str
    database_name: str = "universal_ai_brain"
    collections: MongoDBCollections = field(default_factory=MongoDBCollections)


@dataclass
class IntelligenceConfig:
    """Intelligence and AI configuration."""
    embedding_model: str = "voyage-large-2-instruct"
    vector_dimensions: int = 1024
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    # Hybrid Search Configuration
    enable_hybrid_search: bool = True
    hybrid_search_vector_weight: float = 0.7
    hybrid_search_text_weight: float = 0.3
    hybrid_search_fallback_to_vector: bool = True


@dataclass
class SafetyConfig:
    """Safety and compliance configuration."""
    enable_content_filtering: bool = True
    enable_pii_detection: bool = True
    enable_hallucination_detection: bool = True
    enable_compliance_logging: bool = True
    safety_level: Literal['strict', 'moderate', 'permissive'] = 'moderate'


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_realtime_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_cost_tracking: bool = True
    enable_error_tracking: bool = True
    metrics_retention_days: int = 30
    alerting_enabled: bool = True
    dashboard_refresh_interval: int = 5000  # milliseconds


@dataclass
class SelfImprovementConfig:
    """Self-improvement and learning configuration."""
    enable_automatic_optimization: bool = True
    learning_rate: float = 0.01
    optimization_interval: int = 3600  # seconds
    feedback_loop_enabled: bool = True


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str
    base_url: str = "https://api.openai.com/v1"


@dataclass
class VoyageConfig:
    """Voyage AI API configuration."""
    api_key: str
    base_url: str = "https://api.voyageai.com/v1"


@dataclass
class APIConfig:
    """API configuration for external services."""
    openai: Optional[OpenAIConfig] = None
    voyage: Optional[VoyageConfig] = None


@dataclass
class UniversalAIBrainConfig:
    """
    Universal AI Brain Configuration - Exact match to JavaScript UniversalAIBrainConfig

    Supports both simple and advanced configuration patterns.
    """
    # Advanced configuration
    mongodb: Optional[MongoDBConfig] = None
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    self_improvement: SelfImprovementConfig = field(default_factory=SelfImprovementConfig)
    apis: APIConfig = field(default_factory=APIConfig)

    # Simple configuration (for easy setup)
    mongo_uri: Optional[str] = None
    database_name: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[Literal['voyage', 'openai']] = None
    mode: Optional[Literal['demo', 'basic', 'production']] = None

    # Python-specific settings
    enable_all_systems: bool = True
    enabled_systems: Optional[Set[str]] = None
    max_concurrent_processing: int = 10
    default_timeout: int = 30

    def __post_init__(self):
        """Initialize configuration after creation."""
        # Handle simple configuration mode
        if self.mongo_uri and not self.mongodb:
            collections = MongoDBCollections()
            self.mongodb = MongoDBConfig(
                connection_string=self.mongo_uri,
                database_name=self.database_name or "universal_ai_brain",
                collections=collections
            )

        # Handle simple API configuration
        if self.api_key and self.provider:
            if self.provider == 'openai':
                self.apis.openai = OpenAIConfig(api_key=self.api_key)
            elif self.provider == 'voyage':
                self.apis.voyage = VoyageConfig(api_key=self.api_key)

        # Set enabled systems
        if self.enabled_systems is None:
            self.enabled_systems = set()


# 🎯 SIMPLE CONFIG FOR EASY SETUP (matching JavaScript SimpleAIBrainConfig)
@dataclass
class SimpleAIBrainConfig:
    """Simple configuration for easy setup."""
    mongo_uri: Optional[str] = None
    database_name: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[Literal['voyage', 'openai']] = None
    mode: Optional[Literal['demo', 'basic', 'production']] = None

    def to_universal_config(self) -> UniversalAIBrainConfig:
        """Convert to UniversalAIBrainConfig."""
        return UniversalAIBrainConfig(
            mongo_uri=self.mongo_uri,
            database_name=self.database_name,
            api_key=self.api_key,
            provider=self.provider,
            mode=self.mode
        )


class UniversalAIBrain:
    """
    Universal AI Brain - Main orchestrator for all cognitive systems.
    
    This class coordinates the 16 cognitive systems and provides a unified
    interface for processing cognitive requests across different AI frameworks.
    """
    
    def __init__(self, config: UniversalAIBrainConfig):
        """Initialize the Universal AI Brain."""
        self.config = config

        # Create storage config from MongoDB config
        if not config.mongodb:
            raise ValueError("MongoDB configuration is required")

        storage_config = StorageConfig(
            connection_string=config.mongodb.connection_string,
            database_name=config.mongodb.database_name,
            collections={
                "tracing": config.mongodb.collections.tracing,
                "memory": config.mongodb.collections.memory,
                "context": config.mongodb.collections.context,
                "metrics": config.mongodb.collections.metrics,
                "audit": config.mongodb.collections.audit,
            }
        )
        self.storage_manager = StorageManager(storage_config)
        
        # System state
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_processing)
        
        # Cognitive systems registry - actual instances
        self._cognitive_systems: Dict[str, Any] = {}
        self._system_states: Dict[str, Dict[str, Any]] = {}

        # System class mapping for instantiation
        self._system_classes = {
            # Memory Systems (4)
            "working_memory": WorkingMemoryManager,
            "episodic_memory": EpisodicMemoryEngine,
            "semantic_memory": SemanticMemoryEngine,
            "memory_decay": MemoryDecayEngine,

            # Reasoning Systems (6)
            "analogical_mapping": AnalogicalMappingSystem,
            "causal_reasoning": CausalReasoningEngine,
            "attention_management": AttentionManagementSystem,
            "confidence_tracking": ConfidenceTrackingEngine,
            "context_injection": ContextInjectionEngine,
            "vector_search": VectorSearchEngine,

            # Emotional Systems (3)
            "emotional_intelligence": EmotionalIntelligenceEngine,
            "social_intelligence": SocialIntelligenceEngine,
            "cultural_knowledge": CulturalKnowledgeEngine,

            # Social Systems (3)
            "goal_hierarchy": GoalHierarchyManager,
            "human_feedback": HumanFeedbackIntegrationEngine,
            "safety_guardrails": SafetyGuardrailsEngine,

            # Temporal Systems (2)
            "temporal_planning": TemporalPlanningEngine,
            "skill_capability": SkillCapabilityManager,

            # Meta Systems (6)
            "self_improvement": SelfImprovementEngine,
            "multimodal_processing": MultiModalProcessingEngine,
            "tool_interface": AdvancedToolInterface,
            "workflow_orchestration": WorkflowOrchestrationEngine,
            "hybrid_search": HybridSearchEngine,
            "realtime_monitoring": MonitoringEngine,
        }
        
        # Performance tracking
        self._processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "last_request_time": None,
        }
        
        # Define all 24 cognitive systems (matching JavaScript version exactly)
        self._cognitive_system_types = {
            # Memory Systems (4)
            "working_memory": CognitiveSystemType.WORKING_MEMORY,
            "episodic_memory": CognitiveSystemType.EPISODIC_MEMORY,
            "semantic_memory": CognitiveSystemType.SEMANTIC_MEMORY,
            "memory_decay": CognitiveSystemType.MEMORY_DECAY,

            # Reasoning Systems (6)
            "analogical_mapping": CognitiveSystemType.ANALOGICAL_MAPPING,
            "causal_reasoning": CognitiveSystemType.CAUSAL_REASONING,
            "attention_management": CognitiveSystemType.ATTENTION_MANAGEMENT,
            "confidence_tracking": CognitiveSystemType.CONFIDENCE_TRACKING,
            "context_injection": CognitiveSystemType.CONTEXT_INJECTION,
            "vector_search": CognitiveSystemType.VECTOR_SEARCH,

            # Emotional Systems (3)
            "emotional_intelligence": CognitiveSystemType.EMOTIONAL_INTELLIGENCE,
            "social_intelligence": CognitiveSystemType.SOCIAL_INTELLIGENCE,
            "cultural_knowledge": CognitiveSystemType.CULTURAL_KNOWLEDGE,

            # Social Systems (3)
            "goal_hierarchy": CognitiveSystemType.GOAL_HIERARCHY,
            "human_feedback": CognitiveSystemType.HUMAN_FEEDBACK,
            "safety_guardrails": CognitiveSystemType.SAFETY_GUARDRAILS,

            # Temporal Systems (2)
            "temporal_planning": CognitiveSystemType.TEMPORAL_PLANNING,
            "skill_capability": CognitiveSystemType.SKILL_CAPABILITY,

            # Meta Systems (6)
            "self_improvement": CognitiveSystemType.SELF_IMPROVEMENT,
            "multimodal_processing": CognitiveSystemType.MULTIMODAL_PROCESSING,
            "tool_interface": CognitiveSystemType.TOOL_INTERFACE,
            "workflow_orchestration": CognitiveSystemType.WORKFLOW_ORCHESTRATION,
            "hybrid_search": CognitiveSystemType.HYBRID_SEARCH,
            "realtime_monitoring": CognitiveSystemType.REALTIME_MONITORING,
        }
        
        # Determine enabled systems
        if config.enable_all_systems:
            self._enabled_systems = set(self._cognitive_system_types.keys())
        else:
            self._enabled_systems = (config.enabled_systems or set()).intersection(
                set(self._cognitive_system_types.keys())
            )
        
        logger.info(f"Universal AI Brain initialized with {len(self._enabled_systems)} cognitive systems")
    
    async def initialize(self) -> None:
        """Initialize the AI Brain and all cognitive systems."""
        async with self._initialization_lock:
            if self._is_initialized:
                return
            
            try:
                logger.info("Initializing Universal AI Brain...")
                
                # Initialize storage manager
                await self.storage_manager.initialize()
                
                # Initialize cognitive systems
                await self._initialize_cognitive_systems()
                
                # Load system states from storage
                await self._load_system_states()
                
                # Initialize monitoring if enabled
                if self.config.monitoring.enable_realtime_monitoring:
                    await self._initialize_monitoring()
                
                self._is_initialized = True
                logger.info("Universal AI Brain initialization completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Universal AI Brain: {e}")
                raise
    
    async def shutdown(self) -> None:
        """Shutdown the AI Brain and cleanup resources."""
        if not self._is_initialized:
            return
        
        try:
            logger.info("Shutting down Universal AI Brain...")
            
            # Save system states
            await self._save_system_states()
            
            # Shutdown cognitive systems
            await self._shutdown_cognitive_systems()
            
            # Shutdown storage manager
            await self.storage_manager.shutdown()
            
            self._is_initialized = False
            logger.info("Universal AI Brain shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during AI Brain shutdown: {e}")
    
    async def process_input(
        self, 
        input_data: CognitiveInputData,
        requested_systems: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ) -> CognitiveResponse:
        """
        Process input through the cognitive systems.
        
        Args:
            input_data: Input data to process
            requested_systems: Specific systems to engage (None for all enabled)
            timeout: Processing timeout in seconds
            
        Returns:
            CognitiveResponse with processing results
        """
        if not self._is_initialized:
            raise RuntimeError("AI Brain not initialized. Call initialize() first.")
        
        # Create processing metadata
        start_time = time.time()
        request_id = input_data.id
        timeout = timeout or self.config.default_timeout
        
        processing_metadata = ProcessingMetadata(
            system_id="universal_ai_brain",
            user_id=input_data.context.user_id,
            session_id=input_data.context.session_id,
            request_id=request_id,
        )
        
        # Initialize response
        response = CognitiveResponse(
            status=ProcessingStatus.PROCESSING,
            success=False,
            confidence=0.0,
            processing_metadata=processing_metadata,
        )
        
        async with self._processing_semaphore:
            try:
                # Update stats
                self._processing_stats["total_requests"] += 1
                self._processing_stats["last_request_time"] = datetime.utcnow()
                
                # Determine which systems to engage
                systems_to_process = self._determine_systems_to_process(
                    input_data, requested_systems
                )
                
                # Safety check if enabled
                if self.config.safety.enable_content_filtering:
                    safety_result = await self._perform_safety_check(input_data)
                    if not safety_result.is_valid:
                        response.status = ProcessingStatus.FAILED
                        response.errors = safety_result.violations
                        response.safety_assessment = {
                            "is_safe": False,
                            "violations": safety_result.violations,
                            "warnings": safety_result.warnings
                        }
                        return response
                
                # Process through cognitive systems
                cognitive_results = await self._process_through_systems(
                    input_data, systems_to_process, timeout
                )
                
                # Aggregate results
                response = await self._aggregate_results(
                    input_data, cognitive_results, response, start_time
                )
                
                # Update success stats
                self._processing_stats["successful_requests"] += 1
                response.status = ProcessingStatus.COMPLETED
                response.success = True
                
                logger.debug(f"Successfully processed request {request_id}")
                
            except asyncio.TimeoutError:
                response.status = ProcessingStatus.FAILED
                response.errors.append(f"Processing timeout after {timeout} seconds")
                self._processing_stats["failed_requests"] += 1
                logger.warning(f"Request {request_id} timed out")
                
            except Exception as e:
                response.status = ProcessingStatus.FAILED
                response.errors.append(f"Processing error: {str(e)}")
                self._processing_stats["failed_requests"] += 1
                logger.error(f"Error processing request {request_id}: {e}")
                
            finally:
                # Update processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                response.processing_metadata.processing_time_ms = processing_time
                
                # Update average processing time
                total_requests = self._processing_stats["total_requests"]
                current_avg = self._processing_stats["average_processing_time"]
                self._processing_stats["average_processing_time"] = (
                    (current_avg * (total_requests - 1) + processing_time) / total_requests
                )
                
                # Store processing result if monitoring enabled
                if self.config.monitoring.enable_performance_tracking:
                    await self._store_processing_result(input_data, response)
        
        return response
    
    async def process_request(self, request: CognitiveRequest) -> CognitiveResponse:
        """Process a cognitive request."""
        return await self.process_input(
            input_data=request.input_data,
            requested_systems=request.requested_systems,
            timeout=request.timeout_seconds
        )
    
    async def get_system_state(self, system_name: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the current state of a cognitive system."""
        if system_name not in self._enabled_systems:
            return None
        
        # Try to get from cache first
        cache_key = f"{system_name}:{user_id}" if user_id else system_name
        if cache_key in self._system_states:
            return self._system_states[cache_key]
        
        # Load from storage
        state = await self.storage_manager.get_cognitive_state(system_name, user_id or "global")
        if state:
            self._system_states[cache_key] = state
        
        return state
    
    async def update_system_state(
        self, 
        system_name: str, 
        state: Dict[str, Any], 
        user_id: Optional[str] = None
    ) -> bool:
        """Update the state of a cognitive system."""
        if system_name not in self._enabled_systems:
            return False
        
        try:
            # Store in storage
            await self.storage_manager.store_cognitive_state(
                system_name, user_id or "global", state
            )
            
            # Update cache
            cache_key = f"{system_name}:{user_id}" if user_id else system_name
            self._system_states[cache_key] = state
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating system state for {system_name}: {e}")
            return False
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._processing_stats.copy()
        
        # Add system-specific stats
        stats["enabled_systems"] = list(self._enabled_systems)
        stats["total_systems"] = len(self._cognitive_system_types)
        
        # Add storage stats if available
        try:
            storage_stats = await self.storage_manager.get_statistics()
            stats["storage"] = storage_stats
        except Exception as e:
            logger.warning(f"Could not retrieve storage stats: {e}")
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "initialized": self._is_initialized,
            "components": {}
        }
        
        try:
            # Check storage health
            storage_health = await self.storage_manager.health_check()
            health_status["components"]["storage"] = storage_health
            
            # Check cognitive systems
            systems_health = await self._check_systems_health()
            health_status["components"]["cognitive_systems"] = systems_health
            
            # Check performance
            performance_stats = await self.get_performance_stats()
            health_status["components"]["performance"] = {
                "total_requests": performance_stats["total_requests"],
                "success_rate": (
                    performance_stats["successful_requests"] / 
                    max(performance_stats["total_requests"], 1)
                ),
                "average_processing_time_ms": performance_stats["average_processing_time"]
            }
            
            # Determine overall health
            component_statuses = [
                comp.get("status", "unknown") 
                for comp in health_status["components"].values()
            ]
            
            if any(status == "unhealthy" for status in component_statuses):
                health_status["status"] = "unhealthy"
            elif any(status == "degraded" for status in component_statuses):
                health_status["status"] = "degraded"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status

    # ========================================
    # DIRECT ACCESS PROPERTY GETTERS
    # JavaScript-style direct access: brain.system.method()
    # ========================================

    # Memory Systems (4)
    @property
    def working_memory(self) -> WorkingMemoryManager:
        """Direct access to Working Memory Manager."""
        return self._cognitive_systems.get("working_memory")

    @property
    def episodic_memory(self) -> EpisodicMemoryEngine:
        """Direct access to Episodic Memory Engine."""
        return self._cognitive_systems.get("episodic_memory")

    @property
    def semantic_memory(self) -> SemanticMemoryEngine:
        """Direct access to Semantic Memory Engine."""
        return self._cognitive_systems.get("semantic_memory")

    @property
    def memory_decay(self) -> MemoryDecayEngine:
        """Direct access to Memory Decay Engine."""
        return self._cognitive_systems.get("memory_decay")

    # Reasoning Systems (6)
    @property
    def analogical_mapping(self) -> AnalogicalMappingSystem:
        """Direct access to Analogical Mapping System."""
        return self._cognitive_systems.get("analogical_mapping")

    @property
    def causal_reasoning(self) -> CausalReasoningEngine:
        """Direct access to Causal Reasoning Engine."""
        return self._cognitive_systems.get("causal_reasoning")

    @property
    def attention_management(self) -> AttentionManagementSystem:
        """Direct access to Attention Management System."""
        return self._cognitive_systems.get("attention_management")

    @property
    def confidence_tracking(self) -> ConfidenceTrackingEngine:
        """Direct access to Confidence Tracking Engine."""
        return self._cognitive_systems.get("confidence_tracking")

    @property
    def context_injection(self) -> ContextInjectionEngine:
        """Direct access to Context Injection Engine."""
        return self._cognitive_systems.get("context_injection")

    @property
    def vector_search(self) -> VectorSearchEngine:
        """Direct access to Vector Search Engine."""
        return self._cognitive_systems.get("vector_search")

    # Emotional Systems (3)
    @property
    def emotional_intelligence(self) -> EmotionalIntelligenceEngine:
        """Direct access to Emotional Intelligence Engine."""
        return self._cognitive_systems.get("emotional_intelligence")

    @property
    def social_intelligence(self) -> SocialIntelligenceEngine:
        """Direct access to Social Intelligence Engine."""
        return self._cognitive_systems.get("social_intelligence")

    @property
    def cultural_knowledge(self) -> CulturalKnowledgeEngine:
        """Direct access to Cultural Knowledge Engine."""
        return self._cognitive_systems.get("cultural_knowledge")

    # Social Systems (3)
    @property
    def goal_hierarchy(self) -> GoalHierarchyManager:
        """Direct access to Goal Hierarchy Manager."""
        return self._cognitive_systems.get("goal_hierarchy")

    @property
    def human_feedback(self) -> HumanFeedbackIntegrationEngine:
        """Direct access to Human Feedback Integration Engine."""
        return self._cognitive_systems.get("human_feedback")

    @property
    def safety_guardrails(self) -> SafetyGuardrailsEngine:
        """Direct access to Safety Guardrails Engine."""
        return self._cognitive_systems.get("safety_guardrails")

    # Temporal Systems (2)
    @property
    def temporal_planning(self) -> TemporalPlanningEngine:
        """Direct access to Temporal Planning Engine."""
        return self._cognitive_systems.get("temporal_planning")

    @property
    def skill_capability(self) -> SkillCapabilityManager:
        """Direct access to Skill Capability Manager."""
        return self._cognitive_systems.get("skill_capability")

    # Meta Systems (6)
    @property
    def self_improvement(self) -> SelfImprovementEngine:
        """Direct access to Self Improvement Engine."""
        return self._cognitive_systems.get("self_improvement")

    @property
    def multimodal_processing(self) -> MultiModalProcessingEngine:
        """Direct access to Multi-Modal Processing Engine."""
        return self._cognitive_systems.get("multimodal_processing")

    @property
    def tool_interface(self) -> AdvancedToolInterface:
        """Direct access to Advanced Tool Interface."""
        return self._cognitive_systems.get("tool_interface")

    @property
    def workflow_orchestration(self) -> WorkflowOrchestrationEngine:
        """Direct access to Workflow Orchestration Engine."""
        return self._cognitive_systems.get("workflow_orchestration")

    @property
    def hybrid_search(self) -> HybridSearchEngine:
        """Direct access to Hybrid Search Engine."""
        return self._cognitive_systems.get("hybrid_search")

    @property
    def realtime_monitoring(self) -> MonitoringEngine:
        """Direct access to Real-time Monitoring Engine."""
        return self._cognitive_systems.get("realtime_monitoring")

    # Private methods
    
    async def _initialize_cognitive_systems(self) -> None:
        """Initialize all enabled cognitive systems with 3-phase initialization (matching JavaScript)."""
        logger.info(f"🧠 Initializing {len(self._enabled_systems)} cognitive systems in 3 phases...")

        # Get database connection from storage manager
        db = self.storage_manager.db

        # Phase 1: Core Memory and Basic Systems
        await self._initialize_phase_1_systems(db)

        # Phase 2: Advanced Cognitive Systems
        await self._initialize_phase_2_systems(db)

        # Phase 3: Complex Integration Systems
        await self._initialize_phase_3_systems(db)

        logger.info(f"🎉 ALL {len(self._cognitive_systems)} COGNITIVE SYSTEMS INTEGRATED SUCCESSFULLY!")

    async def _initialize_phase_1_systems(self, db) -> None:
        """Phase 1: Initialize core memory and basic systems."""
        logger.info('🧠 Initializing Phase 1 systems: Core Memory & Basic Systems...')

        phase_1_systems = [
            "semantic_memory", "working_memory", "memory_decay",
            "attention_management", "confidence_tracking", "safety_guardrails"
        ]

        for system_name in phase_1_systems:
            if system_name in self._enabled_systems:
                await self._initialize_single_system(system_name, db)

        logger.info('✅ Phase 1 systems initialized successfully')

    async def _initialize_phase_2_systems(self, db) -> None:
        """Phase 2: Initialize advanced cognitive systems."""
        logger.info('🧠 Initializing Phase 2 systems: Advanced Cognitive Systems...')

        phase_2_systems = [
            "analogical_mapping", "causal_reasoning", "social_intelligence",
            "emotional_intelligence", "cultural_knowledge", "context_injection",
            "vector_search", "goal_hierarchy", "temporal_planning", "skill_capability"
        ]

        for system_name in phase_2_systems:
            if system_name in self._enabled_systems:
                await self._initialize_single_system(system_name, db)

        logger.info('✅ Phase 2 systems initialized successfully')

    async def _initialize_phase_3_systems(self, db) -> None:
        """Phase 3: Initialize complex integration systems."""
        logger.info('🧠 Initializing Phase 3 systems: Complex Integration Systems...')

        phase_3_systems = [
            "episodic_memory", "human_feedback", "self_improvement",
            "multimodal_processing", "tool_interface", "workflow_orchestration",
            "hybrid_search", "realtime_monitoring"
        ]

        for system_name in phase_3_systems:
            if system_name in self._enabled_systems:
                await self._initialize_single_system(system_name, db)

        logger.info('✅ Phase 3 systems initialized successfully')

    async def _initialize_single_system(self, system_name: str, db) -> None:
        """Initialize a single cognitive system."""
        try:
            # Get the system class
            system_class = self._system_classes.get(system_name)
            if not system_class:
                logger.warning(f"No class found for cognitive system: {system_name}")
                return

            # Instantiate the cognitive system
            logger.debug(f"Creating instance of {system_class.__name__} for {system_name}")
            system_instance = system_class(db)

            # Initialize the system
            if hasattr(system_instance, 'initialize'):
                await system_instance.initialize()

            # Store the instance
            self._cognitive_systems[system_name] = system_instance

            logger.debug(f"✅ Initialized cognitive system: {system_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize cognitive system {system_name}: {e}")
            # Continue with other systems instead of failing completely
    
    async def _shutdown_cognitive_systems(self) -> None:
        """Shutdown all cognitive systems."""
        for system_name in self._cognitive_systems:
            try:
                # Perform system-specific shutdown
                # This will be expanded when we implement individual cognitive systems
                self._cognitive_systems[system_name]["initialized"] = False
                logger.debug(f"Shutdown cognitive system: {system_name}")
                
            except Exception as e:
                logger.error(f"Error shutting down cognitive system {system_name}: {e}")
    
    async def _load_system_states(self) -> None:
        """Load system states from storage."""
        logger.debug("Loading cognitive system states from storage...")
        
        for system_name in self._enabled_systems:
            try:
                state = await self.storage_manager.get_cognitive_state(system_name, "global")
                if state:
                    self._system_states[system_name] = state
                    logger.debug(f"Loaded state for system: {system_name}")
                
            except Exception as e:
                logger.warning(f"Could not load state for system {system_name}: {e}")
    
    async def _save_system_states(self) -> None:
        """Save system states to storage."""
        logger.debug("Saving cognitive system states to storage...")
        
        for cache_key, state in self._system_states.items():
            try:
                if ":" in cache_key:
                    system_name, user_id = cache_key.split(":", 1)
                else:
                    system_name, user_id = cache_key, "global"
                
                await self.storage_manager.store_cognitive_state(
                    system_name, user_id, state
                )
                
            except Exception as e:
                logger.warning(f"Could not save state for {cache_key}: {e}")
    
    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring system."""
        logger.debug("Initializing monitoring system...")
        # Monitoring initialization will be implemented with the monitoring system
        pass
    
    def _determine_systems_to_process(
        self, 
        input_data: CognitiveInputData, 
        requested_systems: Optional[List[str]]
    ) -> Set[str]:
        """Determine which cognitive systems should process the input."""
        if requested_systems:
            # Use explicitly requested systems
            return set(requested_systems).intersection(self._enabled_systems)
        
        # Auto-determine based on input characteristics
        systems_to_engage = set()
        
        # Always engage safety if enabled
        if self.config.enable_safety_checks and "safety_guardrails" in self._enabled_systems:
            systems_to_engage.add("safety_guardrails")
        
        # Always engage monitoring if enabled
        if self.config.enable_monitoring and "monitoring" in self._enabled_systems:
            systems_to_engage.add("monitoring")
        
        # Engage systems based on input type and content
        if input_data.text:
            systems_to_engage.update([
                "emotional_intelligence", "semantic_memory", "communication_protocol"
            ])
        
        if input_data.context.user_id:
            systems_to_engage.update([
                "goal_hierarchy", "attention_management", "cultural_knowledge"
            ])
        
        # Filter to only enabled systems
        return systems_to_engage.intersection(self._enabled_systems)
    
    async def _perform_safety_check(self, input_data: CognitiveInputData) -> ValidationResult:
        """Perform safety check on input data."""
        # Basic safety validation - will be expanded with safety system
        violations = []
        warnings = []
        
        if input_data.text and len(input_data.text) > 100000:
            violations.append("Input text exceeds maximum length")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            confidence=1.0 if len(violations) == 0 else 0.0,
            violations=violations,
            warnings=warnings
        )
    
    async def _process_through_systems(
        self, 
        input_data: CognitiveInputData, 
        systems: Set[str], 
        timeout: int
    ) -> Dict[str, Any]:
        """Process input through the specified cognitive systems."""
        results = {}
        
        # Create tasks for parallel processing
        tasks = []
        for system_name in systems:
            task = asyncio.create_task(
                self._process_single_system(system_name, input_data),
                name=f"process_{system_name}"
            )
            tasks.append((system_name, task))
        
        # Wait for all tasks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=timeout
            )
            
            # Collect results
            for system_name, task in tasks:
                try:
                    if task.done():
                        result = task.result()
                        results[system_name] = result
                    else:
                        results[system_name] = {"error": "Task not completed"}
                except Exception as e:
                    results[system_name] = {"error": str(e)}
                    logger.error(f"Error in system {system_name}: {e}")
        
        except asyncio.TimeoutError:
            logger.warning(f"System processing timed out after {timeout} seconds")
            # Cancel remaining tasks
            for _, task in tasks:
                if not task.done():
                    task.cancel()
        
        return results
    
    async def _process_single_system(self, system_name: str, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Process input through a single cognitive system instance."""
        start_time = time.time()

        try:
            # Get the actual cognitive system instance
            system_instance = self._cognitive_systems.get(system_name)
            if not system_instance:
                return {
                    "system": system_name,
                    "status": "error",
                    "error": f"System {system_name} not found or not initialized",
                    "processing_time_ms": 0,
                    "confidence": 0.0,
                }

            # Call the system's process method if it exists
            result = None
            if hasattr(system_instance, 'process'):
                result = await system_instance.process(input_data)
            elif hasattr(system_instance, 'process_input'):
                result = await system_instance.process_input(input_data)
            else:
                # For systems without a standard process method, return basic info
                result = {
                    "system": system_name,
                    "status": "available",
                    "message": f"System {system_name} is available for direct method calls",
                    "methods": [method for method in dir(system_instance) if not method.startswith('_') and callable(getattr(system_instance, method))]
                }

            processing_time = (time.time() - start_time) * 1000

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {
                    "system": system_name,
                    "status": "completed",
                    "result": str(result),
                    "processing_time_ms": processing_time,
                    "confidence": getattr(result, 'confidence', 0.8) if hasattr(result, 'confidence') else 0.8,
                }
            else:
                result.update({
                    "system": system_name,
                    "processing_time_ms": processing_time,
                })

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error processing {system_name}: {e}")
            return {
                "system": system_name,
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time,
                "confidence": 0.0,
            }
    
    async def _aggregate_results(
        self, 
        input_data: CognitiveInputData, 
        cognitive_results: Dict[str, Any], 
        response: CognitiveResponse,
        start_time: float
    ) -> CognitiveResponse:
        """Aggregate results from all cognitive systems."""
        # Update response with cognitive results
        response.cognitive_results = cognitive_results
        
        # Calculate overall confidence
        confidences = [
            result.get("confidence", 0.0) 
            for result in cognitive_results.values() 
            if isinstance(result, dict) and "confidence" in result
        ]
        
        if confidences:
            response.confidence = sum(confidences) / len(confidences)
        else:
            response.confidence = 0.0
        
        # Extract specific cognitive states
        if "emotional_intelligence" in cognitive_results:
            response.emotional_state = cognitive_results["emotional_intelligence"]
        
        if "attention_management" in cognitive_results:
            response.attention_allocation = cognitive_results["attention_management"]
        
        if "goal_hierarchy" in cognitive_results:
            response.goal_hierarchy = cognitive_results["goal_hierarchy"]
        
        # Generate response text if communication protocol was engaged
        if "communication_protocol" in cognitive_results:
            response.response_text = "AI Brain response generated through cognitive processing"
        
        return response
    
    async def _store_processing_result(
        self, 
        input_data: CognitiveInputData, 
        response: CognitiveResponse
    ) -> None:
        """Store processing result for monitoring and analysis."""
        try:
            processing_record = {
                "request_id": input_data.id,
                "user_id": input_data.context.user_id,
                "session_id": input_data.context.session_id,
                "input_type": input_data.input_type,
                "processing_time_ms": response.processing_metadata.processing_time_ms,
                "status": response.status.value,
                "success": response.success,
                "confidence": response.confidence,
                "systems_engaged": list(response.cognitive_results.keys()),
                "timestamp": datetime.utcnow(),
            }
            
            await self.storage_manager.store_document("monitoring_metrics", processing_record)
            
        except Exception as e:
            logger.warning(f"Could not store processing result: {e}")
    
    async def _check_systems_health(self) -> Dict[str, Any]:
        """Check health of all cognitive systems."""
        systems_health = {
            "status": "healthy",
            "enabled_systems": len(self._enabled_systems),
            "total_systems": len(self._cognitive_system_types),
            "system_status": {}
        }
        
        for system_name in self._enabled_systems:
            system_info = self._cognitive_systems.get(system_name, {})
            systems_health["system_status"][system_name] = {
                "initialized": system_info.get("initialized", False),
                "last_activity": system_info.get("last_activity"),
                "status": "healthy" if system_info.get("initialized") else "unhealthy"
            }
        
        return systems_health
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
