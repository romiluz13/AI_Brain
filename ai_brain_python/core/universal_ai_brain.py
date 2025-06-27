"""
Universal AI Brain - Main Orchestrator

The core orchestrator that coordinates all 16 cognitive systems:
- Emotional Intelligence Engine
- Goal Hierarchy Manager
- Confidence Tracking Engine
- Attention Management System
- Cultural Knowledge Engine
- Skill Capability Manager
- Communication Protocol Manager
- Temporal Planning Engine
- Semantic Memory Engine
- Safety Guardrails Engine
- Self-Improvement Engine
- Real-time Monitoring Engine
- Advanced Tool Interface
- Workflow Orchestration Engine
- Multi-Modal Processing Engine
- Human Feedback Integration Engine

Features:
- Async processing with high performance
- Framework-agnostic design
- MongoDB Atlas native integration
- Real-time state management
- Comprehensive error handling
- Performance monitoring
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

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

logger = logging.getLogger(__name__)


class UniversalAIBrainConfig:
    """Configuration for the Universal AI Brain."""
    
    def __init__(
        self,
        storage_config: StorageConfig,
        enable_all_systems: bool = True,
        enabled_systems: Optional[Set[str]] = None,
        max_concurrent_processing: int = 10,
        default_timeout: int = 30,
        enable_monitoring: bool = True,
        enable_safety_checks: bool = True,
        performance_tracking: bool = True,
    ):
        self.storage_config = storage_config
        self.enable_all_systems = enable_all_systems
        self.enabled_systems = enabled_systems or set()
        self.max_concurrent_processing = max_concurrent_processing
        self.default_timeout = default_timeout
        self.enable_monitoring = enable_monitoring
        self.enable_safety_checks = enable_safety_checks
        self.performance_tracking = performance_tracking


class UniversalAIBrain:
    """
    Universal AI Brain - Main orchestrator for all cognitive systems.
    
    This class coordinates the 16 cognitive systems and provides a unified
    interface for processing cognitive requests across different AI frameworks.
    """
    
    def __init__(self, config: UniversalAIBrainConfig):
        """Initialize the Universal AI Brain."""
        self.config = config
        self.storage_manager = StorageManager(config.storage_config)
        
        # System state
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_processing)
        
        # Cognitive systems registry
        self._cognitive_systems: Dict[str, Any] = {}
        self._system_states: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self._processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "last_request_time": None,
        }
        
        # Define the 16 cognitive systems
        self._cognitive_system_types = {
            "emotional_intelligence": CognitiveSystemType.EMOTIONAL_INTELLIGENCE,
            "goal_hierarchy": CognitiveSystemType.GOAL_HIERARCHY,
            "confidence_tracking": CognitiveSystemType.CONFIDENCE_TRACKING,
            "attention_management": CognitiveSystemType.ATTENTION_MANAGEMENT,
            "cultural_knowledge": CognitiveSystemType.CULTURAL_KNOWLEDGE,
            "skill_capability": CognitiveSystemType.SKILL_CAPABILITY,
            "communication_protocol": CognitiveSystemType.COMMUNICATION_PROTOCOL,
            "temporal_planning": CognitiveSystemType.TEMPORAL_PLANNING,
            "semantic_memory": CognitiveSystemType.SEMANTIC_MEMORY,
            "safety_guardrails": CognitiveSystemType.SAFETY_GUARDRAILS,
            "self_improvement": CognitiveSystemType.SELF_IMPROVEMENT,
            "monitoring": CognitiveSystemType.MONITORING,
            "tool_interface": CognitiveSystemType.TOOL_INTERFACE,
            "workflow_orchestration": CognitiveSystemType.WORKFLOW_ORCHESTRATION,
            "multimodal_processing": CognitiveSystemType.MULTIMODAL_PROCESSING,
            "human_feedback": CognitiveSystemType.HUMAN_FEEDBACK,
        }
        
        # Determine enabled systems
        if config.enable_all_systems:
            self._enabled_systems = set(self._cognitive_system_types.keys())
        else:
            self._enabled_systems = config.enabled_systems.intersection(
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
                if self.config.enable_monitoring:
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
                if self.config.enable_safety_checks:
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
                if self.config.performance_tracking:
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
    
    # Private methods
    
    async def _initialize_cognitive_systems(self) -> None:
        """Initialize all enabled cognitive systems."""
        logger.info(f"Initializing {len(self._enabled_systems)} cognitive systems...")
        
        for system_name in self._enabled_systems:
            try:
                # Initialize system-specific components
                # This will be expanded when we implement individual cognitive systems
                self._cognitive_systems[system_name] = {
                    "type": self._cognitive_system_types[system_name],
                    "initialized": True,
                    "last_activity": datetime.utcnow(),
                }
                
                logger.debug(f"Initialized cognitive system: {system_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize cognitive system {system_name}: {e}")
                raise
    
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
        """Process input through a single cognitive system."""
        # Placeholder implementation - will be replaced with actual system implementations
        start_time = time.time()
        
        # Simulate processing
        await asyncio.sleep(0.1)  # Simulate processing time
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "system": system_name,
            "status": "completed",
            "processing_time_ms": processing_time,
            "confidence": 0.8,
            "result": f"Processed by {system_name}",
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
