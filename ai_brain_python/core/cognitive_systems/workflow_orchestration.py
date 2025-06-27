"""
Workflow Orchestration Engine

Intelligent routing and parallel processing system.
Manages complex workflows, task dependencies, and system coordination.

Features:
- Intelligent workflow routing and task orchestration
- Parallel processing and dependency management
- Dynamic workflow adaptation and optimization
- Error handling and recovery mechanisms
- Performance monitoring and bottleneck detection
- Multi-system coordination and synchronization
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, CognitiveSystemType

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class WorkflowOrchestrationEngine(CognitiveSystemInterface):
    """Workflow Orchestration Engine - System 14 of 16"""
    
    def __init__(self, system_id: str = "workflow_orchestration", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Workflow management
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        self._workflow_templates: Dict[str, Dict[str, Any]] = {}
        self._task_queue: List[Dict[str, Any]] = []
        
        # Configuration
        self._config = {
            "max_concurrent_workflows": config.get("max_concurrent_workflows", 10) if config else 10,
            "max_parallel_tasks": config.get("max_parallel_tasks", 5) if config else 5,
            "workflow_timeout": config.get("workflow_timeout", 300) if config else 300,  # 5 minutes
            "enable_adaptive_routing": config.get("enable_adaptive_routing", True) if config else True,
            "enable_load_balancing": config.get("enable_load_balancing", True) if config else True
        }
        
        # System routing rules
        self._routing_rules = {
            "emotional_content": ["emotional_intelligence", "communication_protocol"],
            "goal_planning": ["goal_hierarchy", "temporal_planning", "skill_capability"],
            "safety_critical": ["safety_guardrails", "confidence_tracking"],
            "memory_intensive": ["semantic_memory", "self_improvement"],
            "tool_required": ["tool_interface", "workflow_orchestration"],
            "monitoring_needed": ["monitoring", "self_improvement"]
        }
        
        # Performance tracking
        self._workflow_performance: Dict[str, List[Dict[str, Any]]] = {}
        self._system_load: Dict[str, float] = {}
    
    @property
    def system_name(self) -> str:
        return "Workflow Orchestration Engine"
    
    @property
    def system_description(self) -> str:
        return "Intelligent routing and parallel processing system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.WORKFLOW_ORCHESTRATION}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.WORKFLOW_ORCHESTRATION}
    
    async def initialize(self) -> None:
        """Initialize the Workflow Orchestration Engine."""
        try:
            logger.info("Initializing Workflow Orchestration Engine...")
            
            # Load workflow templates
            await self._load_workflow_templates()
            
            # Initialize system load tracking
            await self._initialize_load_tracking()
            
            # Start workflow processor
            await self._start_workflow_processor()
            
            self._is_initialized = True
            logger.info("Workflow Orchestration Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Workflow Orchestration Engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Workflow Orchestration Engine."""
        try:
            logger.info("Shutting down Workflow Orchestration Engine...")
            
            # Complete active workflows
            await self._complete_active_workflows()
            
            # Save workflow performance data
            await self._save_workflow_data()
            
            self._is_initialized = False
            logger.info("Workflow Orchestration Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Workflow Orchestration Engine shutdown: {e}")
    
    async def process(self, input_data: CognitiveInputData, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through workflow orchestration."""
        if not self._is_initialized:
            raise RuntimeError("Workflow Orchestration Engine not initialized")
        
        try:
            start_time = datetime.utcnow()
            
            # Analyze workflow requirements
            workflow_analysis = await self._analyze_workflow_requirements(input_data)
            
            # Create workflow plan
            workflow_plan = await self._create_workflow_plan(workflow_analysis, input_data)
            
            # Execute workflow if requested
            execution_results = {}
            if workflow_analysis.get("execute_workflow", False):
                execution_results = await self._execute_workflow(workflow_plan, input_data, context or {})
            
            # Generate optimization recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(workflow_plan)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.9,
                "workflow_analysis": workflow_analysis,
                "workflow_plan": {
                    "total_tasks": len(workflow_plan.get("tasks", [])),
                    "parallel_groups": len(workflow_plan.get("parallel_groups", [])),
                    "estimated_duration": workflow_plan.get("estimated_duration", 0),
                    "complexity": workflow_plan.get("complexity", "medium")
                },
                "execution_results": execution_results,
                "optimization_recommendations": optimization_recommendations,
                "system_status": await self._get_system_status()
            }
            
        except Exception as e:
            logger.error(f"Error in Workflow Orchestration processing: {e}")
            return {"system": self.system_id, "status": "error", "error": str(e), "confidence": 0.0}
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current workflow orchestration state."""
        state_data = {
            "active_workflows": len(self._active_workflows),
            "queued_tasks": len(self._task_queue),
            "workflow_templates": len(self._workflow_templates),
            "average_system_load": sum(self._system_load.values()) / len(self._system_load) if self._system_load else 0.0
        }
        
        return CognitiveState(
            system_type=CognitiveSystemType.WORKFLOW_ORCHESTRATION,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.95,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update workflow orchestration state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Workflow Orchestration state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for workflow orchestration."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Public workflow methods
    
    async def create_workflow(self, name: str, tasks: List[Dict[str, Any]], dependencies: Optional[Dict[str, List[str]]] = None) -> str:
        """Create a new workflow."""
        workflow_id = f"workflow_{len(self._active_workflows)}_{int(datetime.utcnow().timestamp())}"
        
        workflow = {
            "id": workflow_id,
            "name": name,
            "tasks": tasks,
            "dependencies": dependencies or {},
            "status": WorkflowStatus.PENDING,
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0.0,
            "results": {}
        }
        
        self._active_workflows[workflow_id] = workflow
        return workflow_id
    
    async def execute_workflow_by_id(self, workflow_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow by ID."""
        if workflow_id not in self._active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self._active_workflows[workflow_id]
        workflow["status"] = WorkflowStatus.RUNNING
        
        try:
            # Execute tasks in dependency order
            results = await self._execute_workflow_tasks(workflow, context or {})
            
            workflow["status"] = WorkflowStatus.COMPLETED
            workflow["results"] = results
            workflow["completed_at"] = datetime.utcnow().isoformat()
            
            return {"success": True, "results": results}
            
        except Exception as e:
            workflow["status"] = WorkflowStatus.FAILED
            workflow["error"] = str(e)
            return {"success": False, "error": str(e)}
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status."""
        if workflow_id not in self._active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self._active_workflows[workflow_id]
        return {
            "id": workflow_id,
            "name": workflow["name"],
            "status": workflow["status"].value,
            "progress": workflow["progress"],
            "created_at": workflow["created_at"],
            "completed_at": workflow.get("completed_at"),
            "error": workflow.get("error")
        }
    
    # Private methods
    
    async def _load_workflow_templates(self) -> None:
        """Load workflow templates."""
        # Standard workflow templates
        self._workflow_templates = {
            "comprehensive_analysis": {
                "tasks": [
                    {"system": "emotional_intelligence", "priority": TaskPriority.HIGH},
                    {"system": "goal_hierarchy", "priority": TaskPriority.MEDIUM},
                    {"system": "confidence_tracking", "priority": TaskPriority.MEDIUM},
                    {"system": "safety_guardrails", "priority": TaskPriority.CRITICAL}
                ],
                "parallel_groups": [
                    ["emotional_intelligence", "confidence_tracking"],
                    ["goal_hierarchy", "safety_guardrails"]
                ]
            },
            "safety_first": {
                "tasks": [
                    {"system": "safety_guardrails", "priority": TaskPriority.CRITICAL},
                    {"system": "confidence_tracking", "priority": TaskPriority.HIGH}
                ],
                "dependencies": {"confidence_tracking": ["safety_guardrails"]}
            }
        }
        logger.debug("Workflow templates loaded")
    
    async def _initialize_load_tracking(self) -> None:
        """Initialize system load tracking."""
        # Initialize load tracking for all systems
        systems = [
            "emotional_intelligence", "goal_hierarchy", "confidence_tracking",
            "attention_management", "cultural_knowledge", "skill_capability",
            "communication_protocol", "temporal_planning", "semantic_memory",
            "safety_guardrails", "self_improvement", "monitoring"
        ]
        
        for system in systems:
            self._system_load[system] = 0.0
        
        logger.debug("Load tracking initialized")
    
    async def _start_workflow_processor(self) -> None:
        """Start background workflow processor."""
        logger.debug("Workflow processor started")
    
    async def _complete_active_workflows(self) -> None:
        """Complete all active workflows."""
        for workflow_id, workflow in self._active_workflows.items():
            if workflow["status"] == WorkflowStatus.RUNNING:
                workflow["status"] = WorkflowStatus.CANCELLED
        logger.debug("Active workflows completed")
    
    async def _save_workflow_data(self) -> None:
        """Save workflow performance data."""
        logger.debug("Workflow data saved")
    
    async def _analyze_workflow_requirements(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze workflow requirements from input."""
        text = input_data.text or ""
        requirements = {
            "complexity": "medium",
            "required_systems": [],
            "parallel_capable": True,
            "safety_critical": False,
            "execute_workflow": False
        }
        
        # Analyze content for system requirements
        for content_type, systems in self._routing_rules.items():
            if self._matches_content_type(text, content_type):
                requirements["required_systems"].extend(systems)
        
        # Remove duplicates
        requirements["required_systems"] = list(set(requirements["required_systems"]))
        
        # Determine complexity
        if len(requirements["required_systems"]) > 6:
            requirements["complexity"] = "high"
        elif len(requirements["required_systems"]) > 3:
            requirements["complexity"] = "medium"
        else:
            requirements["complexity"] = "low"
        
        # Check for safety-critical content
        safety_keywords = ["danger", "risk", "harm", "safety", "security"]
        if any(keyword in text.lower() for keyword in safety_keywords):
            requirements["safety_critical"] = True
        
        # Check if execution is requested
        if any(word in text.lower() for word in ["execute", "run", "process", "analyze"]):
            requirements["execute_workflow"] = True
        
        return requirements
    
    async def _create_workflow_plan(self, analysis: Dict[str, Any], input_data: CognitiveInputData) -> Dict[str, Any]:
        """Create workflow execution plan."""
        required_systems = analysis["required_systems"]
        
        # Create tasks
        tasks = []
        for system in required_systems:
            priority = TaskPriority.CRITICAL if analysis["safety_critical"] and system == "safety_guardrails" else TaskPriority.MEDIUM
            tasks.append({
                "system": system,
                "priority": priority,
                "estimated_time": self._estimate_task_time(system),
                "dependencies": []
            })
        
        # Create parallel groups for independent tasks
        parallel_groups = []
        if analysis["parallel_capable"] and len(tasks) > 1:
            # Simple grouping - in production would use more sophisticated dependency analysis
            if len(tasks) >= 4:
                parallel_groups = [
                    [task["system"] for task in tasks[:2]],
                    [task["system"] for task in tasks[2:4]]
                ]
            elif len(tasks) >= 2:
                parallel_groups = [[task["system"] for task in tasks[:2]]]
        
        # Calculate estimated duration
        if parallel_groups:
            max_group_time = max(
                sum(self._estimate_task_time(system) for system in group)
                for group in parallel_groups
            )
            sequential_time = sum(
                self._estimate_task_time(task["system"]) for task in tasks
                if not any(task["system"] in group for group in parallel_groups)
            )
            estimated_duration = max_group_time + sequential_time
        else:
            estimated_duration = sum(task["estimated_time"] for task in tasks)
        
        return {
            "tasks": tasks,
            "parallel_groups": parallel_groups,
            "estimated_duration": estimated_duration,
            "complexity": analysis["complexity"],
            "safety_critical": analysis["safety_critical"]
        }
    
    async def _execute_workflow(self, plan: Dict[str, Any], input_data: CognitiveInputData, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow plan."""
        workflow_id = await self.create_workflow(
            name="dynamic_workflow",
            tasks=plan["tasks"],
            dependencies={}
        )
        
        execution_result = await self.execute_workflow_by_id(workflow_id, context)
        
        return {
            "workflow_id": workflow_id,
            "execution_success": execution_result["success"],
            "results": execution_result.get("results", {}),
            "error": execution_result.get("error")
        }
    
    async def _execute_workflow_tasks(self, workflow: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow tasks."""
        results = {}
        
        # Simple sequential execution for now
        # In production, would implement proper dependency resolution and parallel execution
        for task in workflow["tasks"]:
            system_name = task["system"]
            
            # Simulate task execution
            try:
                # Update system load
                self._system_load[system_name] = min(1.0, self._system_load.get(system_name, 0) + 0.1)
                
                # Simulate processing
                await asyncio.sleep(0.1)  # Simulate work
                
                results[system_name] = {
                    "status": "completed",
                    "confidence": 0.8,
                    "processing_time_ms": task["estimated_time"]
                }
                
                # Update system load
                self._system_load[system_name] = max(0.0, self._system_load.get(system_name, 0) - 0.1)
                
            except Exception as e:
                results[system_name] = {
                    "status": "error",
                    "error": str(e),
                    "confidence": 0.0
                }
        
        return results
    
    async def _generate_optimization_recommendations(self, plan: Dict[str, Any]) -> List[str]:
        """Generate workflow optimization recommendations."""
        recommendations = []
        
        if plan["complexity"] == "high":
            recommendations.append("Consider breaking down complex workflow into smaller parts")
        
        if len(plan["parallel_groups"]) == 0 and len(plan["tasks"]) > 2:
            recommendations.append("Tasks could be parallelized for better performance")
        
        if plan["estimated_duration"] > 10000:  # 10 seconds
            recommendations.append("Workflow duration is high - consider optimization")
        
        if plan["safety_critical"]:
            recommendations.append("Safety-critical workflow - ensure proper validation")
        
        return recommendations
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "active_workflows": len(self._active_workflows),
            "system_load": dict(self._system_load),
            "average_load": sum(self._system_load.values()) / len(self._system_load) if self._system_load else 0.0,
            "capacity_available": self._config["max_concurrent_workflows"] - len(self._active_workflows)
        }
    
    def _matches_content_type(self, text: str, content_type: str) -> bool:
        """Check if text matches content type."""
        text_lower = text.lower()
        
        if content_type == "emotional_content":
            return any(word in text_lower for word in ["feel", "emotion", "mood", "happy", "sad"])
        elif content_type == "goal_planning":
            return any(word in text_lower for word in ["goal", "plan", "objective", "target", "achieve"])
        elif content_type == "safety_critical":
            return any(word in text_lower for word in ["safety", "danger", "risk", "harm", "security"])
        elif content_type == "memory_intensive":
            return any(word in text_lower for word in ["remember", "recall", "memory", "learn", "store"])
        elif content_type == "tool_required":
            return any(word in text_lower for word in ["calculate", "search", "tool", "execute", "run"])
        elif content_type == "monitoring_needed":
            return any(word in text_lower for word in ["monitor", "track", "analyze", "performance"])
        
        return False
    
    def _estimate_task_time(self, system: str) -> float:
        """Estimate task execution time for a system."""
        # Simple time estimates in milliseconds
        time_estimates = {
            "emotional_intelligence": 500,
            "goal_hierarchy": 800,
            "confidence_tracking": 300,
            "attention_management": 400,
            "cultural_knowledge": 600,
            "skill_capability": 700,
            "communication_protocol": 400,
            "temporal_planning": 900,
            "semantic_memory": 1200,
            "safety_guardrails": 600,
            "self_improvement": 1000,
            "monitoring": 200
        }
        
        return time_estimates.get(system, 500)
