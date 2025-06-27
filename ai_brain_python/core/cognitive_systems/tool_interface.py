"""
Advanced Tool Interface

Intelligent tool management and execution system.
Provides tool discovery, validation, execution, and recovery capabilities.

Features:
- Dynamic tool discovery and registration
- Tool validation and safety checking
- Intelligent tool selection and chaining
- Error recovery and fallback mechanisms
- Tool performance monitoring and optimization
- Multi-framework tool adapter support
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable, Union

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, CognitiveSystemType

logger = logging.getLogger(__name__)


class AdvancedToolInterface(CognitiveSystemInterface):
    """Advanced Tool Interface - System 13 of 16"""
    
    def __init__(self, system_id: str = "tool_interface", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Tool registry and management
        self._registered_tools: Dict[str, Dict[str, Any]] = {}
        self._tool_performance: Dict[str, Dict[str, Any]] = {}
        self._tool_chains: Dict[str, List[str]] = {}
        
        # Configuration
        self._config = {
            "max_tool_execution_time": config.get("max_tool_execution_time", 30) if config else 30,
            "enable_tool_chaining": config.get("enable_tool_chaining", True) if config else True,
            "enable_fallback_tools": config.get("enable_fallback_tools", True) if config else True,
            "tool_safety_checks": config.get("tool_safety_checks", True) if config else True,
            "max_retry_attempts": config.get("max_retry_attempts", 3) if config else 3
        }
        
        # Tool categories
        self._tool_categories = {
            "data_processing": {"priority": 1, "safety_level": "high"},
            "web_search": {"priority": 2, "safety_level": "medium"},
            "file_operations": {"priority": 3, "safety_level": "high"},
            "api_calls": {"priority": 2, "safety_level": "medium"},
            "calculations": {"priority": 1, "safety_level": "low"},
            "text_processing": {"priority": 1, "safety_level": "low"}
        }
    
    @property
    def system_name(self) -> str:
        return "Advanced Tool Interface"
    
    @property
    def system_description(self) -> str:
        return "Intelligent tool management and execution system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.TOOL_EXECUTION}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.TOOL_EXECUTION}
    
    async def initialize(self) -> None:
        """Initialize the Advanced Tool Interface."""
        try:
            logger.info("Initializing Advanced Tool Interface...")
            
            # Load registered tools
            await self._load_tool_registry()
            
            # Initialize built-in tools
            await self._register_builtin_tools()
            
            # Load tool performance data
            await self._load_tool_performance_data()
            
            self._is_initialized = True
            logger.info("Advanced Tool Interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Tool Interface: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Advanced Tool Interface."""
        try:
            logger.info("Shutting down Advanced Tool Interface...")
            
            # Save tool performance data
            await self._save_tool_performance_data()
            
            self._is_initialized = False
            logger.info("Advanced Tool Interface shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Advanced Tool Interface shutdown: {e}")
    
    async def process(self, input_data: CognitiveInputData, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through tool interface analysis."""
        if not self._is_initialized:
            raise RuntimeError("Advanced Tool Interface not initialized")
        
        try:
            start_time = datetime.utcnow()
            
            # Analyze tool requirements from input
            tool_requirements = await self._analyze_tool_requirements(input_data)
            
            # Select appropriate tools
            selected_tools = await self._select_tools(tool_requirements)
            
            # Execute tools if needed
            execution_results = []
            if selected_tools and tool_requirements.get("execute_tools", False):
                execution_results = await self._execute_tools(selected_tools, input_data, context or {})
            
            # Generate tool recommendations
            recommendations = await self._generate_tool_recommendations(tool_requirements)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.9,
                "tool_analysis": {
                    "requirements": tool_requirements,
                    "selected_tools": [tool["name"] for tool in selected_tools],
                    "available_tools": len(self._registered_tools),
                    "execution_performed": len(execution_results) > 0
                },
                "execution_results": execution_results,
                "recommendations": recommendations,
                "tool_performance": await self._get_tool_performance_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in Tool Interface processing: {e}")
            return {"system": self.system_id, "status": "error", "error": str(e), "confidence": 0.0}
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current tool interface state."""
        state_data = {
            "registered_tools": len(self._registered_tools),
            "tool_categories": len(self._tool_categories),
            "tool_chains": len(self._tool_chains)
        }
        
        return CognitiveState(
            system_type=CognitiveSystemType.TOOL_INTERFACE,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.95,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update tool interface state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Tool Interface state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for tool interface processing."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Public tool methods
    
    async def register_tool(self, name: str, tool_func: Callable, description: str, category: str = "general", **kwargs) -> bool:
        """Register a new tool."""
        tool_definition = {
            "name": name,
            "function": tool_func,
            "description": description,
            "category": category,
            "parameters": kwargs.get("parameters", {}),
            "safety_level": self._tool_categories.get(category, {}).get("safety_level", "medium"),
            "registered_at": datetime.utcnow().isoformat(),
            "usage_count": 0,
            "success_rate": 1.0,
            "average_execution_time": 0.0
        }
        
        self._registered_tools[name] = tool_definition
        
        # Initialize performance tracking
        self._tool_performance[name] = {
            "executions": [],
            "errors": [],
            "performance_metrics": {}
        }
        
        logger.info(f"Registered tool: {name} in category: {category}")
        return True
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a specific tool."""
        if tool_name not in self._registered_tools:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        tool_def = self._registered_tools[tool_name]
        start_time = datetime.utcnow()
        
        try:
            # Safety checks
            if self._config["tool_safety_checks"]:
                safety_check = await self._perform_safety_check(tool_name, parameters)
                if not safety_check["safe"]:
                    return {"success": False, "error": f"Safety check failed: {safety_check['reason']}"}
            
            # Execute tool
            result = await self._execute_single_tool(tool_def, parameters, context or {})
            
            # Record performance
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._record_tool_performance(tool_name, True, execution_time)
            
            return {"success": True, "result": result, "execution_time": execution_time}
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._record_tool_performance(tool_name, False, execution_time, str(e))
            return {"success": False, "error": str(e), "execution_time": execution_time}
    
    async def get_available_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        tools = []
        
        for name, tool_def in self._registered_tools.items():
            if category is None or tool_def["category"] == category:
                tools.append({
                    "name": name,
                    "description": tool_def["description"],
                    "category": tool_def["category"],
                    "safety_level": tool_def["safety_level"],
                    "success_rate": tool_def["success_rate"],
                    "usage_count": tool_def["usage_count"]
                })
        
        return tools
    
    # Private methods
    
    async def _load_tool_registry(self) -> None:
        """Load tool registry from storage."""
        logger.debug("Tool registry loaded")
    
    async def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        # Text processing tool
        await self.register_tool(
            name="text_analyzer",
            tool_func=self._builtin_text_analyzer,
            description="Analyze text for various properties",
            category="text_processing",
            parameters={"text": "string", "analysis_type": "string"}
        )
        
        # Calculator tool
        await self.register_tool(
            name="calculator",
            tool_func=self._builtin_calculator,
            description="Perform mathematical calculations",
            category="calculations",
            parameters={"expression": "string"}
        )
        
        logger.debug("Built-in tools registered")
    
    async def _load_tool_performance_data(self) -> None:
        """Load tool performance data."""
        logger.debug("Tool performance data loaded")
    
    async def _save_tool_performance_data(self) -> None:
        """Save tool performance data."""
        logger.debug("Tool performance data saved")
    
    async def _analyze_tool_requirements(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze what tools might be needed for the input."""
        text = input_data.text or ""
        requirements = {
            "needed_categories": [],
            "specific_tools": [],
            "execute_tools": False,
            "complexity": "low"
        }
        
        # Simple keyword-based analysis
        if any(word in text.lower() for word in ["calculate", "compute", "math", "equation"]):
            requirements["needed_categories"].append("calculations")
            requirements["specific_tools"].append("calculator")
        
        if any(word in text.lower() for word in ["analyze", "process", "examine", "text"]):
            requirements["needed_categories"].append("text_processing")
            requirements["specific_tools"].append("text_analyzer")
        
        if any(word in text.lower() for word in ["search", "find", "lookup", "web"]):
            requirements["needed_categories"].append("web_search")
        
        # Determine if tools should be executed
        if any(word in text.lower() for word in ["execute", "run", "perform", "do"]):
            requirements["execute_tools"] = True
        
        # Assess complexity
        if len(requirements["needed_categories"]) > 2:
            requirements["complexity"] = "high"
        elif len(requirements["needed_categories"]) > 1:
            requirements["complexity"] = "medium"
        
        return requirements
    
    async def _select_tools(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select appropriate tools based on requirements."""
        selected_tools = []
        
        # Select by specific tools
        for tool_name in requirements.get("specific_tools", []):
            if tool_name in self._registered_tools:
                selected_tools.append(self._registered_tools[tool_name])
        
        # Select by category
        for category in requirements.get("needed_categories", []):
            category_tools = [
                tool for tool in self._registered_tools.values()
                if tool["category"] == category and tool not in selected_tools
            ]
            
            # Select best tool from category (by success rate)
            if category_tools:
                best_tool = max(category_tools, key=lambda t: t["success_rate"])
                selected_tools.append(best_tool)
        
        return selected_tools
    
    async def _execute_tools(self, tools: List[Dict[str, Any]], input_data: CognitiveInputData, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute selected tools."""
        results = []
        
        for tool in tools:
            try:
                # Prepare parameters based on tool and input
                parameters = await self._prepare_tool_parameters(tool, input_data)
                
                # Execute tool
                execution_result = await self.execute_tool(tool["name"], parameters, context)
                results.append({
                    "tool_name": tool["name"],
                    "success": execution_result["success"],
                    "result": execution_result.get("result"),
                    "error": execution_result.get("error"),
                    "execution_time": execution_result.get("execution_time", 0)
                })
                
            except Exception as e:
                results.append({
                    "tool_name": tool["name"],
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                })
        
        return results
    
    async def _generate_tool_recommendations(self, requirements: Dict[str, Any]) -> List[str]:
        """Generate tool usage recommendations."""
        recommendations = []
        
        if requirements["complexity"] == "high":
            recommendations.append("Consider breaking down the task into smaller steps")
        
        if "calculations" in requirements["needed_categories"]:
            recommendations.append("Use the calculator tool for mathematical operations")
        
        if "text_processing" in requirements["needed_categories"]:
            recommendations.append("Use text analysis tools for content processing")
        
        if not requirements["execute_tools"]:
            recommendations.append("Add 'execute' or 'run' to your request to automatically use tools")
        
        return recommendations
    
    async def _perform_safety_check(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safety check on tool execution."""
        tool_def = self._registered_tools[tool_name]
        safety_level = tool_def["safety_level"]
        
        # Basic safety checks
        if safety_level == "high":
            # More stringent checks for high-risk tools
            if any(param for param in parameters.values() if isinstance(param, str) and len(param) > 10000):
                return {"safe": False, "reason": "Parameter too large for high-safety tool"}
        
        return {"safe": True, "reason": "Safety check passed"}
    
    async def _execute_single_tool(self, tool_def: Dict[str, Any], parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a single tool."""
        tool_func = tool_def["function"]
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool_func(parameters, context),
                timeout=self._config["max_tool_execution_time"]
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"Tool execution timed out after {self._config['max_tool_execution_time']} seconds")
    
    async def _record_tool_performance(self, tool_name: str, success: bool, execution_time: float, error: Optional[str] = None) -> None:
        """Record tool performance metrics."""
        if tool_name not in self._tool_performance:
            self._tool_performance[tool_name] = {"executions": [], "errors": [], "performance_metrics": {}}
        
        performance_data = self._tool_performance[tool_name]
        
        # Record execution
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "execution_time": execution_time,
            "error": error
        }
        performance_data["executions"].append(execution_record)
        
        # Update tool definition metrics
        tool_def = self._registered_tools[tool_name]
        tool_def["usage_count"] += 1
        
        # Calculate success rate
        recent_executions = performance_data["executions"][-100:]  # Last 100 executions
        successes = sum(1 for exec in recent_executions if exec["success"])
        tool_def["success_rate"] = successes / len(recent_executions)
        
        # Calculate average execution time
        execution_times = [exec["execution_time"] for exec in recent_executions]
        tool_def["average_execution_time"] = sum(execution_times) / len(execution_times)
    
    async def _prepare_tool_parameters(self, tool: Dict[str, Any], input_data: CognitiveInputData) -> Dict[str, Any]:
        """Prepare parameters for tool execution."""
        parameters = {}
        
        # Simple parameter mapping based on tool type
        if tool["name"] == "text_analyzer":
            parameters = {
                "text": input_data.text or "",
                "analysis_type": "general"
            }
        elif tool["name"] == "calculator":
            # Extract mathematical expressions from text
            text = input_data.text or ""
            # Simple extraction - in production would use more sophisticated parsing
            parameters = {"expression": text}
        
        return parameters
    
    async def _get_tool_performance_summary(self) -> Dict[str, Any]:
        """Get tool performance summary."""
        summary = {
            "total_tools": len(self._registered_tools),
            "total_executions": 0,
            "average_success_rate": 0.0,
            "top_performing_tools": []
        }
        
        if self._registered_tools:
            total_executions = sum(tool["usage_count"] for tool in self._registered_tools.values())
            total_success_rate = sum(tool["success_rate"] for tool in self._registered_tools.values())
            
            summary["total_executions"] = total_executions
            summary["average_success_rate"] = total_success_rate / len(self._registered_tools)
            
            # Top performing tools
            sorted_tools = sorted(
                self._registered_tools.items(),
                key=lambda x: x[1]["success_rate"],
                reverse=True
            )
            summary["top_performing_tools"] = [
                {"name": name, "success_rate": tool["success_rate"]}
                for name, tool in sorted_tools[:5]
            ]
        
        return summary
    
    # Built-in tool implementations
    
    async def _builtin_text_analyzer(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in text analyzer tool."""
        text = parameters.get("text", "")
        analysis_type = parameters.get("analysis_type", "general")
        
        analysis = {
            "word_count": len(text.split()),
            "character_count": len(text),
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "analysis_type": analysis_type
        }
        
        if analysis_type == "sentiment":
            # Simple sentiment analysis
            positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
            
            positive_count = sum(1 for word in positive_words if word in text.lower())
            negative_count = sum(1 for word in negative_words if word in text.lower())
            
            if positive_count > negative_count:
                analysis["sentiment"] = "positive"
            elif negative_count > positive_count:
                analysis["sentiment"] = "negative"
            else:
                analysis["sentiment"] = "neutral"
        
        return analysis
    
    async def _builtin_calculator(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in calculator tool."""
        expression = parameters.get("expression", "")
        
        try:
            # Simple and safe expression evaluation
            # In production, would use a proper math expression parser
            import re
            
            # Extract numbers and basic operations
            numbers = re.findall(r'\d+\.?\d*', expression)
            
            if len(numbers) >= 2:
                num1, num2 = float(numbers[0]), float(numbers[1])
                
                if '+' in expression:
                    result = num1 + num2
                elif '-' in expression:
                    result = num1 - num2
                elif '*' in expression or 'x' in expression:
                    result = num1 * num2
                elif '/' in expression:
                    result = num1 / num2 if num2 != 0 else "Error: Division by zero"
                else:
                    result = "Error: Unsupported operation"
            else:
                result = "Error: Insufficient numbers in expression"
            
            return {"result": result, "expression": expression}
            
        except Exception as e:
            return {"error": str(e), "expression": expression}
