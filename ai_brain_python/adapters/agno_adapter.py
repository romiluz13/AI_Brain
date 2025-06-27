"""
Agno Adapter

Integrates the Universal AI Brain with Agno framework.
Provides advanced cognitive-enhanced AI agents with Agno's capabilities.

Features:
- Cognitive-enhanced Agno agents with AI Brain integration
- Advanced agent orchestration with cognitive coordination
- Multi-modal cognitive processing with Agno
- Enhanced tool integration and workflow management
- Cognitive memory and learning integration
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union, Callable
from datetime import datetime

try:
    from agno.agent import Agent as AgnoAgent
    from agno.team import Team
    from agno.models.openai import OpenAIChat
    from agno.models.anthropic import Claude
    AGNO_AVAILABLE = True
except ImportError:
    # Fallback classes for when Agno is not installed
    class AgnoAgent:
        def __init__(self, model, tools=None, instructions=None, markdown=True, **kwargs):
            self.model = model
            self.tools = tools or []
            self.instructions = instructions
            self.markdown = markdown
        def print_response(self, prompt, **kwargs):
            return "Agno not available"
    class Team:
        def __init__(self, mode, members, model, **kwargs):
            self.mode = mode
            self.members = members
            self.model = model
        def print_response(self, prompt, **kwargs):
            return "Agno not available"
    class OpenAIChat:
        def __init__(self, id):
            self.id = id
    class Claude:
        def __init__(self, id):
            self.id = id
    AGNO_AVAILABLE = False

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext
from ai_brain_python.adapters.base_adapter import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class CognitiveAgnoAgent(AgnoAgent if AGNO_AVAILABLE else object):
    """
    Cognitive-enhanced Agno Agent with AI Brain integration.

    Extends Agno Agent with comprehensive cognitive capabilities
    including emotional intelligence, goal hierarchy, and advanced reasoning.

    Uses the exact Agno Agent API: model, tools, instructions, markdown, etc.
    """

    def __init__(
        self,
        model,
        ai_brain_config: Optional[UniversalAIBrainConfig] = None,
        cognitive_systems: Optional[List[str]] = None,
        tools: Optional[List] = None,
        instructions: Optional[Union[str, List[str]]] = None,
        markdown: bool = True,
        name: Optional[str] = None,
        role: Optional[str] = None,
        show_tool_calls: bool = False,
        **kwargs
    ):
        if not AGNO_AVAILABLE:
            raise ImportError("Agno is not installed. Please install it with: pip install agno")

        # Initialize Agno Agent with exact API parameters
        super().__init__(
            model=model,
            tools=tools or [],
            instructions=instructions or "You are a cognitive AI agent with advanced reasoning capabilities.",
            markdown=markdown,
            name=name,
            role=role,
            show_tool_calls=show_tool_calls,
            **kwargs
        )
        
        # AI Brain integration
        self.ai_brain_config = ai_brain_config
        self.ai_brain: Optional[UniversalAIBrain] = None
        self.cognitive_systems = cognitive_systems or [
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "attention_management",
            "semantic_memory",
            "safety_guardrails",
            "self_improvement"
        ]
        
        # Cognitive state
        self.agent_id = f"agno_agent_{name}_{id(self)}"
        self.cognitive_state = {
            "emotional_context": {},
            "goal_hierarchy": {},
            "confidence_level": 0.8,
            "attention_focus": [],
            "memory_context": {},
            "learning_progress": {}
        }
        
        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        self.cognitive_insights: List[Dict[str, Any]] = []
    
    async def initialize_cognitive_capabilities(self) -> None:
        """Initialize AI Brain cognitive capabilities."""
        if self.ai_brain_config and not self.ai_brain:
            self.ai_brain = UniversalAIBrain(self.ai_brain_config)
            await self.ai_brain.initialize()
            logger.info(f"AI Brain initialized for Agno agent: {self.name}")
    
    async def cognitive_run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run agent with cognitive enhancement.
        
        Processes the task through AI Brain cognitive systems
        and provides enhanced reasoning and decision-making.
        """
        if not self.ai_brain:
            await self.initialize_cognitive_capabilities()
        
        start_time = datetime.utcnow()
        
        try:
            # Process through AI Brain cognitive systems
            cognitive_analysis = await self._process_with_cognitive_systems(task, context or {})
            
            # Update agent cognitive state
            await self._update_cognitive_state(cognitive_analysis)
            
            # Generate enhanced task execution plan
            execution_plan = await self._generate_cognitive_execution_plan(task, cognitive_analysis)
            
            # Execute task with cognitive enhancement
            if hasattr(super(), 'run'):
                # Use Agno's native run method if available
                base_result = await super().run(task, **kwargs)
            else:
                # Fallback execution
                base_result = await self._fallback_execution(task, execution_plan)
            
            # Post-process with cognitive insights
            enhanced_result = await self._enhance_result_with_cognitive_insights(
                base_result, cognitive_analysis, execution_plan
            )
            
            # Learn from interaction
            await self._learn_from_interaction(task, enhanced_result, cognitive_analysis)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "result": enhanced_result,
                "cognitive_analysis": cognitive_analysis.cognitive_results if cognitive_analysis else {},
                "execution_plan": execution_plan,
                "cognitive_state": self.cognitive_state.copy(),
                "processing_time_ms": processing_time,
                "confidence": cognitive_analysis.confidence if cognitive_analysis else 0.5,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error in cognitive Agno agent run: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "result": f"Error processing task: {str(e)}",
                "error": str(e),
                "processing_time_ms": processing_time,
                "confidence": 0.0,
                "agent_id": self.agent_id
            }
    
    async def _process_with_cognitive_systems(self, task: str, context: Dict[str, Any]):
        """Process task through AI Brain cognitive systems."""
        if not self.ai_brain:
            return None
        
        cognitive_context = CognitiveContext(
            user_id=self.agent_id,
            session_id=f"agno_session_{datetime.utcnow().timestamp()}"
        )
        
        input_data = CognitiveInputData(
            text=task,
            input_type="agno_agent_task",
            context=cognitive_context,
            requested_systems=self.cognitive_systems,
            processing_priority=8
        )
        
        return await self.ai_brain.process_input(input_data)
    
    async def _update_cognitive_state(self, cognitive_analysis) -> None:
        """Update agent cognitive state based on analysis."""
        if not cognitive_analysis:
            return
        
        cognitive_results = cognitive_analysis.cognitive_results
        
        # Update emotional context
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            self.cognitive_state["emotional_context"] = emotional_result.get("emotional_state", {})
        
        # Update goal hierarchy
        if "goal_hierarchy" in cognitive_results:
            goal_result = cognitive_results["goal_hierarchy"]
            self.cognitive_state["goal_hierarchy"] = {
                "extracted_goals": goal_result.get("extracted_goals", []),
                "goal_priorities": goal_result.get("goal_priorities", {})
            }
        
        # Update confidence level
        self.cognitive_state["confidence_level"] = cognitive_analysis.confidence
        
        # Update attention focus
        if "attention_management" in cognitive_results:
            attention_result = cognitive_results["attention_management"]
            self.cognitive_state["attention_focus"] = attention_result.get("attention_allocation", {})
        
        # Update memory context
        if "semantic_memory" in cognitive_results:
            memory_result = cognitive_results["semantic_memory"]
            self.cognitive_state["memory_context"] = {
                "relevant_memories": memory_result.get("relevant_memories", []),
                "memory_insights": memory_result.get("memory_insights", {})
            }
    
    async def _generate_cognitive_execution_plan(self, task: str, cognitive_analysis) -> Dict[str, Any]:
        """Generate cognitive execution plan based on analysis."""
        execution_plan = {
            "task": task,
            "approach": "standard",
            "steps": [],
            "cognitive_considerations": [],
            "risk_assessment": {},
            "success_criteria": []
        }
        
        if not cognitive_analysis:
            return execution_plan
        
        cognitive_results = cognitive_analysis.cognitive_results
        
        # Determine approach based on cognitive analysis
        if "goal_hierarchy" in cognitive_results:
            goal_result = cognitive_results["goal_hierarchy"]
            goals = goal_result.get("extracted_goals", [])
            if goals:
                execution_plan["approach"] = "goal_oriented"
                execution_plan["steps"] = [f"Address goal: {goal.get('text', '')}" for goal in goals[:3]]
        
        # Add cognitive considerations
        if "emotional_intelligence" in cognitive_results:
            emotional_state = cognitive_results["emotional_intelligence"].get("emotional_state", {})
            if emotional_state.get("primary_emotion") != "neutral":
                execution_plan["cognitive_considerations"].append(
                    f"Consider emotional context: {emotional_state.get('primary_emotion', 'unknown')}"
                )
        
        # Add safety considerations
        if "safety_guardrails" in cognitive_results:
            safety_result = cognitive_results["safety_guardrails"]
            safety_assessment = safety_result.get("safety_assessment", {})
            execution_plan["risk_assessment"] = {
                "safety_level": safety_assessment.get("safety_level", "safe"),
                "risk_score": safety_assessment.get("risk_score", 0.0)
            }
        
        # Define success criteria
        execution_plan["success_criteria"] = [
            f"Confidence level > {self.cognitive_state['confidence_level']:.2f}",
            "Safety requirements met",
            "Goal alignment achieved"
        ]
        
        return execution_plan
    
    async def _fallback_execution(self, task: str, execution_plan: Dict[str, Any]) -> str:
        """Fallback execution when Agno's run method is not available."""
        approach = execution_plan.get("approach", "standard")
        steps = execution_plan.get("steps", [])
        
        result = f"Cognitive execution of task: {task}\n"
        result += f"Approach: {approach}\n"
        
        if steps:
            result += "Execution steps:\n"
            for i, step in enumerate(steps, 1):
                result += f"{i}. {step}\n"
        
        # Add cognitive insights
        cognitive_considerations = execution_plan.get("cognitive_considerations", [])
        if cognitive_considerations:
            result += "\nCognitive considerations:\n"
            for consideration in cognitive_considerations:
                result += f"- {consideration}\n"
        
        return result
    
    async def _enhance_result_with_cognitive_insights(
        self,
        base_result: Any,
        cognitive_analysis,
        execution_plan: Dict[str, Any]
    ) -> str:
        """Enhance result with cognitive insights."""
        enhanced_result = str(base_result)
        
        if cognitive_analysis:
            enhanced_result += f"\n\nCognitive Enhancement:\n"
            enhanced_result += f"- Overall Confidence: {cognitive_analysis.confidence:.2f}\n"
            
            # Add emotional insights
            emotional_context = self.cognitive_state.get("emotional_context", {})
            if emotional_context.get("primary_emotion"):
                enhanced_result += f"- Emotional Context: {emotional_context['primary_emotion']}\n"
            
            # Add goal insights
            goal_hierarchy = self.cognitive_state.get("goal_hierarchy", {})
            goals = goal_hierarchy.get("extracted_goals", [])
            if goals:
                enhanced_result += f"- Identified Goals: {len(goals)}\n"
            
            # Add safety assessment
            risk_assessment = execution_plan.get("risk_assessment", {})
            if risk_assessment:
                enhanced_result += f"- Safety Level: {risk_assessment.get('safety_level', 'unknown')}\n"
        
        return enhanced_result
    
    async def _learn_from_interaction(
        self,
        task: str,
        result: str,
        cognitive_analysis
    ) -> None:
        """Learn from interaction for future improvements."""
        interaction_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "task": task,
            "result_length": len(str(result)),
            "confidence": cognitive_analysis.confidence if cognitive_analysis else 0.5,
            "cognitive_state_snapshot": self.cognitive_state.copy()
        }
        
        self.interaction_history.append(interaction_entry)
        
        # Keep only recent history
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
        
        # Extract cognitive insights
        if cognitive_analysis:
            insight_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "systems_used": list(cognitive_analysis.cognitive_results.keys()),
                "overall_confidence": cognitive_analysis.confidence,
                "processing_time": cognitive_analysis.processing_time_ms
            }
            
            self.cognitive_insights.append(insight_entry)
            
            # Keep only recent insights
            if len(self.cognitive_insights) > 50:
                self.cognitive_insights = self.cognitive_insights[-50:]
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive capabilities and performance."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "cognitive_systems": self.cognitive_systems,
            "current_cognitive_state": self.cognitive_state.copy(),
            "interaction_count": len(self.interaction_history),
            "cognitive_insights_count": len(self.cognitive_insights),
            "average_confidence": (
                sum(h.get("confidence", 0) for h in self.interaction_history) / 
                max(1, len(self.interaction_history))
            ),
            "last_interaction": (
                self.interaction_history[-1]["timestamp"] 
                if self.interaction_history else None
            )
        }


class CognitiveAgnoTeam(Team if AGNO_AVAILABLE else object):
    """
    Cognitive-enhanced Agno Team with AI Brain coordination.

    Provides team orchestration with cognitive coordination
    between multiple agents using the exact Agno Team API.

    Uses the exact Agno Team API: mode, members, model, success_criteria, etc.
    """

    def __init__(
        self,
        mode: str,
        members: List[CognitiveAgnoAgent],
        model,
        ai_brain: Optional[UniversalAIBrain] = None,
        success_criteria: Optional[str] = None,
        instructions: Optional[List[str]] = None,
        show_tool_calls: bool = False,
        markdown: bool = True,
        **kwargs
    ):
        if not AGNO_AVAILABLE:
            raise ImportError("Agno is not installed. Please install it with: pip install agno")

        # Initialize Agno Team with exact API parameters
        super().__init__(
            mode=mode,
            members=members,
            model=model,
            success_criteria=success_criteria or "Complete the task with cognitive enhancement",
            instructions=instructions or ["Use cognitive capabilities for enhanced reasoning"],
            show_tool_calls=show_tool_calls,
            markdown=markdown,
            **kwargs
        )

        # AI Brain integration
        self.ai_brain = ai_brain
        self.team_id = f"agno_team_{mode}_{id(self)}"

        # Team cognitive state
        self.team_cognitive_state = {
            "coordination_level": 0.8,
            "collective_confidence": 0.0,
            "shared_context": {},
            "team_insights": []
        }

        # Execution history
        self.execution_history: List[Dict[str, Any]] = []

    def print_response(self, prompt: str, stream: bool = False, **kwargs) -> str:
        """Standard Agno Team print_response method."""
        if AGNO_AVAILABLE:
            # Use the actual Agno Team print_response method
            return super().print_response(prompt, stream=stream, **kwargs)
        else:
            # Fallback when Agno is not available
            return "Agno not available - cognitive execution recommended"
    
    async def cognitive_execute(
        self,
        workflow_input: Dict[str, Any],
        coordination_strategy: str = "collaborative"
    ) -> Dict[str, Any]:
        """Execute workflow with cognitive coordination."""
        start_time = datetime.utcnow()
        
        try:
            # Initialize all agent cognitive capabilities
            for agent in self.cognitive_agents:
                await agent.initialize_cognitive_capabilities()
            
            # Analyze workflow requirements
            workflow_analysis = await self._analyze_workflow_requirements(workflow_input)
            
            # Execute workflow with cognitive coordination
            execution_results = await self._execute_with_cognitive_coordination(
                workflow_input, coordination_strategy, workflow_analysis
            )
            
            # Generate workflow insights
            workflow_insights = await self._generate_workflow_insights(execution_results)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record execution
            execution_record = {
                "timestamp": start_time.isoformat(),
                "processing_time_ms": processing_time,
                "agents_involved": len(self.cognitive_agents),
                "coordination_strategy": coordination_strategy,
                "collective_confidence": self.workflow_cognitive_state["collective_confidence"]
            }
            
            self.execution_history.append(execution_record)
            
            return {
                "workflow_id": self.workflow_id,
                "execution_results": execution_results,
                "workflow_analysis": workflow_analysis,
                "workflow_insights": workflow_insights,
                "cognitive_state": self.workflow_cognitive_state.copy(),
                "processing_time_ms": processing_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in cognitive Agno workflow execution: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "workflow_id": self.workflow_id,
                "error": str(e),
                "processing_time_ms": processing_time,
                "success": False
            }
    
    async def _analyze_workflow_requirements(self, workflow_input: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow requirements using AI Brain."""
        if not self.ai_brain:
            return {"analysis": "No AI Brain available for workflow analysis"}
        
        input_text = str(workflow_input)
        
        cognitive_context = CognitiveContext(
            user_id=self.workflow_id,
            session_id=f"workflow_analysis_{datetime.utcnow().timestamp()}"
        )
        
        input_data = CognitiveInputData(
            text=input_text,
            input_type="agno_workflow_analysis",
            context=cognitive_context,
            requested_systems=["workflow_orchestration", "goal_hierarchy", "attention_management"],
            processing_priority=7
        )
        
        response = await self.ai_brain.process_input(input_data)
        
        return {
            "workflow_complexity": "medium",  # Would be determined by analysis
            "coordination_needs": "high",
            "cognitive_insights": response.cognitive_results,
            "recommended_strategy": "collaborative"
        }
    
    async def _execute_with_cognitive_coordination(
        self,
        workflow_input: Dict[str, Any],
        coordination_strategy: str,
        workflow_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute workflow with cognitive coordination between agents."""
        results = []
        
        # Simple sequential execution with shared context
        shared_context = workflow_input.copy()
        
        for i, agent in enumerate(self.cognitive_agents):
            task = f"Agent {agent.name} task from workflow: {workflow_input.get('task', 'general task')}"
            
            # Add shared context from previous agents
            agent_context = shared_context.copy()
            agent_context["agent_index"] = i
            agent_context["previous_results"] = results
            
            # Execute agent task
            agent_result = await agent.cognitive_run(task, agent_context)
            results.append(agent_result)
            
            # Update shared context
            shared_context[f"agent_{i}_result"] = agent_result
            
            # Update workflow cognitive state
            self._update_workflow_cognitive_state(agent_result)
        
        return results
    
    def _update_workflow_cognitive_state(self, agent_result: Dict[str, Any]) -> None:
        """Update workflow cognitive state based on agent results."""
        # Update collective confidence
        agent_confidence = agent_result.get("confidence", 0.0)
        current_confidence = self.workflow_cognitive_state["collective_confidence"]
        
        # Simple averaging for collective confidence
        agent_count = len([r for r in self.execution_history if "collective_confidence" in r])
        self.workflow_cognitive_state["collective_confidence"] = (
            (current_confidence * agent_count + agent_confidence) / (agent_count + 1)
        )
        
        # Update shared context
        cognitive_state = agent_result.get("cognitive_state", {})
        if cognitive_state:
            self.workflow_cognitive_state["shared_context"].update(cognitive_state)
    
    async def _generate_workflow_insights(self, execution_results: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from workflow execution."""
        insights = []
        
        if execution_results:
            avg_confidence = sum(r.get("confidence", 0) for r in execution_results) / len(execution_results)
            insights.append(f"Average agent confidence: {avg_confidence:.2f}")
            
            successful_agents = sum(1 for r in execution_results if r.get("confidence", 0) > 0.7)
            insights.append(f"High-confidence agents: {successful_agents}/{len(execution_results)}")
            
            total_processing_time = sum(r.get("processing_time_ms", 0) for r in execution_results)
            insights.append(f"Total processing time: {total_processing_time:.0f}ms")
        
        insights.append(f"Workflow coordination level: {self.workflow_cognitive_state['coordination_level']:.2f}")
        
        return insights


class AgnoAdapter(BaseFrameworkAdapter):
    """
    Agno Framework Adapter for Universal AI Brain.
    
    Provides advanced cognitive-enhanced AI agents with Agno's capabilities.
    """
    
    def __init__(self, ai_brain_config: UniversalAIBrainConfig):
        super().__init__("agno", ai_brain_config)
        
        if not AGNO_AVAILABLE:
            logger.warning("Agno is not installed. Some features may not be available.")
    
    async def _framework_specific_initialization(self) -> None:
        """Agno specific initialization."""
        logger.debug("Agno adapter initialization complete")
    
    async def _framework_specific_shutdown(self) -> None:
        """Agno specific shutdown."""
        logger.debug("Agno adapter shutdown complete")
    
    def _get_default_systems(self) -> List[str]:
        """Get default cognitive systems for Agno."""
        return [
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "attention_management",
            "semantic_memory",
            "safety_guardrails",
            "self_improvement",
            "workflow_orchestration"
        ]
    
    async def _enhance_response(self, response, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Agno specific enhancements."""
        return self._format_response_for_framework(response, {
            "agno_features": [
                "Cognitive Agents",
                "Cognitive Workflows",
                "Advanced Reasoning",
                "Multi-Modal Processing",
                "Learning Integration"
            ]
        })
    
    def create_cognitive_agent(
        self,
        model,
        cognitive_systems: Optional[List[str]] = None,
        tools: Optional[List] = None,
        instructions: Optional[Union[str, List[str]]] = None,
        markdown: bool = True,
        name: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs
    ) -> CognitiveAgnoAgent:
        """Create a cognitive-enhanced Agno agent using exact API."""
        return CognitiveAgnoAgent(
            model=model,
            ai_brain_config=self.ai_brain_config,
            cognitive_systems=cognitive_systems,
            tools=tools,
            instructions=instructions,
            markdown=markdown,
            name=name,
            role=role,
            **kwargs
        )

    def create_cognitive_team(
        self,
        mode: str,
        members: List[CognitiveAgnoAgent],
        model,
        success_criteria: Optional[str] = None,
        instructions: Optional[List[str]] = None,
        **kwargs
    ) -> CognitiveAgnoTeam:
        """Create a cognitive-enhanced Agno team using exact API."""
        return CognitiveAgnoTeam(
            mode=mode,
            members=members,
            model=model,
            ai_brain=self.ai_brain,
            success_criteria=success_criteria,
            instructions=instructions,
            **kwargs
        )
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get Agno framework information."""
        return {
            "framework": "Agno",
            "version": "latest",
            "available": AGNO_AVAILABLE,
            "cognitive_features": [
                "Cognitive Agents",
                "Cognitive Workflows",
                "Advanced Reasoning",
                "Multi-Agent Coordination",
                "Learning Integration",
                "Tool Integration"
            ],
            "integration_level": "advanced",
            "components": [
                "CognitiveAgnoAgent",
                "CognitiveAgnoTeam"
            ]
        }
