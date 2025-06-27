"""
CrewAI Adapter

Integrates the Universal AI Brain with CrewAI framework.
Provides seamless integration with CrewAI agents, crews, and tools.

Features:
- Native CrewAI agent integration with AI Brain capabilities
- Cognitive-enhanced crew orchestration
- AI Brain tools as CrewAI tools
- Advanced memory and learning integration
- Multi-agent cognitive coordination
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime

try:
    from crewai import Agent, Crew, Task, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    # Fallback classes for when CrewAI is not installed
    class Agent:
        def __init__(self, role, goal, backstory, **kwargs):
            self.role = role
            self.goal = goal
            self.backstory = backstory
    class Crew:
        def __init__(self, agents, tasks, **kwargs):
            self.agents = agents
            self.tasks = tasks
        def kickoff(self):
            return "CrewAI not available"
    class Task:
        def __init__(self, description, expected_output, agent, **kwargs):
            self.description = description
            self.expected_output = expected_output
            self.agent = agent
    class Process:
        sequential = "sequential"
        hierarchical = "hierarchical"
    class BaseTool:
        pass
    CREWAI_AVAILABLE = False

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext
from ai_brain_python.adapters.base_adapter import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class CognitiveAgent(Agent if CREWAI_AVAILABLE else object):
    """
    Enhanced CrewAI Agent with AI Brain cognitive capabilities.
    
    Extends CrewAI Agent with emotional intelligence, goal hierarchy,
    memory, and all 16 cognitive systems.
    """
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        ai_brain_config: Optional[UniversalAIBrainConfig] = None,
        cognitive_systems: Optional[List[str]] = None,
        tools: Optional[List] = None,
        llm: Optional[str] = None,
        memory: bool = True,
        verbose: bool = False,
        allow_delegation: bool = False,
        max_iter: int = 20,
        **kwargs
    ):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not installed. Please install it with: pip install crewai")

        # Initialize CrewAI Agent with exact API parameters
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm=llm,
            memory=memory,
            verbose=verbose,
            allow_delegation=allow_delegation,
            max_iter=max_iter,
            **kwargs
        )
        
        # Initialize AI Brain
        self.ai_brain_config = ai_brain_config
        self.ai_brain: Optional[UniversalAIBrain] = None
        self.cognitive_systems = cognitive_systems or [
            "emotional_intelligence",
            "goal_hierarchy", 
            "confidence_tracking",
            "attention_management",
            "semantic_memory",
            "safety_guardrails"
        ]
        
        # Agent cognitive state
        self.agent_id = f"crewai_agent_{id(self)}"
        self.cognitive_history: List[Dict[str, Any]] = []
        
        # Enhanced agent properties
        self.emotional_state = "neutral"
        self.confidence_level = 0.8
        self.learning_enabled = True
    
    async def initialize_ai_brain(self) -> None:
        """Initialize the AI Brain for this agent."""
        if self.ai_brain_config and not self.ai_brain:
            self.ai_brain = UniversalAIBrain(self.ai_brain_config)
            await self.ai_brain.initialize()
            logger.info(f"AI Brain initialized for agent: {self.role}")
    
    async def cognitive_execute(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute task with cognitive enhancement."""
        if not self.ai_brain:
            await self.initialize_ai_brain()
        
        if not self.ai_brain:
            # Fallback to standard execution
            return {"result": "AI Brain not available", "confidence": 0.5}
        
        # Create cognitive input
        cognitive_context = CognitiveContext(
            user_id=self.agent_id,
            session_id=f"task_{datetime.utcnow().timestamp()}"
        )
        
        input_data = CognitiveInputData(
            text=task_description,
            input_type="task_execution",
            context=cognitive_context,
            requested_systems=self.cognitive_systems,
            processing_priority=8
        )
        
        # Process through AI Brain
        response = await self.ai_brain.process_input(input_data)
        
        # Update agent state based on cognitive results
        await self._update_agent_state(response)
        
        # Record cognitive history
        self.cognitive_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "task": task_description,
            "cognitive_response": response.cognitive_results,
            "confidence": response.confidence
        })
        
        return {
            "result": self._generate_enhanced_response(task_description, response),
            "confidence": response.confidence,
            "cognitive_insights": self._extract_cognitive_insights(response),
            "emotional_state": self.emotional_state,
            "agent_learning": self._get_learning_summary()
        }
    
    async def _update_agent_state(self, response) -> None:
        """Update agent state based on cognitive results."""
        cognitive_results = response.cognitive_results
        
        # Update emotional state
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            if "emotional_state" in emotional_result:
                self.emotional_state = emotional_result["emotional_state"].get("primary_emotion", "neutral")
        
        # Update confidence level
        if "confidence_tracking" in cognitive_results:
            confidence_result = cognitive_results["confidence_tracking"]
            if "confidence_assessment" in confidence_result:
                self.confidence_level = confidence_result["confidence_assessment"].get("overall_confidence", 0.8)
    
    def _generate_enhanced_response(self, task: str, response) -> str:
        """Generate enhanced response using cognitive insights."""
        base_response = f"Task: {task}\n\n"
        
        # Add cognitive insights
        cognitive_results = response.cognitive_results
        
        if "goal_hierarchy" in cognitive_results:
            goal_result = cognitive_results["goal_hierarchy"]
            if "extracted_goals" in goal_result:
                goals = goal_result["extracted_goals"]
                if goals:
                    base_response += f"Identified Goals: {', '.join([g.get('text', '') for g in goals[:3]])}\n"
        
        if "emotional_intelligence" in cognitive_results:
            emotional_result = cognitive_results["emotional_intelligence"]
            if "empathy_response" in emotional_result:
                empathy = emotional_result["empathy_response"]
                base_response += f"Emotional Context: {empathy.get('response_strategy', 'neutral')}\n"
        
        base_response += f"\nConfidence Level: {self.confidence_level:.2f}\n"
        base_response += f"Emotional State: {self.emotional_state}\n"
        
        return base_response
    
    def _extract_cognitive_insights(self, response) -> Dict[str, Any]:
        """Extract key cognitive insights."""
        insights = {}
        cognitive_results = response.cognitive_results
        
        for system, result in cognitive_results.items():
            if isinstance(result, dict) and result.get("status") == "completed":
                insights[system] = {
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": result.get("processing_time_ms", 0),
                    "key_insights": self._get_system_insights(system, result)
                }
        
        return insights
    
    def _get_system_insights(self, system: str, result: Dict[str, Any]) -> List[str]:
        """Get key insights from a cognitive system."""
        insights = []
        
        if system == "emotional_intelligence" and "emotional_state" in result:
            emotional_state = result["emotional_state"]
            insights.append(f"Primary emotion: {emotional_state.get('primary_emotion', 'unknown')}")
            insights.append(f"Emotion intensity: {emotional_state.get('emotion_intensity', 0):.2f}")
        
        elif system == "goal_hierarchy" and "extracted_goals" in result:
            goals = result["extracted_goals"]
            insights.append(f"Identified {len(goals)} goals")
            if goals:
                insights.append(f"Top priority: {goals[0].get('text', 'unknown')}")
        
        elif system == "confidence_tracking" and "confidence_assessment" in result:
            confidence = result["confidence_assessment"]
            insights.append(f"Overall confidence: {confidence.get('overall_confidence', 0):.2f}")
            insights.append(f"Uncertainty level: {confidence.get('epistemic_uncertainty', 0):.2f}")
        
        return insights
    
    def _get_learning_summary(self) -> Dict[str, Any]:
        """Get learning summary for the agent."""
        return {
            "total_tasks": len(self.cognitive_history),
            "average_confidence": sum(h.get("confidence", 0) for h in self.cognitive_history) / max(1, len(self.cognitive_history)),
            "learning_enabled": self.learning_enabled,
            "cognitive_systems_active": len(self.cognitive_systems)
        }


class CognitiveCrew(Crew if CREWAI_AVAILABLE else object):
    """
    Enhanced CrewAI Crew with AI Brain coordination.
    
    Provides cognitive coordination between agents with shared memory,
    emotional intelligence, and collaborative decision-making.
    """
    
    def __init__(
        self,
        agents: List[CognitiveAgent],
        tasks: List[Task],
        ai_brain_config: Optional[UniversalAIBrainConfig] = None,
        enable_cognitive_coordination: bool = True,
        process: Optional[str] = None,
        verbose: bool = False,
        manager_llm: Optional[str] = None,
        **kwargs
    ):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not installed. Please install it with: pip install crewai")

        # Initialize CrewAI Crew with exact API parameters
        super().__init__(
            agents=agents,
            tasks=tasks,
            process=process or Process.sequential,
            verbose=verbose,
            manager_llm=manager_llm,
            **kwargs
        )
        
        self.ai_brain_config = ai_brain_config
        self.enable_cognitive_coordination = enable_cognitive_coordination
        self.crew_ai_brain: Optional[UniversalAIBrain] = None
        self.crew_id = f"crewai_crew_{id(self)}"
        
        # Crew cognitive state
        self.shared_memory: Dict[str, Any] = {}
        self.crew_emotional_state = "collaborative"
        self.coordination_history: List[Dict[str, Any]] = []
    
    async def initialize_crew_brain(self) -> None:
        """Initialize crew-level AI Brain for coordination."""
        if self.ai_brain_config and not self.crew_ai_brain:
            self.crew_ai_brain = UniversalAIBrain(self.ai_brain_config)
            await self.crew_ai_brain.initialize()
            
            # Initialize all agent brains
            for agent in self.agents:
                if isinstance(agent, CognitiveAgent):
                    await agent.initialize_ai_brain()
            
            logger.info(f"Crew AI Brain initialized with {len(self.agents)} cognitive agents")

    def kickoff(self) -> str:
        """Standard CrewAI kickoff method."""
        if CREWAI_AVAILABLE:
            # Use the actual CrewAI kickoff method
            return super().kickoff()
        else:
            # Fallback when CrewAI is not available
            return "CrewAI not available - cognitive kickoff recommended"

    async def cognitive_kickoff(self) -> Dict[str, Any]:
        """Enhanced kickoff with cognitive coordination."""
        if not self.crew_ai_brain:
            await self.initialize_crew_brain()
        
        # Pre-execution cognitive analysis
        crew_analysis = await self._analyze_crew_dynamics()
        
        # Execute tasks with cognitive coordination
        results = []
        for i, task in enumerate(self.tasks):
            task_result = await self._execute_cognitive_task(task, i)
            results.append(task_result)
            
            # Update shared memory
            await self._update_shared_memory(task, task_result)
        
        # Post-execution analysis
        crew_summary = await self._generate_crew_summary(results)
        
        return {
            "results": results,
            "crew_analysis": crew_analysis,
            "crew_summary": crew_summary,
            "shared_memory": self.shared_memory,
            "coordination_insights": self._get_coordination_insights()
        }
    
    async def _analyze_crew_dynamics(self) -> Dict[str, Any]:
        """Analyze crew dynamics and coordination needs."""
        if not self.crew_ai_brain:
            return {"analysis": "No crew brain available"}
        
        # Analyze crew composition and task distribution
        crew_description = f"Crew with {len(self.agents)} agents and {len(self.tasks)} tasks. "
        crew_description += f"Agent roles: {', '.join([agent.role for agent in self.agents])}"
        
        cognitive_context = CognitiveContext(
            user_id=self.crew_id,
            session_id=f"crew_analysis_{datetime.utcnow().timestamp()}"
        )
        
        input_data = CognitiveInputData(
            text=crew_description,
            input_type="crew_analysis",
            context=cognitive_context,
            requested_systems=["workflow_orchestration", "attention_management", "goal_hierarchy"],
            processing_priority=7
        )
        
        response = await self.crew_ai_brain.process_input(input_data)
        
        return {
            "crew_composition": len(self.agents),
            "task_complexity": len(self.tasks),
            "cognitive_insights": response.cognitive_results,
            "coordination_strategy": self._determine_coordination_strategy(response)
        }
    
    async def _execute_cognitive_task(self, task: Task, task_index: int) -> Dict[str, Any]:
        """Execute task with cognitive coordination."""
        # Select best agent for task
        selected_agent = await self._select_agent_for_task(task)
        
        if isinstance(selected_agent, CognitiveAgent):
            # Execute with cognitive enhancement
            task_description = getattr(task, 'description', str(task))
            context = {
                "task_index": task_index,
                "shared_memory": self.shared_memory,
                "crew_context": True
            }
            
            result = await selected_agent.cognitive_execute(task_description, context)
            
            # Record coordination
            self.coordination_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_index": task_index,
                "agent_role": selected_agent.role,
                "result_confidence": result.get("confidence", 0.0)
            })
            
            return result
        else:
            # Fallback for non-cognitive agents
            return {"result": "Task executed by standard agent", "confidence": 0.5}
    
    async def _select_agent_for_task(self, task: Task) -> CognitiveAgent:
        """Select the best agent for a specific task."""
        # Simple selection - in production would use more sophisticated matching
        if self.agents:
            return self.agents[0]
        raise ValueError("No agents available")
    
    async def _update_shared_memory(self, task: Task, result: Dict[str, Any]) -> None:
        """Update shared memory with task results."""
        task_key = f"task_{len(self.shared_memory)}"
        self.shared_memory[task_key] = {
            "task": str(task),
            "result": result.get("result", ""),
            "confidence": result.get("confidence", 0.0),
            "timestamp": datetime.utcnow().isoformat(),
            "cognitive_insights": result.get("cognitive_insights", {})
        }
    
    async def _generate_crew_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive crew execution summary."""
        total_confidence = sum(r.get("confidence", 0) for r in results)
        avg_confidence = total_confidence / len(results) if results else 0
        
        return {
            "total_tasks": len(results),
            "average_confidence": avg_confidence,
            "crew_emotional_state": self.crew_emotional_state,
            "coordination_effectiveness": self._calculate_coordination_effectiveness(),
            "learning_insights": self._extract_crew_learning_insights(results)
        }
    
    def _determine_coordination_strategy(self, response) -> str:
        """Determine coordination strategy based on cognitive analysis."""
        cognitive_results = response.cognitive_results
        
        if "workflow_orchestration" in cognitive_results:
            workflow_result = cognitive_results["workflow_orchestration"]
            complexity = workflow_result.get("workflow_plan", {}).get("complexity", "medium")
            
            if complexity == "high":
                return "hierarchical_coordination"
            elif complexity == "medium":
                return "collaborative_coordination"
            else:
                return "autonomous_coordination"
        
        return "default_coordination"
    
    def _calculate_coordination_effectiveness(self) -> float:
        """Calculate coordination effectiveness score."""
        if not self.coordination_history:
            return 0.5
        
        # Simple effectiveness calculation
        avg_confidence = sum(h.get("result_confidence", 0) for h in self.coordination_history) / len(self.coordination_history)
        return min(1.0, avg_confidence + 0.1)  # Slight boost for coordination
    
    def _extract_crew_learning_insights(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract learning insights from crew execution."""
        insights = []
        
        if results:
            avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results)
            insights.append(f"Average task confidence: {avg_confidence:.2f}")
            
            if avg_confidence > 0.8:
                insights.append("High confidence execution - crew is well-coordinated")
            elif avg_confidence < 0.6:
                insights.append("Lower confidence - consider agent specialization or training")
        
        insights.append(f"Coordination events: {len(self.coordination_history)}")
        insights.append(f"Shared memory entries: {len(self.shared_memory)}")
        
        return insights
    
    def _get_coordination_insights(self) -> Dict[str, Any]:
        """Get coordination insights and metrics."""
        return {
            "coordination_events": len(self.coordination_history),
            "shared_memory_size": len(self.shared_memory),
            "crew_emotional_state": self.crew_emotional_state,
            "coordination_effectiveness": self._calculate_coordination_effectiveness()
        }


class CrewAIAdapter(BaseFrameworkAdapter):
    """
    CrewAI Framework Adapter for Universal AI Brain.
    
    Provides seamless integration between AI Brain and CrewAI framework.
    """
    
    def __init__(self, ai_brain_config: UniversalAIBrainConfig):
        super().__init__("crewai", ai_brain_config)
        
        if not CREWAI_AVAILABLE:
            logger.warning("CrewAI is not installed. Some features may not be available.")
    
    def create_cognitive_agent(
        self,
        role: str,
        goal: str,
        backstory: str,
        cognitive_systems: Optional[List[str]] = None,
        **kwargs
    ) -> CognitiveAgent:
        """Create a cognitive-enhanced CrewAI agent."""
        return CognitiveAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            ai_brain_config=self.ai_brain_config,
            cognitive_systems=cognitive_systems,
            **kwargs
        )
    
    def create_cognitive_crew(
        self,
        agents: List[CognitiveAgent],
        tasks: List[Task],
        enable_cognitive_coordination: bool = True,
        **kwargs
    ) -> CognitiveCrew:
        """Create a cognitive-enhanced CrewAI crew."""
        return CognitiveCrew(
            agents=agents,
            tasks=tasks,
            ai_brain_config=self.ai_brain_config,
            enable_cognitive_coordination=enable_cognitive_coordination,
            **kwargs
        )

    def create_task(
        self,
        description: str,
        expected_output: str,
        agent: CognitiveAgent,
        tools: Optional[List] = None,
        **kwargs
    ) -> Task:
        """Create a CrewAI task with exact API parameters."""
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            tools=tools,
            **kwargs
        )

    async def enhance_existing_agent(self, agent: Agent) -> CognitiveAgent:
        """Enhance an existing CrewAI agent with cognitive capabilities."""
        # Convert existing agent to cognitive agent
        cognitive_agent = CognitiveAgent(
            role=agent.role,
            goal=agent.goal,
            backstory=agent.backstory,
            ai_brain_config=self.ai_brain_config
        )
        
        await cognitive_agent.initialize_ai_brain()
        return cognitive_agent
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get CrewAI framework information."""
        return {
            "framework": "CrewAI",
            "version": "latest",
            "available": CREWAI_AVAILABLE,
            "cognitive_features": [
                "Cognitive Agents",
                "Cognitive Crews", 
                "Shared Memory",
                "Emotional Intelligence",
                "Goal Hierarchy",
                "Confidence Tracking",
                "Cognitive Coordination"
            ],
            "integration_level": "native"
        }
