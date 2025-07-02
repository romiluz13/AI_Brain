"""
ChangeStreamManager - Real-time Multi-Agent Coordination

Exact Python equivalent of JavaScript ChangeStreamManager.ts with:
- Real-time memory updates across agents
- Event-driven workflow coordination
- Live context synchronization
- Multi-agent conversation awareness
- Automatic conflict resolution
- Performance monitoring and optimization

Features:
- Real-time memory updates across agents
- Event-driven workflow coordination
- Live context synchronization
- Multi-agent conversation awareness
- Automatic conflict resolution
- Performance monitoring and optimization
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from bson import ObjectId
import json

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorChangeStream

from ..core.interfaces.cognitive_system import CognitiveSystemInterface
from ..core.models import CognitiveInputData, CognitiveResponse
from ..utils.logger import logger


@dataclass
class ChangeStreamConfig:
    """Change stream configuration."""
    enable_memory_sync: bool = True
    enable_workflow_sync: bool = True
    enable_context_sync: bool = True
    enable_safety_sync: bool = True
    batch_size: int = 100
    max_await_time_ms: int = 1000
    resume_after: Optional[Any] = None
    start_at_operation_time: Optional[Any] = None


@dataclass
class AgentCoordinationEvent:
    """Agent coordination event data structure."""
    type: str  # 'memory_update' | 'workflow_step' | 'context_change' | 'safety_alert' | 'agent_join' | 'agent_leave'
    agent_id: str
    session_id: str
    framework: str
    timestamp: datetime
    data: Any
    priority: str  # 'low' | 'medium' | 'high' | 'critical'


@dataclass
class MultiAgentSession:
    """Multi-agent session data structure."""
    session_id: str
    active_agents: Set[str] = field(default_factory=set)
    frameworks: Set[str] = field(default_factory=set)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    shared_context: List[Any] = field(default_factory=list)
    coordination_rules: Dict[str, Any] = field(default_factory=lambda: {
        "allow_memory_sharing": True,
        "allow_workflow_handoff": True,
        "conflict_resolution": "last_wins"
    })


class ChangeStreamManager(CognitiveSystemInterface):
    """
    ChangeStreamManager - Real-time Multi-Agent Coordination
    
    Exact Python equivalent of JavaScript ChangeStreamManager with:
    - Real-time memory updates across agents
    - Event-driven workflow coordination
    - Live context synchronization
    - Multi-agent conversation awareness
    - Automatic conflict resolution
    - Performance monitoring and optimization
    """
    
    def __init__(self, db: AsyncIOMotorDatabase, config: Optional[ChangeStreamConfig] = None):
        super().__init__(db)
        self.db = db
        self.config = config or ChangeStreamConfig()
        self.change_streams: Dict[str, AsyncIOMotorChangeStream] = {}
        self.active_sessions: Dict[str, MultiAgentSession] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.is_running = False
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize change stream monitoring."""
        if self.is_initialized:
            return
            
        logger.info("🔄 Initializing Change Stream Manager...")
        
        try:
            # Start change streams for different collections
            if self.config.enable_memory_sync:
                await self._start_memory_change_stream()
            
            if self.config.enable_workflow_sync:
                await self._start_workflow_change_stream()
            
            if self.config.enable_context_sync:
                await self._start_context_change_stream()
            
            if self.config.enable_safety_sync:
                await self._start_safety_change_stream()
            
            self.is_running = True
            self.is_initialized = True
            logger.info("✅ Change Stream Manager initialized successfully")
            
        except Exception as error:
            logger.error(f"❌ Error initializing Change Stream Manager: {error}")
            raise error
    
    async def process(self, input_data: CognitiveInputData) -> CognitiveResponse:
        """Process change stream coordination requests."""
        try:
            await self.initialize()
            
            # Extract coordination request from input
            request_data = input_data.additional_context.get("coordination_request", {})
            
            if not request_data:
                return CognitiveResponse(
                    response_text="No coordination request provided",
                    confidence=0.0,
                    processing_metadata={
                        "system": "change_stream_manager",
                        "error": "Missing coordination request"
                    }
                )
            
            action = request_data.get("action", "")
            
            if action == "register_agent":
                await self.register_agent(
                    request_data.get("agentId", ""),
                    request_data.get("sessionId", ""),
                    request_data.get("framework", ""),
                    request_data.get("options", {})
                )
                response_text = "Agent registered for coordination"
                
            elif action == "unregister_agent":
                await self.unregister_agent(
                    request_data.get("agentId", ""),
                    request_data.get("sessionId", "")
                )
                response_text = "Agent unregistered from coordination"
                
            elif action == "broadcast_event":
                await self.broadcast_to_session(
                    request_data.get("sessionId", ""),
                    request_data.get("event", {}),
                    request_data.get("excludeAgent")
                )
                response_text = "Event broadcasted to session"
                
            else:
                response_text = f"Unknown coordination action: {action}"
            
            return CognitiveResponse(
                response_text=response_text,
                confidence=0.9,
                processing_metadata={
                    "system": "change_stream_manager",
                    "action": action,
                    "active_sessions": len(self.active_sessions),
                    "change_streams": len(self.change_streams)
                }
            )
            
        except Exception as error:
            logger.error(f"Error in ChangeStreamManager.process: {error}")
            return CognitiveResponse(
                response_text=f"Change stream coordination error: {str(error)}",
                confidence=0.0,
                processing_metadata={
                    "system": "change_stream_manager",
                    "error": str(error)
                }
            )
    
    async def register_agent(
        self,
        agent_id: str,
        session_id: str,
        framework: str,
        options: Dict[str, Any]
    ) -> None:
        """Register an agent for multi-agent coordination."""
        try:
            # Get or create session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = MultiAgentSession(
                    session_id=session_id,
                    coordination_rules={
                        "allow_memory_sharing": options.get("allowMemorySharing", True),
                        "allow_workflow_handoff": options.get("allowWorkflowHandoff", True),
                        "conflict_resolution": options.get("conflictResolution", "last_wins")
                    }
                )
            
            session = self.active_sessions[session_id]
            session.active_agents.add(agent_id)
            session.frameworks.add(framework)
            session.last_activity = datetime.utcnow()
            
            # Emit agent join event
            await self._emit_coordination_event(AgentCoordinationEvent(
                type="agent_join",
                agent_id=agent_id,
                session_id=session_id,
                framework=framework,
                timestamp=datetime.utcnow(),
                data={"options": options},
                priority="medium"
            ))
            
            logger.info(f"Agent {agent_id} registered for session {session_id}")
            
        except Exception as error:
            logger.error(f"Error registering agent: {error}")
            raise error
    
    async def unregister_agent(self, agent_id: str, session_id: str) -> None:
        """Unregister an agent from coordination."""
        try:
            session = self.active_sessions.get(session_id)
            
            if session:
                session.active_agents.discard(agent_id)
                
                # Emit agent leave event
                await self._emit_coordination_event(AgentCoordinationEvent(
                    type="agent_leave",
                    agent_id=agent_id,
                    session_id=session_id,
                    framework="unknown",
                    timestamp=datetime.utcnow(),
                    data={},
                    priority="medium"
                ))
                
                # Remove session if no active agents
                if not session.active_agents:
                    del self.active_sessions[session_id]
                    logger.info(f"Session {session_id} removed (no active agents)")
                
                logger.info(f"Agent {agent_id} unregistered from session {session_id}")
            
        except Exception as error:
            logger.error(f"Error unregistering agent: {error}")
            raise error
    
    async def broadcast_to_session(
        self,
        session_id: str,
        event: Dict[str, Any],
        exclude_agent: Optional[str] = None
    ) -> None:
        """Broadcast event to all agents in a session."""
        try:
            session = self.active_sessions.get(session_id)
            
            if not session:
                logger.warning(f"Session {session_id} not found for broadcast")
                return
            
            # Create coordination event
            coordination_event = AgentCoordinationEvent(
                type=event.get("type", "unknown"),
                agent_id=event.get("agentId", "system"),
                session_id=session_id,
                framework=event.get("framework", "unknown"),
                timestamp=datetime.utcnow(),
                data=event.get("data", {}),
                priority=event.get("priority", "medium")
            )
            
            # Emit to all agents except excluded one
            for agent_id in session.active_agents:
                if agent_id != exclude_agent:
                    await self._emit_coordination_event(coordination_event)
            
            session.last_activity = datetime.utcnow()
            logger.debug(f"Event broadcasted to session {session_id}")
            
        except Exception as error:
            logger.error(f"Error broadcasting to session: {error}")
            raise error

    async def _start_memory_change_stream(self) -> None:
        """Start memory change stream."""
        try:
            memory_collection = self.db.agent_memory

            pipeline = [
                {
                    "$match": {
                        "operationType": {"$in": ["insert", "update", "replace"]},
                        "fullDocument.metadata.framework": {"$exists": True}
                    }
                }
            ]

            change_stream = memory_collection.watch(
                pipeline,
                batch_size=self.config.batch_size,
                max_await_time_ms=self.config.max_await_time_ms
            )

            self.change_streams["memory"] = change_stream

            # Start monitoring in background
            asyncio.create_task(self._monitor_memory_changes(change_stream))
            logger.debug("Memory change stream started")

        except Exception as error:
            logger.error(f"Error starting memory change stream: {error}")
            raise error

    async def _start_workflow_change_stream(self) -> None:
        """Start workflow change stream."""
        try:
            workflow_collection = self.db.agent_workflows

            pipeline = [
                {
                    "$match": {
                        "operationType": {"$in": ["insert", "update", "replace"]},
                        "fullDocument.status": {"$exists": True}
                    }
                }
            ]

            change_stream = workflow_collection.watch(
                pipeline,
                batch_size=self.config.batch_size,
                max_await_time_ms=self.config.max_await_time_ms
            )

            self.change_streams["workflow"] = change_stream

            # Start monitoring in background
            asyncio.create_task(self._monitor_workflow_changes(change_stream))
            logger.debug("Workflow change stream started")

        except Exception as error:
            logger.error(f"Error starting workflow change stream: {error}")
            raise error

    async def _start_context_change_stream(self) -> None:
        """Start context change stream."""
        try:
            context_collection = self.db.agent_context

            pipeline = [
                {
                    "$match": {
                        "operationType": {"$in": ["insert", "update", "replace"]},
                        "fullDocument.agentId": {"$exists": True}
                    }
                }
            ]

            change_stream = context_collection.watch(
                pipeline,
                batch_size=self.config.batch_size,
                max_await_time_ms=self.config.max_await_time_ms
            )

            self.change_streams["context"] = change_stream

            # Start monitoring in background
            asyncio.create_task(self._monitor_context_changes(change_stream))
            logger.debug("Context change stream started")

        except Exception as error:
            logger.error(f"Error starting context change stream: {error}")
            raise error

    async def _start_safety_change_stream(self) -> None:
        """Start safety change stream."""
        try:
            safety_collection = self.db.agent_safety_logs

            pipeline = [
                {
                    "$match": {
                        "operationType": {"$in": ["insert"]},
                        "fullDocument.severity": {"$in": ["high", "critical"]}
                    }
                }
            ]

            change_stream = safety_collection.watch(
                pipeline,
                batch_size=self.config.batch_size,
                max_await_time_ms=self.config.max_await_time_ms
            )

            self.change_streams["safety"] = change_stream

            # Start monitoring in background
            asyncio.create_task(self._monitor_safety_changes(change_stream))
            logger.debug("Safety change stream started")

        except Exception as error:
            logger.error(f"Error starting safety change stream: {error}")
            raise error

    async def _monitor_memory_changes(self, change_stream: AsyncIOMotorChangeStream) -> None:
        """Monitor memory changes."""
        try:
            async for change in change_stream:
                await self._handle_memory_change(change)
        except Exception as error:
            logger.error(f"Error monitoring memory changes: {error}")

    async def _monitor_workflow_changes(self, change_stream: AsyncIOMotorChangeStream) -> None:
        """Monitor workflow changes."""
        try:
            async for change in change_stream:
                await self._handle_workflow_change(change)
        except Exception as error:
            logger.error(f"Error monitoring workflow changes: {error}")

    async def _monitor_context_changes(self, change_stream: AsyncIOMotorChangeStream) -> None:
        """Monitor context changes."""
        try:
            async for change in change_stream:
                await self._handle_context_change(change)
        except Exception as error:
            logger.error(f"Error monitoring context changes: {error}")

    async def _monitor_safety_changes(self, change_stream: AsyncIOMotorChangeStream) -> None:
        """Monitor safety changes."""
        try:
            async for change in change_stream:
                await self._handle_safety_change(change)
        except Exception as error:
            logger.error(f"Error monitoring safety changes: {error}")

    async def _handle_memory_change(self, change: Dict[str, Any]) -> None:
        """Handle memory changes."""
        try:
            document = change.get("fullDocument")

            if not document or not document.get("metadata"):
                return

            metadata = document["metadata"]
            agent_id = document.get("agentId", "unknown")
            session_id = metadata.get("sessionId")

            if session_id and session_id in self.active_sessions:
                await self._emit_coordination_event(AgentCoordinationEvent(
                    type="memory_update",
                    agent_id=agent_id,
                    session_id=session_id,
                    framework=metadata.get("framework", "unknown"),
                    timestamp=datetime.utcnow(),
                    data={
                        "operation": change.get("operationType"),
                        "memoryType": document.get("type"),
                        "content": document.get("content", {})
                    },
                    priority="medium"
                ))

        except Exception as error:
            logger.error(f"Error handling memory change: {error}")

    async def _handle_workflow_change(self, change: Dict[str, Any]) -> None:
        """Handle workflow changes."""
        try:
            document = change.get("fullDocument")

            if not document:
                return

            agent_id = document.get("agentId", "unknown")
            session_id = document.get("sessionId")

            if session_id and session_id in self.active_sessions:
                await self._emit_coordination_event(AgentCoordinationEvent(
                    type="workflow_step",
                    agent_id=agent_id,
                    session_id=session_id,
                    framework=document.get("framework", "unknown"),
                    timestamp=datetime.utcnow(),
                    data={
                        "operation": change.get("operationType"),
                        "status": document.get("status"),
                        "step": document.get("currentStep"),
                        "progress": document.get("progress", 0)
                    },
                    priority="high" if document.get("status") == "completed" else "medium"
                ))

        except Exception as error:
            logger.error(f"Error handling workflow change: {error}")

    async def _handle_context_change(self, change: Dict[str, Any]) -> None:
        """Handle context changes."""
        try:
            document = change.get("fullDocument")

            if not document:
                return

            agent_id = document.get("agentId", "unknown")
            session_id = document.get("sessionId")

            if session_id and session_id in self.active_sessions:
                await self._emit_coordination_event(AgentCoordinationEvent(
                    type="context_change",
                    agent_id=agent_id,
                    session_id=session_id,
                    framework=document.get("framework", "unknown"),
                    timestamp=datetime.utcnow(),
                    data={
                        "operation": change.get("operationType"),
                        "contextType": document.get("type"),
                        "context": document.get("context", {})
                    },
                    priority="medium"
                ))

        except Exception as error:
            logger.error(f"Error handling context change: {error}")

    async def _handle_safety_change(self, change: Dict[str, Any]) -> None:
        """Handle safety changes (alerts)."""
        try:
            document = change.get("fullDocument")

            if not document:
                return

            agent_id = document.get("agentId", "unknown")
            session_id = document.get("sessionId")
            severity = document.get("severity", "medium")

            # Broadcast safety alerts to all sessions if critical
            if severity == "critical":
                for session_id, session in self.active_sessions.items():
                    await self._emit_coordination_event(AgentCoordinationEvent(
                        type="safety_alert",
                        agent_id=agent_id,
                        session_id=session_id,
                        framework=document.get("framework", "unknown"),
                        timestamp=datetime.utcnow(),
                        data={
                            "severity": severity,
                            "alertType": document.get("alertType"),
                            "message": document.get("message"),
                            "details": document.get("details", {})
                        },
                        priority="critical"
                    ))
            elif session_id and session_id in self.active_sessions:
                await self._emit_coordination_event(AgentCoordinationEvent(
                    type="safety_alert",
                    agent_id=agent_id,
                    session_id=session_id,
                    framework=document.get("framework", "unknown"),
                    timestamp=datetime.utcnow(),
                    data={
                        "severity": severity,
                        "alertType": document.get("alertType"),
                        "message": document.get("message"),
                        "details": document.get("details", {})
                    },
                    priority="high"
                ))

        except Exception as error:
            logger.error(f"Error handling safety change: {error}")

    async def _emit_coordination_event(self, event: AgentCoordinationEvent) -> None:
        """Emit coordination event."""
        try:
            # Call registered event handlers
            event_type = event.type
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as handler_error:
                        logger.error(f"Error in event handler: {handler_error}")

            # Also call generic coordination event handlers
            if "coordination_event" in self.event_handlers:
                for handler in self.event_handlers["coordination_event"]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as handler_error:
                        logger.error(f"Error in coordination event handler: {handler_error}")

            logger.debug(f"Emitted coordination event: {event.type} for agent {event.agent_id}")

        except Exception as error:
            logger.error(f"Error emitting coordination event: {error}")

    def on(self, event_type: str, handler: Callable) -> None:
        """Register event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event type: {event_type}")

    def off(self, event_type: str, handler: Callable) -> None:
        """Unregister event handler."""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                logger.debug(f"Unregistered handler for event type: {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for event type: {event_type}")

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        return {
            "sessionId": session.session_id,
            "activeAgents": list(session.active_agents),
            "frameworks": list(session.frameworks),
            "lastActivity": session.last_activity.isoformat(),
            "sharedContext": session.shared_context,
            "coordinationRules": session.coordination_rules
        }

    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions."""
        sessions = []
        for session in self.active_sessions.values():
            sessions.append({
                "sessionId": session.session_id,
                "activeAgents": list(session.active_agents),
                "frameworks": list(session.frameworks),
                "lastActivity": session.last_activity.isoformat(),
                "agentCount": len(session.active_agents)
            })
        return sessions

    async def shutdown(self) -> None:
        """Shutdown change streams."""
        logger.info("🛑 Shutting down Change Stream Manager...")

        # Close all change streams
        for name, stream in self.change_streams.items():
            try:
                await stream.close()
                logger.debug(f"Closed change stream: {name}")
            except Exception as error:
                logger.error(f"Error closing change stream {name}: {error}")

        self.change_streams.clear()
        self.active_sessions.clear()
        self.event_handlers.clear()
        self.is_running = False

        logger.info("✅ Change Stream Manager shutdown complete")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.shutdown()
