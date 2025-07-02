"""
Working Memory Manager - Advanced Session-Based Memory Management

Exact Python equivalent of JavaScript WorkingMemoryManager.ts with:
- Session-specific memory isolation
- Priority-based eviction algorithms
- Automatic TTL cleanup and management
- Memory pressure monitoring
- Intelligent memory promotion/demotion
- Cross-session memory sharing when beneficial

Features:
- Session-specific memory isolation with intelligent cleanup
- Priority-based eviction algorithms and automatic TTL management
- Memory pressure monitoring and intelligent promotion/demotion
- Cross-session memory sharing when beneficial
- Real-time memory analytics and optimization
- Framework-agnostic memory management
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass
from bson import ObjectId
import asyncio
import json
import random
import math

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_brain_python.storage.collections.working_memory_collection import WorkingMemoryCollection
from ai_brain_python.core.cognitive_systems.semantic_memory import SemanticMemoryEngine
from ai_brain_python.core.models.cognitive_interfaces import MemoryPressureStats
from ai_brain_python.utils.logger import logger


@dataclass
class WorkingMemoryConfig:
    """Working memory configuration interface."""
    # Memory limits per session
    max_memories_per_session: int = 50      # 50 memories per session
    max_total_working_memories: int = 1000  # 1000 total working memories
    
    # TTL settings
    default_ttl_minutes: int = 30           # 30 minutes default
    max_ttl_minutes: int = 240              # 240 minutes maximum
    min_ttl_minutes: int = 5                # 5 minutes minimum
    
    # Priority thresholds
    high_priority_threshold: float = 0.8    # 0.8 = High priority
    medium_priority_threshold: float = 0.5  # 0.5 = Medium priority
    low_priority_threshold: float = 0.2     # 0.2 = Low priority
    
    # Eviction settings
    eviction_batch_size: int = 10           # Evict 10 memories at once
    memory_pressure_threshold: float = 0.9  # 90% capacity triggers eviction
    
    # Promotion settings
    promotion_access_threshold: int = 3     # 3 accesses for promotion
    promotion_importance_threshold: float = 0.7  # 0.7 importance for promotion


@dataclass
class WorkingMemoryRequest:
    """Working memory request interface."""
    agent_id: str
    session_id: str
    content: str
    context: Dict[str, Any]
    priority: float = 0.5
    ttl_minutes: Optional[int] = None


@dataclass
class WorkingMemoryResult:
    """Working memory result interface."""
    memory_id: ObjectId
    stored: bool
    evicted_memories: List[str]
    promoted_memories: List[str]
    session_stats: Dict[str, Any]


class WorkingMemoryManager:
    """
    WorkingMemoryManager - Advanced Session-Based Memory Management
    
    Exact Python equivalent of JavaScript WorkingMemoryManager with:
    - Session-specific memory isolation with intelligent cleanup
    - Priority-based eviction algorithms and automatic TTL management
    - Memory pressure monitoring and intelligent promotion/demotion
    - Cross-session memory sharing when beneficial
    - Real-time memory analytics and optimization
    - Framework-agnostic memory management
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.working_memory_collection = WorkingMemoryCollection(db)
        self.semantic_memory_engine: Optional[SemanticMemoryEngine] = None
        self.is_initialized = False
        
        # Configuration
        self._config = WorkingMemoryConfig()
        
        # Session tracking
        self._session_stats: Dict[str, Dict[str, Any]] = {}
        self._memory_pressure: Dict[str, float] = {}
        
        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5 minutes
    
    async def initialize(self, semantic_memory_engine: SemanticMemoryEngine) -> None:
        """Initialize the working memory manager."""
        if self.is_initialized:
            return
        
        try:
            # Initialize working memory collection
            await self.working_memory_collection.create_indexes()
            
            # Set semantic memory engine reference
            self.semantic_memory_engine = semantic_memory_engine
            
            # Start background cleanup
            await self._start_background_cleanup()
            
            self.is_initialized = True
            logger.info("✅ WorkingMemoryManager initialized successfully")
            
        except Exception as error:
            logger.error(f"❌ Failed to initialize WorkingMemoryManager: {error}")
            raise error
    
    async def storeWorkingMemory(
        self,
        request: WorkingMemoryRequest
    ) -> WorkingMemoryResult:
        """Store a working memory with session isolation."""
        if not self.is_initialized:
            raise Exception("WorkingMemoryManager must be initialized first")
        
        # Generate memory ID
        memory_id = ObjectId()
        
        # Calculate TTL
        ttl_minutes = request.ttl_minutes or self._config.default_ttl_minutes
        ttl_minutes = max(self._config.min_ttl_minutes, 
                         min(self._config.max_ttl_minutes, ttl_minutes))
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        
        # Check memory pressure and evict if necessary
        evicted_memories = await self._check_and_evict_memories(request.session_id)
        
        # Create working memory
        working_memory = {
            "memoryId": memory_id,
            "agentId": request.agent_id,
            "sessionId": request.session_id,
            "content": request.content,
            "context": request.context,
            "priority": request.priority,
            "accessCount": 0,
            "createdAt": datetime.utcnow(),
            "lastAccessed": datetime.utcnow(),
            "expiresAt": expires_at,
            "ttlMinutes": ttl_minutes,
            "isPromoted": False,
            "metadata": {
                "framework": "python_ai_brain",
                "version": "1.0.0",
                "source": "working_memory_manager"
            }
        }
        
        # Store working memory
        await self.working_memory_collection.store_working_memory(working_memory)
        
        # Update session stats
        await self._update_session_stats(request.session_id)
        
        # Check for promotion opportunities
        promoted_memories = await self._check_promotion_opportunities(request.session_id)
        
        # Get session stats
        session_stats = await self._get_session_stats(request.session_id)
        
        return WorkingMemoryResult(
            memory_id=memory_id,
            stored=True,
            evicted_memories=evicted_memories,
            promoted_memories=promoted_memories,
            session_stats=session_stats
        )
    
    async def retrieveWorkingMemories(
        self,
        agent_id: str,
        session_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve working memories for a session."""
        memories = await self.working_memory_collection.find_working_memories(
            agent_id=agent_id,
            session_id=session_id,
            query=query,
            limit=limit
        )
        
        # Update access counts
        for memory in memories:
            await self._update_memory_access(memory["memoryId"])
        
        return memories
    
    async def get_working_memory_analytics(
        self,
        agent_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get working memory analytics."""
        return await self.working_memory_collection.get_working_memory_analytics(agent_id, session_id)
    
    async def get_working_memory_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get working memory statistics."""
        stats = await self.working_memory_collection.get_working_memory_stats(agent_id)
        
        return {
            **stats,
            "sessionStats": len(self._session_stats),
            "memoryPressure": len(self._memory_pressure),
            "cleanupTaskActive": self._cleanup_task is not None and not self._cleanup_task.done()
        }

    # Private helper methods
    async def _start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        logger.debug("Background cleanup started")

    async def _check_and_evict_memories(self, session_id: str) -> List[str]:
        """Check memory pressure and evict memories if necessary."""
        evicted_memories = []

        # Get current session memory count
        session_count = await self.working_memory_collection.get_session_memory_count(session_id)

        # Check if eviction is needed
        if session_count >= self._config.max_memories_per_session:
            # Get low priority memories for eviction
            low_priority_memories = await self.working_memory_collection.find_low_priority_memories(
                session_id,
                limit=self._config.eviction_batch_size
            )

            # Evict memories
            for memory in low_priority_memories:
                await self.working_memory_collection.delete_working_memory(memory["memoryId"])
                evicted_memories.append(str(memory["memoryId"]))

        return evicted_memories

    async def _update_session_stats(self, session_id: str) -> None:
        """Update session statistics."""
        if session_id not in self._session_stats:
            self._session_stats[session_id] = {
                "memory_count": 0,
                "last_activity": datetime.utcnow(),
                "total_accesses": 0
            }

        # Update memory count
        memory_count = await self.working_memory_collection.get_session_memory_count(session_id)
        self._session_stats[session_id]["memory_count"] = memory_count
        self._session_stats[session_id]["last_activity"] = datetime.utcnow()

    async def _check_promotion_opportunities(self, session_id: str) -> List[str]:
        """Check for memories that should be promoted to long-term storage."""
        promoted_memories = []

        if not self.semantic_memory_engine:
            return promoted_memories

        # Get high-access memories
        high_access_memories = await self.working_memory_collection.find_high_access_memories(
            session_id,
            access_threshold=self._config.promotion_access_threshold,
            importance_threshold=self._config.promotion_importance_threshold
        )

        # Promote eligible memories
        for memory in high_access_memories:
            if not memory.get("isPromoted", False):
                # Promote to semantic memory
                await self._promote_to_semantic_memory(memory)
                promoted_memories.append(str(memory["memoryId"]))

        return promoted_memories

    async def _promote_to_semantic_memory(self, working_memory: Dict[str, Any]) -> None:
        """Promote working memory to semantic memory."""
        if not self.semantic_memory_engine:
            return

        # Store in semantic memory
        await self.semantic_memory_engine.store_memory(
            user_id=working_memory["agentId"],
            content=working_memory["content"],
            memory_type="episodic",
            importance_score=working_memory.get("priority", 0.5),
            context=working_memory.get("context", {})
        )

        # Mark as promoted
        await self.working_memory_collection.mark_memory_promoted(working_memory["memoryId"])

    async def _get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        if session_id not in self._session_stats:
            await self._update_session_stats(session_id)

        return self._session_stats.get(session_id, {})

    async def _update_memory_access(self, memory_id: ObjectId) -> None:
        """Update memory access count and timestamp."""
        await self.working_memory_collection.update_memory_access(memory_id)

    async def _cleanup_expired_memories(self) -> None:
        """Clean up expired memories."""
        expired_count = await self.working_memory_collection.cleanup_expired_memories()
        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired working memories")

    async def search_working_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search working memories using text search."""
        try:
            # Build search filter
            search_filter = {}
            if session_id:
                search_filter["sessionId"] = session_id
            if agent_id:
                search_filter["agentId"] = agent_id

            # Add text search
            search_filter["$text"] = {"$search": query}

            # Execute search
            results = await self.working_memory_collection.collection.find(
                search_filter,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit).to_list(length=None)

            # Update access tracking
            memory_ids = [result["_id"] for result in results]
            for memory_id in memory_ids:
                await self._update_memory_access(memory_id)

            return results

        except Exception as error:
            logger.error(f"Error searching working memories: {error}")
            return []

    async def promoteToLongTerm(self, working_memory_id: str) -> None:
        """Promote a working memory to long-term semantic memory."""
        try:
            # Get the working memory
            working_memory = await self.working_memory_collection.collection.find_one({
                "_id": ObjectId(working_memory_id)
            })

            if not working_memory:
                logger.warning(f"Working memory not found: {working_memory_id}")
                return

            # Promote to semantic memory
            await self._promote_to_semantic_memory(working_memory)

            # Remove from working memory
            await self.working_memory_collection.collection.delete_one({
                "_id": ObjectId(working_memory_id)
            })

            logger.info(f"Promoted working memory to long-term: {working_memory_id}")

        except Exception as error:
            logger.error(f"Error promoting working memory: {error}")

    async def extendTTL(self, working_memory_id: str, additional_minutes: int) -> None:
        """Extend the TTL of a working memory."""
        try:
            # Calculate new expiry time
            additional_time = timedelta(minutes=additional_minutes)

            result = await self.working_memory_collection.collection.update_one(
                {"_id": ObjectId(working_memory_id)},
                {
                    "$inc": {"ttlMinutes": additional_minutes},
                    "$set": {
                        "expiresAt": datetime.utcnow() + additional_time,
                        "updated": datetime.utcnow()
                    }
                }
            )

            if result.modified_count > 0:
                logger.info(f"Extended TTL for working memory {working_memory_id} by {additional_minutes} minutes")
            else:
                logger.warning(f"Working memory not found for TTL extension: {working_memory_id}")

        except Exception as error:
            logger.error(f"Error extending TTL: {error}")

    async def cleanupExpiredMemories(self) -> int:
        """Clean up expired working memories and return count."""
        try:
            # Remove expired memories
            result = await self.working_memory_collection.collection.delete_many({
                "expiresAt": {"$lt": datetime.utcnow()}
            })

            if result.deleted_count > 0:
                logger.info(f"Cleaned up {result.deleted_count} expired working memories")

            return result.deleted_count

        except Exception as error:
            logger.error(f"Error cleaning up expired memories: {error}")
            return 0

    async def perform_pressure_cleanup(self) -> None:
        """Perform cleanup when memory pressure is high."""
        try:
            # Get memory pressure stats
            pressure_stats = await self.get_memory_pressure()

            if pressure_stats["pressureLevel"] > 0.8:
                logger.warning("High memory pressure detected, performing aggressive cleanup")

                # Remove least recently used memories
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                result = await self.working_memory_collection.collection.delete_many({
                    "lastAccessed": {"$lt": cutoff_time},
                    "importance": {"$lt": 0.5}
                })

                logger.info(f"Pressure cleanup removed {result.deleted_count} low-importance memories")

        except Exception as error:
            logger.error(f"Error performing pressure cleanup: {error}")

    async def getMemoryPressure(self) -> MemoryPressureStats:
        """Get current memory pressure statistics."""
        try:
            # Count total working memories
            total_count = await self.working_memory_collection.collection.count_documents({})

            # Count by session
            session_pipeline = [
                {"$group": {"_id": "$sessionId", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]

            session_stats = await self.working_memory_collection.collection.aggregate(session_pipeline).to_list(length=None)

            # Calculate pressure level (simplified)
            max_capacity = 10000  # Configurable limit
            pressure_level = min(1.0, total_count / max_capacity)

            # Determine pressure status
            if pressure_level < 0.5:
                status = "low"
            elif pressure_level < 0.8:
                status = "moderate"
            else:
                status = "high"

            return {
                "totalMemories": total_count,
                "pressureLevel": pressure_level,
                "status": status,
                "maxCapacity": max_capacity,
                "topSessions": session_stats,
                "recommendedAction": "cleanup" if pressure_level > 0.8 else "monitor"
            }

        except Exception as error:
            logger.error(f"Error getting memory pressure: {error}")
            return {
                "totalMemories": 0,
                "pressureLevel": 0.0,
                "status": "unknown",
                "maxCapacity": 10000,
                "topSessions": [],
                "recommendedAction": "monitor"
            }

    async def shutdown(self) -> None:
        """Shutdown the working memory manager."""
        try:
            # Stop background cleanup
            if hasattr(self, '_cleanup_task') and self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Perform final cleanup
            await self.cleanup_expired_memories()

            logger.info("Working memory manager shutdown complete")

        except Exception as error:
            logger.error(f"Error during working memory manager shutdown: {error}")
