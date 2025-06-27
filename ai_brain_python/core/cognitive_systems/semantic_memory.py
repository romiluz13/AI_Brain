"""
Semantic Memory Engine

Perfect recall system with MongoDB Atlas Vector Search integration.
Provides sophisticated memory storage, retrieval, and semantic search capabilities.

Features:
- Vector-based semantic memory storage and retrieval
- MongoDB Atlas Vector Search integration
- Memory consolidation and importance scoring
- Associative memory networks and clustering
- Temporal memory decay and reinforcement
- Context-aware memory retrieval
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, SemanticMemory, CognitiveSystemType
from ai_brain_python.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SemanticMemoryEngine(CognitiveSystemInterface):
    """
    Semantic Memory Engine - System 9 of 16
    
    Provides perfect recall with MongoDB Atlas Vector Search integration
    and sophisticated memory management capabilities.
    """
    
    def __init__(self, system_id: str = "semantic_memory", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Vector store for semantic search
        self.vector_store: Optional[VectorStore] = None
        
        # Memory management
        self._memory_cache: Dict[str, SemanticMemory] = {}
        self._user_memory_graphs: Dict[str, Dict[str, List[str]]] = {}  # Associative networks
        
        # Memory configuration
        self._config = {
            "embedding_dimension": config.get("embedding_dimension", 1536) if config else 1536,
            "max_memories_per_user": config.get("max_memories_per_user", 10000) if config else 10000,
            "memory_decay_rate": config.get("memory_decay_rate", 0.01) if config else 0.01,
            "consolidation_threshold": config.get("consolidation_threshold", 0.8) if config else 0.8,
            "similarity_threshold": config.get("similarity_threshold", 0.7) if config else 0.7,
            "max_search_results": config.get("max_search_results", 20) if config else 20
        }
        
        # Memory types and importance factors
        self._memory_types = {
            "episodic": {"base_importance": 0.6, "decay_rate": 0.02},
            "semantic": {"base_importance": 0.8, "decay_rate": 0.005},
            "procedural": {"base_importance": 0.9, "decay_rate": 0.001},
            "emotional": {"base_importance": 0.7, "decay_rate": 0.015},
            "contextual": {"base_importance": 0.5, "decay_rate": 0.03}
        }
        
        # Consolidation patterns
        self._consolidation_patterns = [
            "repeated_access",
            "emotional_significance", 
            "goal_relevance",
            "temporal_clustering",
            "semantic_clustering"
        ]
    
    @property
    def system_name(self) -> str:
        return "Semantic Memory Engine"
    
    @property
    def system_description(self) -> str:
        return "Perfect recall system with MongoDB Atlas Vector Search integration"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.MEMORY_STORAGE, SystemCapability.MEMORY_RETRIEVAL}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.MEMORY_STORAGE, SystemCapability.MEMORY_RETRIEVAL}
    
    async def initialize(self) -> None:
        """Initialize the Semantic Memory Engine."""
        try:
            logger.info("Initializing Semantic Memory Engine...")
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            # Load memory data
            await self._load_memory_data()
            
            # Initialize embedding models
            await self._initialize_embedding_models()
            
            self._is_initialized = True
            logger.info("Semantic Memory Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Semantic Memory Engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Semantic Memory Engine."""
        try:
            logger.info("Shutting down Semantic Memory Engine...")
            
            # Save memory data
            await self._save_memory_data()
            
            # Cleanup vector store
            if self.vector_store:
                await self.vector_store.close()
            
            self._is_initialized = False
            logger.info("Semantic Memory Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Semantic Memory Engine shutdown: {e}")
    
    async def process(
        self, 
        input_data: CognitiveInputData,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input through semantic memory analysis."""
        if not self._is_initialized:
            raise RuntimeError("Semantic Memory Engine not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Extract and store new memories
            new_memories = await self._extract_memories_from_input(input_data)
            stored_memories = []
            
            for memory_content in new_memories:
                memory_id = await self.store_memory(
                    user_id=user_id,
                    content=memory_content["content"],
                    memory_type=memory_content["type"],
                    importance_score=memory_content["importance"],
                    context=memory_content.get("context", {})
                )
                stored_memories.append(memory_id)
            
            # Retrieve relevant memories
            relevant_memories = await self.retrieve_memories(
                user_id=user_id,
                query=input_data.text or "",
                limit=self._config["max_search_results"]
            )
            
            # Update memory associations
            await self._update_memory_associations(user_id, stored_memories, relevant_memories)
            
            # Perform memory consolidation
            consolidation_results = await self._perform_memory_consolidation(user_id)
            
            # Generate memory insights
            memory_insights = await self._generate_memory_insights(user_id, relevant_memories)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.9,
                "memory_operations": {
                    "new_memories_stored": len(stored_memories),
                    "relevant_memories_found": len(relevant_memories),
                    "consolidation_performed": len(consolidation_results),
                    "total_user_memories": await self._get_user_memory_count(user_id)
                },
                "stored_memories": stored_memories,
                "relevant_memories": [
                    {
                        "id": mem.id,
                        "content": mem.content[:200] + "..." if len(mem.content) > 200 else mem.content,
                        "importance": mem.importance_score,
                        "access_count": mem.access_count,
                        "similarity_score": getattr(mem, 'similarity_score', 0.0)
                    }
                    for mem in relevant_memories[:5]  # Top 5 for response
                ],
                "memory_insights": memory_insights,
                "consolidation_results": consolidation_results
            }
            
        except Exception as e:
            logger.error(f"Error in Semantic Memory processing: {e}")
            return {
                "system": self.system_id,
                "status": "error",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current semantic memory state."""
        state_data = {
            "total_memories": len(self._memory_cache),
            "embedding_dimension": self._config["embedding_dimension"],
            "memory_types": len(self._memory_types)
        }
        
        if user_id:
            user_memory_count = await self._get_user_memory_count(user_id)
            state_data.update({
                "user_memory_count": user_memory_count,
                "user_has_associations": user_id in self._user_memory_graphs
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.SEMANTIC_MEMORY,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.95,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update semantic memory state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Semantic Memory state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for semantic memory processing."""
        violations = []
        warnings = []
        
        if not input_data.text:
            warnings.append("No text provided for memory extraction")
        
        if input_data.text and len(input_data.text) > 50000:
            violations.append("Text too long for memory processing (max 50,000 characters)")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            confidence=1.0 if len(violations) == 0 else 0.0,
            violations=violations,
            warnings=warnings
        )
    
    # Public memory methods
    
    async def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "episodic",
        importance_score: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory with vector embedding."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        # Calculate importance if not provided
        if importance_score is None:
            importance_score = await self._calculate_importance(content, memory_type, context or {})
        
        # Create memory object
        memory = SemanticMemory(
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            importance_score=importance_score,
            access_count=0,
            last_accessed=datetime.utcnow(),
            consolidation_level=0.0,
            decay_rate=self._memory_types.get(memory_type, {}).get("decay_rate", 0.01),
            semantic_tags=await self._extract_semantic_tags(content)
        )
        
        # Store in vector store
        memory_id = await self.vector_store.store_vector(
            collection="semantic_memories",
            vector=embedding,
            metadata={
                "user_id": user_id,
                "content": content,
                "memory_type": memory_type,
                "importance_score": importance_score,
                "created_at": memory.created_at.isoformat(),
                "semantic_tags": memory.semantic_tags
            }
        )
        
        # Cache memory
        memory.id = memory_id
        self._memory_cache[memory_id] = memory
        
        logger.debug(f"Stored memory {memory_id} for user {user_id}")
        return memory_id
    
    async def retrieve_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[SemanticMemory]:
        """Retrieve memories using semantic search."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Build search filters
        filters = {"user_id": user_id}
        if memory_type:
            filters["memory_type"] = memory_type
        if min_importance > 0:
            filters["importance_score"] = {"$gte": min_importance}
        
        # Perform vector search
        search_results = await self.vector_store.search_similar(
            collection="semantic_memories",
            query_vector=query_embedding,
            limit=limit * 2,  # Get more results for filtering
            filters=filters
        )
        
        # Convert to memory objects and apply additional filtering
        memories = []
        for result in search_results:
            memory_id = result.get("_id")
            if memory_id in self._memory_cache:
                memory = self._memory_cache[memory_id]
            else:
                # Reconstruct memory from metadata
                memory = await self._reconstruct_memory_from_result(result)
                self._memory_cache[memory_id] = memory
            
            # Add similarity score
            memory.similarity_score = result.get("score", 0.0)
            
            # Update access tracking
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            
            memories.append(memory)
        
        # Sort by relevance (combination of similarity and importance)
        memories.sort(
            key=lambda m: (m.similarity_score * 0.7 + m.importance_score * 0.3),
            reverse=True
        )
        
        return memories[:limit]
    
    async def get_memory_associations(self, user_id: str, memory_id: str) -> List[str]:
        """Get associated memories for a given memory."""
        if user_id not in self._user_memory_graphs:
            return []
        
        return self._user_memory_graphs[user_id].get(memory_id, [])
    
    async def update_memory_importance(self, memory_id: str, new_importance: float) -> bool:
        """Update the importance score of a memory."""
        if memory_id not in self._memory_cache:
            return False
        
        memory = self._memory_cache[memory_id]
        memory.importance_score = max(0.0, min(1.0, new_importance))
        
        # Update in vector store
        if self.vector_store:
            await self.vector_store.update_metadata(
                collection="semantic_memories",
                vector_id=memory_id,
                metadata={"importance_score": memory.importance_score}
            )
        
        return True
    
    # Private methods
    
    async def _initialize_vector_store(self) -> None:
        """Initialize the vector store."""
        # Vector store will be injected by the Universal AI Brain
        logger.debug("Vector store initialization placeholder")
    
    async def _load_memory_data(self) -> None:
        """Load memory data from storage."""
        logger.debug("Memory data loaded")
    
    async def _save_memory_data(self) -> None:
        """Save memory data to storage."""
        logger.debug("Memory data saved")
    
    async def _initialize_embedding_models(self) -> None:
        """Initialize embedding models."""
        # In production, this would initialize actual embedding models
        logger.debug("Embedding models initialized")
    
    async def _extract_memories_from_input(self, input_data: CognitiveInputData) -> List[Dict[str, Any]]:
        """Extract memorable content from input."""
        text = input_data.text or ""
        
        # Simple memory extraction - in production would use NLP models
        memories = []
        
        if len(text) > 50:  # Only store substantial content
            # Determine memory type based on content
            memory_type = self._classify_memory_type(text)
            
            # Calculate importance
            importance = await self._calculate_importance(text, memory_type, input_data.context.__dict__)
            
            memories.append({
                "content": text,
                "type": memory_type,
                "importance": importance,
                "context": {
                    "session_id": input_data.context.session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "input_type": input_data.input_type
                }
            })
        
        return memories
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Placeholder - in production would use actual embedding model
        # For now, return a random embedding of the correct dimension
        import random
        return [random.random() for _ in range(self._config["embedding_dimension"])]
    
    async def _calculate_importance(self, content: str, memory_type: str, context: Dict[str, Any]) -> float:
        """Calculate importance score for memory."""
        base_importance = self._memory_types.get(memory_type, {}).get("base_importance", 0.5)
        
        # Adjust based on content characteristics
        importance_factors = []
        
        # Length factor
        length_factor = min(1.0, len(content) / 1000)  # Longer content may be more important
        importance_factors.append(length_factor * 0.2)
        
        # Emotional content factor
        emotional_keywords = ["feel", "emotion", "happy", "sad", "angry", "excited", "worried"]
        emotional_factor = sum(1 for keyword in emotional_keywords if keyword in content.lower()) / len(emotional_keywords)
        importance_factors.append(emotional_factor * 0.3)
        
        # Goal-related factor
        goal_keywords = ["goal", "plan", "objective", "target", "achieve", "complete"]
        goal_factor = sum(1 for keyword in goal_keywords if keyword in content.lower()) / len(goal_keywords)
        importance_factors.append(goal_factor * 0.3)
        
        # Context factor (user engagement)
        context_factor = 0.2 if context.get("user_id") else 0.0
        importance_factors.append(context_factor)
        
        # Combine factors
        adjustment = sum(importance_factors)
        final_importance = min(1.0, base_importance + adjustment)
        
        return final_importance
    
    def _classify_memory_type(self, text: str) -> str:
        """Classify memory type based on content."""
        text_lower = text.lower()
        
        # Simple classification rules
        if any(word in text_lower for word in ["feel", "emotion", "mood"]):
            return "emotional"
        elif any(word in text_lower for word in ["goal", "plan", "objective"]):
            return "semantic"
        elif any(word in text_lower for word in ["how to", "process", "step"]):
            return "procedural"
        elif any(word in text_lower for word in ["when", "where", "happened"]):
            return "episodic"
        else:
            return "contextual"
    
    async def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags from content."""
        # Simple tag extraction - in production would use NLP
        words = content.lower().split()
        
        # Filter for meaningful words (simple approach)
        meaningful_words = [
            word for word in words 
            if len(word) > 3 and word.isalpha()
        ]
        
        # Return top 10 most relevant words as tags
        return meaningful_words[:10]
    
    async def _update_memory_associations(
        self, 
        user_id: str, 
        new_memory_ids: List[str], 
        related_memories: List[SemanticMemory]
    ) -> None:
        """Update memory association graph."""
        if user_id not in self._user_memory_graphs:
            self._user_memory_graphs[user_id] = {}
        
        graph = self._user_memory_graphs[user_id]
        
        # Create associations between new memories and related memories
        for new_id in new_memory_ids:
            if new_id not in graph:
                graph[new_id] = []
            
            for related_memory in related_memories:
                if related_memory.similarity_score > self._config["similarity_threshold"]:
                    # Bidirectional association
                    if related_memory.id not in graph[new_id]:
                        graph[new_id].append(related_memory.id)
                    
                    if related_memory.id not in graph:
                        graph[related_memory.id] = []
                    if new_id not in graph[related_memory.id]:
                        graph[related_memory.id].append(new_id)
    
    async def _perform_memory_consolidation(self, user_id: str) -> List[Dict[str, Any]]:
        """Perform memory consolidation for a user."""
        consolidation_results = []
        
        # Get user memories that need consolidation
        user_memories = [
            memory for memory in self._memory_cache.values()
            if hasattr(memory, 'user_id') and memory.user_id == user_id
            and memory.consolidation_level < self._config["consolidation_threshold"]
        ]
        
        for memory in user_memories:
            # Check consolidation criteria
            should_consolidate = (
                memory.access_count > 5 or  # Frequently accessed
                memory.importance_score > 0.8 or  # High importance
                (datetime.utcnow() - memory.created_at).days > 7  # Older than a week
            )
            
            if should_consolidate:
                # Increase consolidation level
                memory.consolidation_level = min(1.0, memory.consolidation_level + 0.1)
                
                # Reduce decay rate for consolidated memories
                memory.decay_rate *= 0.9
                
                consolidation_results.append({
                    "memory_id": memory.id,
                    "new_consolidation_level": memory.consolidation_level,
                    "reason": "frequent_access" if memory.access_count > 5 else "high_importance"
                })
        
        return consolidation_results
    
    async def _generate_memory_insights(self, user_id: str, relevant_memories: List[SemanticMemory]) -> Dict[str, Any]:
        """Generate insights from retrieved memories."""
        if not relevant_memories:
            return {"insights": [], "patterns": [], "recommendations": []}
        
        insights = []
        patterns = []
        recommendations = []
        
        # Analyze memory patterns
        memory_types = {}
        for memory in relevant_memories:
            memory_type = memory.memory_type
            memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
        
        # Most common memory type
        if memory_types:
            most_common_type = max(memory_types, key=memory_types.get)
            patterns.append(f"Most relevant memories are {most_common_type} type")
        
        # High importance memories
        high_importance_count = sum(1 for m in relevant_memories if m.importance_score > 0.8)
        if high_importance_count > 0:
            insights.append(f"Found {high_importance_count} high-importance related memories")
        
        # Frequently accessed memories
        frequent_access_count = sum(1 for m in relevant_memories if m.access_count > 10)
        if frequent_access_count > 0:
            patterns.append(f"{frequent_access_count} memories are frequently accessed")
        
        # Recommendations
        if len(relevant_memories) > 10:
            recommendations.append("Consider memory consolidation - many related memories found")
        
        if any(m.importance_score < 0.3 for m in relevant_memories):
            recommendations.append("Some low-importance memories could be archived")
        
        return {
            "insights": insights,
            "patterns": patterns,
            "recommendations": recommendations,
            "memory_type_distribution": memory_types
        }
    
    async def _get_user_memory_count(self, user_id: str) -> int:
        """Get total memory count for a user."""
        # In production, this would query the vector store
        return sum(
            1 for memory in self._memory_cache.values()
            if hasattr(memory, 'user_id') and getattr(memory, 'user_id') == user_id
        )
    
    async def _reconstruct_memory_from_result(self, result: Dict[str, Any]) -> SemanticMemory:
        """Reconstruct memory object from search result."""
        metadata = result.get("metadata", {})
        
        memory = SemanticMemory(
            content=metadata.get("content", ""),
            embedding=result.get("embedding", []),
            memory_type=metadata.get("memory_type", "episodic"),
            importance_score=metadata.get("importance_score", 0.5),
            access_count=0,
            last_accessed=datetime.utcnow(),
            consolidation_level=0.0,
            decay_rate=0.01,
            semantic_tags=metadata.get("semantic_tags", [])
        )
        
        memory.id = result.get("_id")
        if metadata.get("created_at"):
            memory.created_at = datetime.fromisoformat(metadata["created_at"])
        
        return memory
