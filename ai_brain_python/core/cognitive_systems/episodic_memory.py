"""
EpisodicMemoryEngine - Advanced episodic memory system using MongoDB Atlas rich document storage

Exact Python equivalent of JavaScript EpisodicMemoryEngine.ts with:
- Rich BSON document storage (Atlas optimized)
- Nested documents and arrays for complex experiences
- Advanced querying capabilities (Atlas enhanced)
- Complex data modeling for episodic memories

CRITICAL: This uses MongoDB Atlas EXCLUSIVE features:
- Rich BSON document storage (Atlas optimized)
- Nested documents and arrays for complex experiences
- Advanced querying capabilities (Atlas enhanced)
- Complex data modeling for episodic memories

Features:
- Rich document storage for complex episodic memories
- Contextual memory retrieval and organization
- Temporal and spatial memory indexing
- Emotional and social memory patterns
- Learning and insight extraction from experiences
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.interfaces.cognitive_system import CognitiveSystemInterface
from ..core.models import CognitiveInputData, CognitiveResponse
from ..storage.collections.episodic_memory_collection import EpisodicMemoryCollection
from ..utils.logger import logger


@dataclass
class MemoryStorageRequest:
    """Memory storage request data structure."""
    agent_id: str
    experience: Dict[str, Any]
    processing: Dict[str, Any]
    learning: Optional[Dict[str, Any]] = None


@dataclass
class MemoryRetrievalRequest:
    """Memory retrieval request data structure."""
    agent_id: str
    query: Dict[str, Any]
    constraints: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class MemoryAnalysisRequest:
    """Memory analysis request data structure."""
    agent_id: str
    analysis_type: str
    parameters: Dict[str, Any]


@dataclass
class RetrievedMemory:
    """Retrieved memory data structure."""
    memory: Dict[str, Any]
    relevance_score: float
    retrieval_reason: str
    contextual_fit: float


@dataclass
class MemoryRetrievalResult:
    """Memory retrieval result data structure."""
    memories: List[RetrievedMemory]
    patterns: Dict[str, Any]
    insights: List[str]
    metadata: Dict[str, Any]


class EpisodicMemoryEngine(CognitiveSystemInterface):
    """
    EpisodicMemoryEngine - Advanced episodic memory system using MongoDB Atlas rich document storage
    
    Exact Python equivalent of JavaScript EpisodicMemoryEngine with:
    - Rich BSON document storage (Atlas optimized)
    - Nested documents and arrays for complex experiences
    - Advanced querying capabilities (Atlas enhanced)
    - Complex data modeling for episodic memories
    
    CRITICAL: Optimized for MongoDB Atlas (not local MongoDB)
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)
        self.db = db
        self.episodic_collection = EpisodicMemoryCollection(db)
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the episodic memory engine."""
        if self.is_initialized:
            return
            
        try:
            await self.episodic_collection.initialize_indexes()
            self.is_initialized = True
            logger.info("✅ EpisodicMemoryEngine initialized successfully")
            logger.info("📝 Note: Optimized for MongoDB Atlas rich document storage")
        except Exception as error:
            logger.error(f"❌ Error initializing EpisodicMemoryEngine: {error}")
            raise error
    
    async def process(self, input_data: CognitiveInputData) -> CognitiveResponse:
        """Process episodic memory requests."""
        try:
            await self.initialize()
            
            # Extract episodic memory request from input
            request_data = input_data.additional_context.get("episodic_request", {})
            
            if not request_data:
                return CognitiveResponse(
                    response_text="No episodic memory request provided",
                    confidence=0.0,
                    processing_metadata={
                        "system": "episodic_memory",
                        "error": "Missing episodic memory request"
                    }
                )
            
            action = request_data.get("action", "")
            
            if action == "store_memory":
                result = await self.store_memory(MemoryStorageRequest(
                    agent_id=request_data.get("agentId", ""),
                    experience=request_data.get("experience", {}),
                    processing=request_data.get("processing", {}),
                    learning=request_data.get("learning")
                ))
                response_text = f"Memory stored with ID: {result['memoryId']}"
                
            elif action == "retrieve_memories":
                result = await self.retrieve_memories(MemoryRetrievalRequest(
                    agent_id=request_data.get("agentId", ""),
                    query=request_data.get("query", {}),
                    constraints=request_data.get("constraints", {}),
                    context=request_data.get("context")
                ))
                response_text = f"Retrieved {len(result.memories)} memories"
                
            elif action == "analyze_memories":
                result = await self.analyze_memories(MemoryAnalysisRequest(
                    agent_id=request_data.get("agentId", ""),
                    analysis_type=request_data.get("analysisType", ""),
                    parameters=request_data.get("parameters", {})
                ))
                response_text = f"Memory analysis completed with {len(result['insights'])} insights"
                
            else:
                response_text = f"Unknown episodic memory action: {action}"
            
            return CognitiveResponse(
                response_text=response_text,
                confidence=0.9,
                processing_metadata={
                    "system": "episodic_memory",
                    "action": action,
                    "atlas_optimized": True
                }
            )
            
        except Exception as error:
            logger.error(f"Error in EpisodicMemoryEngine.process: {error}")
            return CognitiveResponse(
                response_text=f"Episodic memory error: {str(error)}",
                confidence=0.0,
                processing_metadata={
                    "system": "episodic_memory",
                    "error": str(error)
                }
            )
    
    async def store_memory(self, request: MemoryStorageRequest) -> Dict[str, Any]:
        """Store an episodic memory using Atlas rich document storage."""
        if not self.is_initialized:
            raise ValueError("EpisodicMemoryEngine not initialized")
        
        try:
            # Create memory ID
            memory_id = str(ObjectId())
            
            # Categorize experience
            experience_type = self._categorize_experience(request.experience)
            category = self._determine_category(request.experience)
            
            # Calculate relative time
            start_time = request.experience.get("temporal", {}).get("startTime")
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            elif not isinstance(start_time, datetime):
                start_time = datetime.utcnow()
            
            relative_time = self._calculate_relative_time(start_time)
            
            # Create rich episodic memory document
            memory_document = {
                "memoryId": memory_id,
                "agentId": request.agent_id,
                "timestamp": datetime.utcnow(),
                "episode": {
                    "experience": request.experience,
                    "type": experience_type,
                    "category": category,
                    "importance": request.processing.get("importance", 0.5),
                    "vividness": request.processing.get("vividness", 0.5),
                    "confidence": request.processing.get("confidence", 0.5),
                    "encodingStrategy": request.processing.get("encodingStrategy", "default"),
                    "relativeTime": relative_time,
                    "psychology": {
                        "emotions": request.experience.get("emotions", []),
                        "cognitiveLoad": 0.5,  # Default cognitive load
                        "attention": 0.7,      # Default attention level
                        "arousal": sum(e.get("arousal", 0) for e in request.experience.get("emotions", [])) / max(len(request.experience.get("emotions", [])), 1)
                    },
                    "social": {
                        "participants": request.experience.get("participants", []),
                        "relationships": [p.get("relationship", "unknown") for p in request.experience.get("participants", [])],
                        "socialContext": request.experience.get("context", {}).get("social", "individual")
                    },
                    "learning": request.learning or {
                        "knowledge": [],
                        "skills": [],
                        "insights": []
                    }
                },
                "retrieval": {
                    "accessCount": 0,
                    "lastAccessed": None,
                    "retrievalContexts": [],
                    "strengthening": 1.0
                },
                "connections": {
                    "related": [],
                    "causal": [],
                    "temporal": [],
                    "thematic": []
                },
                "metadata": {
                    "framework": "python_ai_brain",
                    "version": "1.0.0",
                    "atlasOptimized": True,
                    "richDocument": True
                }
            }
            
            # Store the memory
            await self.episodic_collection.create_memory(memory_document)
            
            # Find connections to existing memories
            connections = await self._find_memory_connections(request.agent_id, memory_document)
            
            # Update memory with connections
            if connections:
                await self.episodic_collection.collection.update_one(
                    {"memoryId": memory_id},
                    {"$set": {"connections.related": connections}}
                )
            
            # Generate processing insights
            processing_insights = self._generate_processing_insights(memory_document, connections)
            
            logger.info(f"Stored episodic memory: {memory_id}")
            
            return {
                "memoryId": memory_id,
                "processingInsights": processing_insights,
                "connections": connections
            }
            
        except Exception as error:
            logger.error(f"Error storing episodic memory: {error}")
            raise error

    async def retrieve_memories(self, request: MemoryRetrievalRequest) -> MemoryRetrievalResult:
        """Retrieve memories using contextual search with Atlas rich document queries."""
        if not self.is_initialized:
            raise ValueError("EpisodicMemoryEngine not initialized")

        start_time = datetime.utcnow()

        try:
            # Build query based on request type
            query_filter = {"agentId": request.agent_id}

            # Add constraints
            if request.constraints.get("timeRange"):
                time_range = request.constraints["timeRange"]
                query_filter["episode.experience.temporal.startTime"] = {
                    "$gte": time_range.get("start"),
                    "$lte": time_range.get("end")
                }

            if request.constraints.get("minImportance"):
                query_filter["episode.importance"] = {"$gte": request.constraints["minImportance"]}

            # Query type specific filters
            query_type = request.query.get("type", "free_text")
            parameters = request.query.get("parameters", {})

            if query_type == "temporal":
                if parameters.get("timeOfDay"):
                    query_filter["episode.experience.temporal.timeOfDay"] = parameters["timeOfDay"]
                if parameters.get("dayOfWeek"):
                    query_filter["episode.experience.temporal.dayOfWeek"] = parameters["dayOfWeek"]

            elif query_type == "spatial":
                if parameters.get("location"):
                    query_filter["episode.experience.spatial.location"] = {"$regex": parameters["location"], "$options": "i"}
                if parameters.get("environment"):
                    query_filter["episode.experience.spatial.environment"] = parameters["environment"]

            elif query_type == "social":
                if parameters.get("participants"):
                    query_filter["episode.experience.participants.name"] = {"$in": parameters["participants"]}
                if parameters.get("relationship"):
                    query_filter["episode.experience.participants.relationship"] = parameters["relationship"]

            elif query_type == "emotional":
                if parameters.get("emotions"):
                    query_filter["episode.experience.emotions.emotion"] = {"$in": parameters["emotions"]}
                if parameters.get("valence"):
                    query_filter["episode.experience.emotions.valence"] = {"$gte": parameters["valence"]}

            elif query_type == "thematic":
                if parameters.get("theme"):
                    query_filter["$or"] = [
                        {"episode.experience.event.name": {"$regex": parameters["theme"], "$options": "i"}},
                        {"episode.experience.event.description": {"$regex": parameters["theme"], "$options": "i"}},
                        {"episode.category": parameters["theme"]}
                    ]

            elif query_type == "free_text":
                if parameters.get("text"):
                    query_filter["$text"] = {"$search": parameters["text"]}

            # Execute query
            max_results = request.constraints.get("maxResults", 10)
            memories = await self.episodic_collection.collection.find(query_filter).limit(max_results).to_list(length=None)

            # Process retrieved memories
            retrieved_memories = []
            for memory in memories:
                relevance_score = self._calculate_relevance_score(memory, request)
                retrieval_reason = self._determine_retrieval_reason(memory, request)
                contextual_fit = self._calculate_contextual_fit(memory, request.context or {})

                retrieved_memories.append(RetrievedMemory(
                    memory=memory,
                    relevance_score=relevance_score,
                    retrieval_reason=retrieval_reason,
                    contextual_fit=contextual_fit
                ))

                # Update retrieval statistics
                await self.episodic_collection.collection.update_one(
                    {"_id": memory["_id"]},
                    {
                        "$inc": {"retrieval.accessCount": 1},
                        "$set": {"retrieval.lastAccessed": datetime.utcnow()},
                        "$push": {"retrieval.retrievalContexts": {
                            "timestamp": datetime.utcnow(),
                            "queryType": query_type,
                            "relevanceScore": relevance_score
                        }}
                    }
                )

            # Sort by relevance
            retrieved_memories.sort(key=lambda x: x.relevance_score, reverse=True)

            # Detect patterns
            patterns = await self._detect_memory_patterns([rm.memory for rm in retrieved_memories])

            # Generate insights
            insights = self._generate_retrieval_insights(retrieved_memories, patterns)

            # Calculate metadata
            end_time = datetime.utcnow()
            retrieval_time = (end_time - start_time).total_seconds() * 1000

            metadata = {
                "retrievalTime": retrieval_time,
                "totalMemoriesSearched": len(memories),
                "queryType": query_type,
                "atlasRichDocuments": True,
                "averageRelevance": sum(rm.relevance_score for rm in retrieved_memories) / len(retrieved_memories) if retrieved_memories else 0
            }

            return MemoryRetrievalResult(
                memories=retrieved_memories,
                patterns=patterns,
                insights=insights,
                metadata=metadata
            )

        except Exception as error:
            logger.error(f"Error retrieving episodic memories: {error}")
            raise error

    async def analyze_memories(self, request: MemoryAnalysisRequest) -> Dict[str, Any]:
        """Analyze memory patterns and insights."""
        if not self.is_initialized:
            raise ValueError("EpisodicMemoryEngine not initialized")

        try:
            # Get all memories for agent
            memories = await self.episodic_collection.get_agent_memories(request.agent_id)

            analysis = {}
            insights = []
            recommendations = []

            if request.analysis_type == "temporal_patterns":
                # Analyze temporal patterns
                temporal_analysis = self._analyze_temporal_patterns(memories)
                analysis["temporal"] = temporal_analysis
                insights.extend(temporal_analysis.get("insights", []))

            elif request.analysis_type == "emotional_patterns":
                # Analyze emotional patterns
                emotional_analysis = self._analyze_emotional_patterns(memories)
                analysis["emotional"] = emotional_analysis
                insights.extend(emotional_analysis.get("insights", []))

            elif request.analysis_type == "learning_progression":
                # Analyze learning progression
                learning_analysis = self._analyze_learning_progression(memories)
                analysis["learning"] = learning_analysis
                insights.extend(learning_analysis.get("insights", []))

            elif request.analysis_type == "social_patterns":
                # Analyze social interaction patterns
                social_analysis = self._analyze_social_patterns(memories)
                analysis["social"] = social_analysis
                insights.extend(social_analysis.get("insights", []))

            else:
                # Comprehensive analysis
                analysis = {
                    "temporal": self._analyze_temporal_patterns(memories),
                    "emotional": self._analyze_emotional_patterns(memories),
                    "learning": self._analyze_learning_progression(memories),
                    "social": self._analyze_social_patterns(memories)
                }

                for category_analysis in analysis.values():
                    insights.extend(category_analysis.get("insights", []))

            # Generate recommendations
            recommendations = self._generate_analysis_recommendations(analysis, insights)

            return {
                "analysis": analysis,
                "insights": insights,
                "recommendations": recommendations
            }

        except Exception as error:
            logger.error(f"Error analyzing memories: {error}")
            raise error

    # Helper methods

    def _categorize_experience(self, experience: Dict[str, Any]) -> str:
        """Categorize experience type."""
        participants = experience.get("participants", [])
        event_type = experience.get("event", {}).get("type", "")

        if len(participants) > 1:
            return "interaction"
        elif "learn" in event_type.lower() or "study" in event_type.lower():
            return "learning"
        elif "decision" in event_type.lower() or "choose" in event_type.lower():
            return "decision"
        elif "observe" in event_type.lower() or "watch" in event_type.lower():
            return "observation"
        elif "reflect" in event_type.lower() or "think" in event_type.lower():
            return "reflection"
        else:
            return "experience"

    def _determine_category(self, experience: Dict[str, Any]) -> str:
        """Determine memory category."""
        context = experience.get("context", {})
        participants = experience.get("participants", [])

        if context.get("work"):
            return "professional"
        elif len(participants) > 0:
            return "social"
        elif "learn" in str(experience.get("event", {})).lower():
            return "educational"
        elif experience.get("emotions") and len(experience["emotions"]) > 0:
            return "emotional"
        elif "procedure" in str(experience.get("event", {})).lower():
            return "procedural"
        else:
            return "personal"

    def _calculate_relative_time(self, timestamp: datetime) -> str:
        """Calculate relative time description."""
        now = datetime.utcnow()
        diff = now - timestamp
        diff_hours = diff.total_seconds() / 3600

        if diff_hours < 1:
            return "just_now"
        elif diff_hours < 24:
            return "today"
        elif diff_hours < 48:
            return "yesterday"
        elif diff_hours < 168:  # 7 days
            return "this_week"
        elif diff_hours < 720:  # 30 days
            return "this_month"
        else:
            return "long_ago"

    async def _find_memory_connections(
        self,
        agent_id: str,
        memory: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find connections to existing memories."""
        try:
            # Get recent memories for comparison
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_memories = await self.episodic_collection.collection.find({
                "agentId": agent_id,
                "episode.experience.temporal.startTime": {"$gte": week_ago}
            }).to_list(length=None)

            connections = []
            memory_location = memory.get("episode", {}).get("experience", {}).get("spatial", {}).get("location", "")
            memory_participants = [p.get("name", "") for p in memory.get("episode", {}).get("experience", {}).get("participants", [])]

            for existing_memory in recent_memories:
                if existing_memory.get("memoryId") == memory.get("memoryId"):
                    continue

                existing_location = existing_memory.get("episode", {}).get("experience", {}).get("spatial", {}).get("location", "")
                existing_participants = [p.get("name", "") for p in existing_memory.get("episode", {}).get("experience", {}).get("participants", [])]

                # Check for location similarity
                if memory_location and existing_location and memory_location == existing_location:
                    connections.append({
                        "type": "spatial",
                        "relatedMemoryId": existing_memory.get("memoryId", ""),
                        "strength": 0.8
                    })

                # Check for participant overlap
                common_participants = set(memory_participants) & set(existing_participants)
                if common_participants:
                    connections.append({
                        "type": "social",
                        "relatedMemoryId": existing_memory.get("memoryId", ""),
                        "strength": len(common_participants) / max(len(memory_participants), len(existing_participants))
                    })

            return connections[:5]  # Limit to top 5 connections

        except Exception as error:
            logger.error(f"Error finding memory connections: {error}")
            return []

    def _generate_processing_insights(
        self,
        memory: Dict[str, Any],
        connections: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate processing insights."""
        insights = []

        episode = memory.get("episode", {})
        vividness = episode.get("vividness", 0)
        importance = episode.get("importance", 0)

        insights.append(f"Memory encoded with {vividness:.2f} vividness")
        insights.append(f"Importance level: {importance:.2f}")

        if connections:
            insights.append(f"Found {len(connections)} connections to existing memories")

        emotions = episode.get("experience", {}).get("emotions", [])
        if emotions:
            avg_valence = sum(e.get("valence", 0) for e in emotions) / len(emotions)
            if avg_valence > 0.5:
                insights.append("Positive emotional experience detected")
            elif avg_valence < -0.5:
                insights.append("Negative emotional experience detected")

        return insights

    def _calculate_relevance_score(
        self,
        memory: Dict[str, Any],
        request: MemoryRetrievalRequest
    ) -> float:
        """Calculate relevance score for memory retrieval."""
        episode = memory.get("episode", {})

        # Base score from memory properties
        score = episode.get("importance", 0.5) * 0.4
        score += episode.get("vividness", 0.5) * 0.3
        score += episode.get("confidence", 0.5) * 0.3

        # Adjust based on recency
        timestamp = episode.get("experience", {}).get("temporal", {}).get("startTime")
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            days_ago = (datetime.utcnow() - timestamp).days
            recency_factor = max(0.1, 1.0 - (days_ago / 365))  # Decay over a year
            score *= recency_factor

        return min(1.0, score)

    def _determine_retrieval_reason(
        self,
        memory: Dict[str, Any],
        request: MemoryRetrievalRequest
    ) -> str:
        """Determine retrieval reason."""
        query_type = request.query.get("type", "free_text")

        if query_type == "temporal":
            return "temporal_match"
        elif query_type == "spatial":
            return "spatial_match"
        elif query_type == "social":
            return "social_match"
        elif query_type == "emotional":
            return "emotional_match"
        elif query_type == "thematic":
            return "thematic_match"
        else:
            return "content_match"

    def _calculate_contextual_fit(
        self,
        memory: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate contextual fit."""
        fit = 0.5  # Base fit

        # Adjust based on current emotions
        current_emotions = context.get("currentEmotions", [])
        memory_emotions = memory.get("episode", {}).get("experience", {}).get("emotions", [])

        if current_emotions and memory_emotions:
            memory_emotion_names = [e.get("emotion", "") for e in memory_emotions]
            emotion_match = any(emotion in memory_emotion_names for emotion in current_emotions)
            if emotion_match:
                fit += 0.2

        # Adjust based on current location
        current_location = context.get("currentLocation", "")
        memory_location = memory.get("episode", {}).get("experience", {}).get("spatial", {}).get("location", "")

        if current_location and memory_location and current_location == memory_location:
            fit += 0.3

        return min(1.0, fit)

    async def _detect_memory_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in retrieved memories."""
        temporal_patterns = []
        spatial_patterns = []
        social_patterns = []
        emotional_patterns = []

        if not memories:
            return {
                "temporalPatterns": temporal_patterns,
                "spatialPatterns": spatial_patterns,
                "socialPatterns": social_patterns,
                "emotionalPatterns": emotional_patterns
            }

        # Temporal patterns
        time_of_day_counts = {}
        for memory in memories:
            time_of_day = memory.get("episode", {}).get("experience", {}).get("temporal", {}).get("timeOfDay")
            if time_of_day:
                time_of_day_counts[time_of_day] = time_of_day_counts.get(time_of_day, 0) + 1

        for time_period, count in time_of_day_counts.items():
            if count > 1:
                temporal_patterns.append({
                    "pattern": f"frequent_{time_period}_activities",
                    "frequency": count,
                    "significance": count / len(memories)
                })

        # Spatial patterns
        location_counts = {}
        for memory in memories:
            location = memory.get("episode", {}).get("experience", {}).get("spatial", {}).get("location")
            if location:
                location_counts[location] = location_counts.get(location, 0) + 1

        for location, count in location_counts.items():
            if count > 1:
                spatial_patterns.append({
                    "pattern": f"frequent_location_{location}",
                    "frequency": count,
                    "significance": count / len(memories)
                })

        # Social patterns
        participant_counts = {}
        for memory in memories:
            participants = memory.get("episode", {}).get("experience", {}).get("participants", [])
            for participant in participants:
                name = participant.get("name", "")
                if name:
                    participant_counts[name] = participant_counts.get(name, 0) + 1

        for participant, count in participant_counts.items():
            if count > 1:
                social_patterns.append({
                    "pattern": f"frequent_interaction_{participant}",
                    "frequency": count,
                    "significance": count / len(memories)
                })

        # Emotional patterns
        emotion_counts = {}
        for memory in memories:
            emotions = memory.get("episode", {}).get("experience", {}).get("emotions", [])
            for emotion in emotions:
                emotion_name = emotion.get("emotion", "")
                if emotion_name:
                    emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1

        for emotion, count in emotion_counts.items():
            if count > 1:
                emotional_patterns.append({
                    "pattern": f"frequent_emotion_{emotion}",
                    "frequency": count,
                    "significance": count / len(memories)
                })

        return {
            "temporalPatterns": temporal_patterns,
            "spatialPatterns": spatial_patterns,
            "socialPatterns": social_patterns,
            "emotionalPatterns": emotional_patterns
        }

    def _generate_retrieval_insights(
        self,
        retrieved_memories: List[RetrievedMemory],
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from retrieved memories."""
        insights = []

        if not retrieved_memories:
            insights.append("No memories retrieved for the given query")
            return insights

        avg_relevance = sum(rm.relevance_score for rm in retrieved_memories) / len(retrieved_memories)
        insights.append(f"Retrieved {len(retrieved_memories)} memories with average relevance {avg_relevance:.2f}")

        # Pattern insights
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                strongest_pattern = max(pattern_list, key=lambda p: p["significance"])
                insights.append(f"Strongest {pattern_type[:-8]} pattern: {strongest_pattern['pattern']} (significance: {strongest_pattern['significance']:.2f})")

        # Temporal insights
        if retrieved_memories:
            categories = [rm.memory.get("episode", {}).get("category", "unknown") for rm in retrieved_memories]
            most_common_category = max(set(categories), key=categories.count)
            insights.append(f"Most common memory category: {most_common_category}")

        return insights

    def _analyze_temporal_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in memories."""
        insights = []

        # Time of day analysis
        time_counts = {}
        for memory in memories:
            time_of_day = memory.get("episode", {}).get("experience", {}).get("temporal", {}).get("timeOfDay")
            if time_of_day:
                time_counts[time_of_day] = time_counts.get(time_of_day, 0) + 1

        if time_counts:
            most_active_time = max(time_counts.items(), key=lambda x: x[1])
            insights.append(f"Most active time: {most_active_time[0]} ({most_active_time[1]} memories)")

        return {
            "timeOfDayDistribution": time_counts,
            "insights": insights
        }

    def _analyze_emotional_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotional patterns in memories."""
        insights = []
        emotion_counts = {}
        valence_sum = 0
        valence_count = 0

        for memory in memories:
            emotions = memory.get("episode", {}).get("experience", {}).get("emotions", [])
            for emotion in emotions:
                emotion_name = emotion.get("emotion", "")
                if emotion_name:
                    emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1

                valence = emotion.get("valence", 0)
                if valence != 0:
                    valence_sum += valence
                    valence_count += 1

        if emotion_counts:
            most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            insights.append(f"Most common emotion: {most_common_emotion[0]} ({most_common_emotion[1]} occurrences)")

        if valence_count > 0:
            avg_valence = valence_sum / valence_count
            if avg_valence > 0.2:
                insights.append("Generally positive emotional experiences")
            elif avg_valence < -0.2:
                insights.append("Generally negative emotional experiences")
            else:
                insights.append("Balanced emotional experiences")

        return {
            "emotionDistribution": emotion_counts,
            "averageValence": avg_valence if valence_count > 0 else 0,
            "insights": insights
        }

    def _analyze_learning_progression(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning progression in memories."""
        insights = []
        learning_memories = [m for m in memories if m.get("episode", {}).get("type") == "learning"]

        if learning_memories:
            insights.append(f"Found {len(learning_memories)} learning experiences")

            # Analyze knowledge growth
            knowledge_items = []
            for memory in learning_memories:
                learning = memory.get("episode", {}).get("learning", {})
                knowledge_items.extend(learning.get("knowledge", []))

            if knowledge_items:
                insights.append(f"Acquired {len(knowledge_items)} knowledge items")

        return {
            "learningMemoryCount": len(learning_memories),
            "knowledgeAcquisition": len(knowledge_items) if 'knowledge_items' in locals() else 0,
            "insights": insights
        }

    def _analyze_social_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze social interaction patterns."""
        insights = []
        participant_counts = {}

        for memory in memories:
            participants = memory.get("episode", {}).get("experience", {}).get("participants", [])
            for participant in participants:
                name = participant.get("name", "")
                if name:
                    participant_counts[name] = participant_counts.get(name, 0) + 1

        if participant_counts:
            most_frequent_contact = max(participant_counts.items(), key=lambda x: x[1])
            insights.append(f"Most frequent contact: {most_frequent_contact[0]} ({most_frequent_contact[1]} interactions)")

        return {
            "participantFrequency": participant_counts,
            "insights": insights
        }

    def _generate_analysis_recommendations(
        self,
        analysis: Dict[str, Any],
        insights: List[str]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Temporal recommendations
        temporal = analysis.get("temporal", {})
        if temporal.get("timeOfDayDistribution"):
            recommendations.append("Consider scheduling important activities during your most active time periods")

        # Emotional recommendations
        emotional = analysis.get("emotional", {})
        avg_valence = emotional.get("averageValence", 0)
        if avg_valence < -0.3:
            recommendations.append("Consider strategies to improve emotional well-being")
        elif avg_valence > 0.3:
            recommendations.append("Continue engaging in activities that bring positive emotions")

        # Learning recommendations
        learning = analysis.get("learning", {})
        if learning.get("learningMemoryCount", 0) < 5:
            recommendations.append("Consider increasing learning activities for better cognitive development")

        # Social recommendations
        social = analysis.get("social", {})
        if not social.get("participantFrequency"):
            recommendations.append("Consider increasing social interactions for better well-being")

        return recommendations

    async def get_memory_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an agent."""
        try:
            memories = await self.episodic_collection.get_agent_memories(agent_id)

            if not memories:
                return {
                    "totalMemories": 0,
                    "averageImportance": 0,
                    "averageVividness": 0,
                    "memoryTypes": {},
                    "temporalDistribution": {}
                }

            total_memories = len(memories)
            avg_importance = sum(m.get("episode", {}).get("importance", 0) for m in memories) / total_memories
            avg_vividness = sum(m.get("episode", {}).get("vividness", 0) for m in memories) / total_memories

            # Memory types
            memory_types = {}
            for memory in memories:
                mem_type = memory.get("episode", {}).get("type", "unknown")
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

            # Temporal distribution
            temporal_distribution = {}
            for memory in memories:
                relative_time = memory.get("episode", {}).get("relativeTime", "unknown")
                temporal_distribution[relative_time] = temporal_distribution.get(relative_time, 0) + 1

            return {
                "totalMemories": total_memories,
                "averageImportance": avg_importance,
                "averageVividness": avg_vividness,
                "memoryTypes": memory_types,
                "temporalDistribution": temporal_distribution
            }

        except Exception as error:
            logger.error(f"Error getting memory statistics: {error}")
            return {
                "totalMemories": 0,
                "averageImportance": 0,
                "averageVividness": 0,
                "memoryTypes": {},
                "temporalDistribution": {}
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup any resources if needed
        pass
