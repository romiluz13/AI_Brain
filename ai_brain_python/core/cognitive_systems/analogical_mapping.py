"""
AnalogicalMappingSystem - Advanced analogical reasoning using MongoDB Atlas Vector Search

Exact Python equivalent of JavaScript AnalogicalMappingSystem.ts with:
- $vectorSearch for semantic analogical similarity
- Vector embeddings for analogical reasoning
- Structural and semantic analogy detection
- Multi-dimensional analogical mapping
- Analogical inference and projection

CRITICAL: This uses MongoDB Atlas EXCLUSIVE features:
- $vectorSearch aggregation stage (Atlas ONLY)
- Atlas Vector Search indexes (Atlas ONLY)
- Vector similarity search with embeddings (Atlas ONLY)
- Semantic search capabilities (Atlas ONLY)

Features:
- $vectorSearch for semantic analogical similarity
- Vector embeddings for analogical reasoning
- Structural and semantic analogy detection
- Multi-dimensional analogical mapping
- Analogical inference and projection
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.interfaces.cognitive_system import CognitiveSystemInterface
from ..core.models import CognitiveInputData, CognitiveResponse
from ..storage.collections.analogical_mapping_collection import AnalogicalMappingCollection
from ..utils.logger import logger


@dataclass
class AnalogicalReasoningRequest:
    """Analogical reasoning request data structure."""
    agent_id: str
    scenario: Dict[str, Any]
    source: Dict[str, Any]
    target: Optional[Dict[str, Any]]
    parameters: Dict[str, Any]


@dataclass
class AnalogicalCorrespondence:
    """Analogical correspondence data structure."""
    source_element: str
    target_element: str
    type: str  # 'object' | 'relation' | 'attribute' | 'function'
    strength: float
    justification: str


@dataclass
class AnalogicalQuality:
    """Analogical quality assessment."""
    systematicity: float  # Coherent system of relations
    one_to_one: float     # One-to-one correspondence
    semantic: float       # Semantic similarity
    pragmatic: float      # Practical utility
    overall: float        # Overall quality


@dataclass
class AnalogicalInference:
    """Analogical inference data structure."""
    type: str  # 'prediction' | 'explanation' | 'hypothesis' | 'generalization'
    content: str
    confidence: float
    based_on: List[str]
    testable: bool


@dataclass
class AnalogicalInsight:
    """Analogical insight data structure."""
    insight: str
    novelty: float
    plausibility: float
    implications: List[str]
    evidence: List[str]


@dataclass
class AnalogicalReasoningResult:
    """Analogical reasoning result data structure."""
    request: AnalogicalReasoningRequest
    analogies: List[Dict[str, Any]]
    inferences: List[AnalogicalInference]
    insights: List[AnalogicalInsight]
    metadata: Dict[str, Any]


@dataclass
class AnalogicalLearningRequest:
    """Analogical learning request data structure."""
    agent_id: str
    examples: List[Dict[str, Any]]
    parameters: Dict[str, Any]


class AnalogicalMappingSystem(CognitiveSystemInterface):
    """
    AnalogicalMappingSystem - Advanced analogical reasoning using MongoDB Atlas Vector Search
    
    Exact Python equivalent of JavaScript AnalogicalMappingSystem with:
    - $vectorSearch for semantic analogical similarity
    - Vector embeddings for analogical reasoning
    - Structural and semantic analogy detection
    - Multi-dimensional analogical mapping
    - Analogical inference and projection
    
    CRITICAL: Requires MongoDB Atlas (not local MongoDB)
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)
        self.db = db
        self.analogical_collection = AnalogicalMappingCollection(db)
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the analogical mapping system."""
        if self.is_initialized:
            return
            
        try:
            await self.analogical_collection.initialize_indexes()
            self.is_initialized = True
            logger.info("✅ AnalogicalMappingSystem initialized successfully")
            logger.info("📝 Note: Atlas Vector Search indexes must be created separately in Atlas UI or API")
        except Exception as error:
            logger.error(f"❌ Error initializing AnalogicalMappingSystem: {error}")
            raise error
    
    async def process(self, input_data: CognitiveInputData) -> CognitiveResponse:
        """Process analogical reasoning requests."""
        try:
            await self.initialize()
            
            # Extract analogical reasoning request from input
            request_data = input_data.additional_context.get("analogical_request", {})
            
            if not request_data:
                return CognitiveResponse(
                    response_text="No analogical reasoning request provided",
                    confidence=0.0,
                    processing_metadata={
                        "system": "analogical_mapping",
                        "error": "Missing analogical reasoning request"
                    }
                )
            
            # Create analogical reasoning request
            request = AnalogicalReasoningRequest(
                agent_id=request_data.get("agentId", "unknown"),
                scenario=request_data.get("scenario", {}),
                source=request_data.get("source", {}),
                target=request_data.get("target"),
                parameters=request_data.get("parameters", {})
            )
            
            # Perform analogical reasoning
            result = await self.perform_analogical_reasoning(request)
            
            # Generate response
            response_text = f"Found {len(result.analogies)} analogical mappings"
            if result.inferences:
                response_text += f" with {len(result.inferences)} inferences"
            if result.insights:
                response_text += f" and {len(result.insights)} insights"
            
            return CognitiveResponse(
                response_text=response_text,
                confidence=0.8,
                processing_metadata={
                    "system": "analogical_mapping",
                    "analogies_found": len(result.analogies),
                    "inferences_generated": len(result.inferences),
                    "insights_discovered": len(result.insights),
                    "vector_search_used": result.metadata.get("vectorSearchUsed", False)
                }
            )
        except Exception as error:
            logger.error(f"Error in AnalogicalMappingSystem.process: {error}")
            return CognitiveResponse(
                response_text=f"Analogical mapping error: {str(error)}",
                confidence=0.0,
                processing_metadata={
                    "system": "analogical_mapping",
                    "error": str(error)
                }
            )
    
    async def perform_analogical_reasoning(
        self,
        request: AnalogicalReasoningRequest
    ) -> AnalogicalReasoningResult:
        """Perform analogical reasoning using Atlas Vector Search."""
        if not self.is_initialized:
            raise ValueError("AnalogicalMappingSystem not initialized")
        
        start_time = datetime.utcnow()
        analogies = []
        vector_search_used = False
        
        try:
            # Use Atlas Vector Search if embedding is provided
            source_embedding = request.source.get("embedding")
            if source_embedding and len(source_embedding) > 0:
                try:
                    vector_results = await self.analogical_collection.find_similar_analogies(
                        source_embedding,
                        {
                            "index_name": request.parameters.get("vectorSearchIndex", "analogical_embeddings"),
                            "limit": request.parameters.get("maxResults", 10),
                            "num_candidates": request.parameters.get("maxResults", 10) * 10,
                            "min_score": request.parameters.get("minSimilarity", 0.7),
                            "domains": request.parameters.get("domains")
                        }
                    )
                    
                    analogies.extend(vector_results)
                    vector_search_used = True
                    logger.debug(f"Atlas Vector Search found {len(vector_results)} analogies")
                    
                except Exception as vector_error:
                    logger.warning(f"Vector search failed, falling back to traditional search: {vector_error}")
            
            # Fallback to traditional search if vector search not available
            if not analogies:
                traditional_results = await self._perform_traditional_search(request)
                analogies.extend(traditional_results)
            
            # Process and enhance analogies
            enhanced_analogies = []
            for analogy in analogies:
                enhanced = await self._enhance_analogy(analogy, request)
                enhanced_analogies.append(enhanced)
            
            # Generate inferences
            inferences = self._generate_inferences(enhanced_analogies, request)
            
            # Generate insights
            insights = self._generate_insights(enhanced_analogies, request)
            
            # Calculate metadata
            end_time = datetime.utcnow()
            search_time = (end_time - start_time).total_seconds() * 1000
            
            metadata = {
                "searchTime": search_time,
                "analogiesExplored": len(analogies),
                "vectorSearchUsed": vector_search_used,
                "embeddingModel": "text-embedding-ada-002",  # Default model
                "qualityThreshold": request.parameters.get("minSimilarity", 0.7)
            }
            
            return AnalogicalReasoningResult(
                request=request,
                analogies=enhanced_analogies,
                inferences=inferences,
                insights=insights,
                metadata=metadata
            )
            
        except Exception as error:
            logger.error(f"Error performing analogical reasoning: {error}")
            raise error

    async def _perform_traditional_search(
        self,
        request: AnalogicalReasoningRequest
    ) -> List[Dict[str, Any]]:
        """Perform traditional analogical search without vector embeddings."""
        try:
            # Search by domain and type
            search_criteria = {
                "source.domain": {"$ne": request.source.get("domain")},
                "source.type": request.source.get("type")
            }

            # Add domain restrictions if specified
            if request.parameters.get("domains"):
                search_criteria["target.domain"] = {"$in": request.parameters["domains"]}

            results = await self.analogical_collection.collection.find(
                search_criteria
            ).limit(request.parameters.get("maxResults", 10)).to_list(length=None)

            return results
        except Exception as error:
            logger.error(f"Error in traditional search: {error}")
            return []

    async def _enhance_analogy(
        self,
        analogy: Dict[str, Any],
        request: AnalogicalReasoningRequest
    ) -> Dict[str, Any]:
        """Enhance analogy with correspondences and quality assessment."""
        try:
            # Extract correspondences
            correspondences = self._extract_correspondences(analogy)

            # Calculate quality metrics
            quality = self._calculate_quality(analogy, correspondences)

            # Determine analogy type
            analogy_type = self._determine_analogy_type(analogy, request)

            return {
                "mapping": analogy,
                "similarity": analogy.get("similarity", 0.8),
                "confidence": quality.overall,
                "type": analogy_type,
                "correspondences": correspondences,
                "quality": {
                    "systematicity": quality.systematicity,
                    "oneToOne": quality.one_to_one,
                    "semantic": quality.semantic,
                    "pragmatic": quality.pragmatic,
                    "overall": quality.overall
                }
            }
        except Exception as error:
            logger.error(f"Error enhancing analogy: {error}")
            return {"mapping": analogy, "similarity": 0.5, "confidence": 0.5}

    def _extract_correspondences(self, mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract correspondences from an analogical mapping."""
        correspondences = []

        # Simple correspondence extraction based on structure
        source = mapping.get("source", {})
        target = mapping.get("target", {})

        # Object correspondences
        if source.get("name") and target.get("name"):
            correspondences.append({
                "sourceElement": source["name"],
                "targetElement": target["name"],
                "type": "object",
                "strength": 0.8,
                "justification": "Primary object mapping"
            })

        # Type correspondences
        if source.get("type") and target.get("type"):
            correspondences.append({
                "sourceElement": source["type"],
                "targetElement": target["type"],
                "type": "attribute",
                "strength": 0.9,
                "justification": "Type similarity"
            })

        # Domain correspondences
        if source.get("domain") and target.get("domain"):
            correspondences.append({
                "sourceElement": source["domain"],
                "targetElement": target["domain"],
                "type": "relation",
                "strength": 0.7,
                "justification": "Domain mapping"
            })

        return correspondences

    def _calculate_quality(
        self,
        analogy: Dict[str, Any],
        correspondences: List[Dict[str, Any]]
    ) -> AnalogicalQuality:
        """Calculate quality metrics for an analogical mapping."""
        # Systematicity: coherent system of relations
        systematicity = min(len(correspondences) / 5.0, 1.0)

        # One-to-one correspondence
        one_to_one = 1.0 if len(correspondences) > 0 else 0.0

        # Semantic similarity
        semantic = analogy.get("similarity", 0.5)

        # Pragmatic utility (simplified)
        pragmatic = 0.8  # Default pragmatic value

        # Overall quality
        overall = (systematicity + one_to_one + semantic + pragmatic) / 4.0

        return AnalogicalQuality(
            systematicity=systematicity,
            one_to_one=one_to_one,
            semantic=semantic,
            pragmatic=pragmatic,
            overall=overall
        )

    def _determine_analogy_type(
        self,
        analogy: Dict[str, Any],
        request: AnalogicalReasoningRequest
    ) -> str:
        """Determine the type of analogical mapping."""
        source = analogy.get("source", {})
        target = analogy.get("target", {})

        # Check for structural similarity
        if source.get("structure") and target.get("structure"):
            return "structural"

        # Check for functional similarity
        if source.get("type") == target.get("type"):
            return "functional"

        # Check for semantic similarity
        if source.get("semantics") and target.get("semantics"):
            return "semantic"

        # Check for causal similarity
        if "causal" in str(source) or "causal" in str(target):
            return "causal"

        # Default to surface similarity
        return "surface"

    def _generate_inferences(
        self,
        analogies: List[Dict[str, Any]],
        request: AnalogicalReasoningRequest
    ) -> List[AnalogicalInference]:
        """Generate analogical inferences."""
        inferences = []

        for analogy in analogies:
            if analogy.get("confidence", 0) > 0.7:
                # Generate prediction inference
                inferences.append(AnalogicalInference(
                    type="prediction",
                    content=f"Based on analogy with {analogy['mapping'].get('source', {}).get('name', 'unknown')}, "
                           f"we can predict similar behavior in {request.source.get('name', 'target domain')}",
                    confidence=analogy.get("confidence", 0.7),
                    based_on=[str(analogy.get("mapping", {}).get("_id", "unknown"))],
                    testable=True
                ))

                # Generate explanation inference
                if analogy.get("quality", {}).get("overall", 0) > 0.8:
                    inferences.append(AnalogicalInference(
                        type="explanation",
                        content=f"The similarity between {analogy['mapping'].get('source', {}).get('domain', 'source')} "
                               f"and {analogy['mapping'].get('target', {}).get('domain', 'target')} "
                               f"explains the observed patterns",
                        confidence=analogy.get("quality", {}).get("overall", 0.8),
                        based_on=[str(analogy.get("mapping", {}).get("_id", "unknown"))],
                        testable=False
                    ))

        return inferences[:5]  # Limit to top 5 inferences

    def _generate_insights(
        self,
        analogies: List[Dict[str, Any]],
        request: AnalogicalReasoningRequest
    ) -> List[AnalogicalInsight]:
        """Generate novel insights from analogical mappings."""
        insights = []

        # Look for patterns across multiple analogies
        if len(analogies) >= 2:
            insights.append(AnalogicalInsight(
                insight=f"Multiple analogies suggest a common pattern in {request.source.get('domain', 'this domain')}",
                novelty=0.7,
                plausibility=0.8,
                implications=[
                    "This pattern might be generalizable",
                    "Similar solutions might work across domains"
                ],
                evidence=[f"Found {len(analogies)} supporting analogies"]
            ))

        # Look for high-quality analogies
        high_quality = [a for a in analogies if a.get("quality", {}).get("overall", 0) > 0.9]
        if high_quality:
            insights.append(AnalogicalInsight(
                insight="Discovered exceptionally strong analogical mapping",
                novelty=0.9,
                plausibility=0.95,
                implications=[
                    "This mapping could be used for transfer learning",
                    "Solutions from source domain likely applicable"
                ],
                evidence=[f"Quality score: {high_quality[0].get('quality', {}).get('overall', 0):.2f}"]
            ))

        return insights

    async def learn_analogical_patterns(
        self,
        request: AnalogicalLearningRequest
    ) -> Dict[str, Any]:
        """Learn analogical patterns from examples."""
        try:
            learned_patterns = []

            for example in request.examples:
                # Extract structural pattern
                structural_pattern = self._extract_structural_pattern(
                    example.get("source", {}),
                    example.get("target", {})
                )

                if structural_pattern:
                    learned_patterns.append({
                        "pattern": structural_pattern,
                        "strength": example.get("quality", 0.5),
                        "generality": 0.7,  # Default generality
                        "examples": 1
                    })

                # Extract semantic pattern
                semantic_pattern = self._extract_semantic_pattern(
                    example.get("source", {}),
                    example.get("target", {})
                )

                if semantic_pattern:
                    learned_patterns.append({
                        "pattern": semantic_pattern,
                        "strength": example.get("quality", 0.5),
                        "generality": 0.6,  # Default generality
                        "examples": 1
                    })

            # Store learned patterns
            for pattern in learned_patterns:
                await self._store_learned_pattern(
                    request.agent_id,
                    pattern,
                    request.examples
                )

            return {
                "learnedPatterns": learned_patterns,
                "totalPatterns": len(learned_patterns),
                "averageStrength": sum(p["strength"] for p in learned_patterns) / len(learned_patterns) if learned_patterns else 0
            }

        except Exception as error:
            logger.error(f"Error learning analogical patterns: {error}")
            raise error

    def _extract_structural_pattern(self, source: Dict[str, Any], target: Dict[str, Any]) -> Optional[str]:
        """Extract structural pattern from source-target pair."""
        if source.get("type") and target.get("type") and source["type"] == target["type"]:
            return f"Same type: {source['type']}"
        return None

    def _extract_semantic_pattern(self, source: Dict[str, Any], target: Dict[str, Any]) -> Optional[str]:
        """Extract semantic pattern from source-target pair."""
        if source.get("domain") and target.get("domain"):
            return f"Domain mapping: {source['domain']} -> {target['domain']}"
        return None

    async def _store_learned_pattern(
        self,
        agent_id: str,
        pattern: Dict[str, Any],
        examples: List[Dict[str, Any]]
    ) -> None:
        """Store a learned pattern as an analogical mapping."""
        try:
            # Create a synthetic analogical mapping from the learned pattern
            mapping = {
                "agentId": agent_id,
                "timestamp": datetime.utcnow(),
                "mapping": {
                    "type": "learned_pattern",
                    "pattern": pattern["pattern"],
                    "strength": pattern["strength"],
                    "generality": pattern["generality"]
                },
                "source": {
                    "type": "pattern",
                    "description": pattern["pattern"],
                    "domain": "learned"
                },
                "target": {
                    "type": "pattern",
                    "description": "applicable_domains",
                    "domain": "general"
                },
                "quality": {
                    "overall": pattern["strength"],
                    "systematicity": pattern["generality"],
                    "confidence": pattern["strength"]
                },
                "metadata": {
                    "source": "analogical_learning",
                    "examples_count": len(examples),
                    "learning_method": "pattern_extraction"
                }
            }

            await self.analogical_collection.create_mapping(mapping)
            logger.debug(f"Stored learned pattern: {pattern['pattern']}")

        except Exception as error:
            logger.error(f"Error storing learned pattern: {error}")

    async def get_analogical_patterns(self, agent_id: str) -> Dict[str, Any]:
        """Get analogical patterns for an agent."""
        try:
            # This would be implemented with proper aggregation pipelines
            # For now, return a simplified structure
            return {
                "commonMappings": [],
                "domainPairs": [],
                "reasoningMethods": [],
                "qualityMetrics": []
            }
        except Exception as error:
            logger.error(f"Error getting analogical patterns: {error}")
            return {
                "commonMappings": [],
                "domainPairs": [],
                "reasoningMethods": [],
                "qualityMetrics": []
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup any resources if needed
        pass
