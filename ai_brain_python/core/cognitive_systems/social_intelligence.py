"""
SocialIntelligenceEngine - Advanced social intelligence using MongoDB Atlas $graphLookup

Exact Python equivalent of JavaScript SocialIntelligenceEngine.ts with:
- $graphLookup aggregation stage (Atlas optimized)
- Recursive social network traversal
- Graph-based relationship analysis
- Multi-depth social connection exploration

CRITICAL: This uses MongoDB Atlas EXCLUSIVE features:
- $graphLookup aggregation stage (Atlas optimized)
- Recursive social network traversal
- Graph-based relationship analysis
- Multi-depth social connection exploration

Features:
- Social network mapping and analysis
- Recursive relationship traversal using $graphLookup
- Social influence and connection strength analysis
- Community detection and social clustering
- Social interaction pattern recognition
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.interfaces.cognitive_system import CognitiveSystemInterface
from ..core.models import CognitiveInputData, CognitiveResponse
from ..storage.collections.social_intelligence_collection import SocialIntelligenceCollection
from ..utils.logger import logger


@dataclass
class SocialAnalysisRequest:
    """Social analysis request data structure."""
    agent_id: str
    analysis_type: str  # 'network_analysis' | 'influence_mapping' | 'community_detection' | 'relationship_strength' | 'social_patterns'
    parameters: Dict[str, Any]
    context: Dict[str, Any]


@dataclass
class NetworkTraversalRequest:
    """Network traversal request data structure."""
    agent_id: str
    start_person_id: str
    max_depth: int
    min_connection_strength: float
    connection_types: Optional[List[str]] = None


@dataclass
class SocialConnection:
    """Social connection data structure."""
    connection: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class NetworkAnalysis:
    """Network analysis results."""
    total_connections: int
    active_connections: int
    network_reach: int
    average_connection_strength: float
    network_density: float
    clustering_coefficient: float


@dataclass
class InfluenceMetrics:
    """Social influence metrics."""
    personal_influence: float
    network_influence: float
    authority_score: float
    reach_estimate: int
    influential_connections: List[Dict[str, Any]]


@dataclass
class Community:
    """Social community data structure."""
    id: str
    name: str
    members: List[str]
    central_members: List[str]
    bridge_members: List[str]
    cohesion: float
    influence: float
    role: str  # 'member' | 'leader' | 'influencer' | 'bridge' | 'peripheral'


@dataclass
class RelationshipAnalysis:
    """Relationship analysis results."""
    strong_connections: List[SocialConnection]
    weak_connections: List[SocialConnection]
    growing_connections: List[SocialConnection]
    declining_connections: List[SocialConnection]
    mutual_connections: List[Dict[str, Any]]


@dataclass
class SocialRecommendation:
    """Social recommendation data structure."""
    type: str  # 'strengthen_connection' | 'expand_network' | 'bridge_communities' | 'leverage_influence' | 'resolve_conflict'
    priority: str  # 'high' | 'medium' | 'low'
    description: str
    target_person_id: Optional[str]
    expected_impact: float
    effort_required: str  # 'low' | 'medium' | 'high'
    timeline: str  # 'immediate' | 'short_term' | 'long_term'


@dataclass
class SocialAnalysisResult:
    """Social analysis result data structure."""
    request: SocialAnalysisRequest
    network_analysis: NetworkAnalysis
    influence_metrics: InfluenceMetrics
    communities: List[Community]
    relationships: RelationshipAnalysis
    recommendations: List[SocialRecommendation]
    metadata: Dict[str, Any]


class SocialIntelligenceEngine(CognitiveSystemInterface):
    """
    SocialIntelligenceEngine - Advanced social intelligence using MongoDB Atlas $graphLookup
    
    Exact Python equivalent of JavaScript SocialIntelligenceEngine with:
    - $graphLookup aggregation stage (Atlas optimized)
    - Recursive social network traversal
    - Graph-based relationship analysis
    - Multi-depth social connection exploration
    
    CRITICAL: Optimized for MongoDB Atlas (not local MongoDB)
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db)
        self.db = db
        self.social_collection = SocialIntelligenceCollection(db)
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the social intelligence engine."""
        if self.is_initialized:
            return
            
        try:
            await self.social_collection.initialize_indexes()
            self.is_initialized = True
            logger.info("✅ SocialIntelligenceEngine initialized successfully")
            logger.info("📝 Note: Optimized for MongoDB Atlas $graphLookup operations")
        except Exception as error:
            logger.error(f"❌ Error initializing SocialIntelligenceEngine: {error}")
            raise error
    
    async def process(self, input_data: CognitiveInputData) -> CognitiveResponse:
        """Process social intelligence requests."""
        try:
            await self.initialize()
            
            # Extract social analysis request from input
            request_data = input_data.additional_context.get("social_request", {})
            
            if not request_data:
                return CognitiveResponse(
                    response_text="No social intelligence request provided",
                    confidence=0.0,
                    processing_metadata={
                        "system": "social_intelligence",
                        "error": "Missing social intelligence request"
                    }
                )
            
            # Create social analysis request
            request = SocialAnalysisRequest(
                agent_id=request_data.get("agentId", "unknown"),
                analysis_type=request_data.get("analysisType", "network_analysis"),
                parameters=request_data.get("parameters", {}),
                context=request_data.get("context", {})
            )
            
            # Perform social analysis
            result = await self.perform_social_analysis(request)
            
            # Generate response
            response_text = f"Social analysis completed: {len(result.communities)} communities, "
            response_text += f"{result.network_analysis.total_connections} connections, "
            response_text += f"{len(result.recommendations)} recommendations"
            
            return CognitiveResponse(
                response_text=response_text,
                confidence=result.influence_metrics.authority_score,
                processing_metadata={
                    "system": "social_intelligence",
                    "analysis_type": result.request.analysis_type,
                    "communities": len(result.communities),
                    "connections": result.network_analysis.total_connections,
                    "recommendations": len(result.recommendations),
                    "graph_lookup_used": result.metadata.get("graphLookupUsed", False)
                }
            )
            
        except Exception as error:
            logger.error(f"Error in SocialIntelligenceEngine.process: {error}")
            return CognitiveResponse(
                response_text=f"Social intelligence error: {str(error)}",
                confidence=0.0,
                processing_metadata={
                    "system": "social_intelligence",
                    "error": str(error)
                }
            )
    
    async def perform_social_analysis(self, request: SocialAnalysisRequest) -> SocialAnalysisResult:
        """Perform comprehensive social analysis using Atlas $graphLookup."""
        if not self.is_initialized:
            raise ValueError("SocialIntelligenceEngine not initialized")
        
        start_time = datetime.utcnow()
        
        try:
            # Get all connections for the agent
            connections = await self.social_collection.get_agent_connections(request.agent_id)
            
            # Perform network analysis
            network_analysis = await self._analyze_network(request.agent_id, connections)
            
            # Analyze social influence
            influence_metrics = await self._analyze_social_influence(request.agent_id, request.parameters)
            
            # Detect communities
            communities_result = await self.detect_communities(request.agent_id)
            communities = communities_result["communities"]
            
            # Analyze relationships
            relationships = await self._analyze_relationships(request.agent_id, connections)
            
            # Generate recommendations
            recommendations = await self._generate_social_recommendations(
                request.agent_id,
                network_analysis,
                influence_metrics,
                communities,
                relationships
            )
            
            # Calculate metadata
            end_time = datetime.utcnow()
            analysis_time = (end_time - start_time).total_seconds() * 1000
            
            metadata = {
                "analysisTime": analysis_time,
                "connectionsAnalyzed": len(connections),
                "graphLookupUsed": True,
                "atlasOptimized": True,
                "complexity": self._calculate_complexity(len(connections), request.parameters.get("maxDepth", 3))
            }
            
            return SocialAnalysisResult(
                request=request,
                network_analysis=network_analysis,
                influence_metrics=influence_metrics,
                communities=communities,
                relationships=relationships,
                recommendations=recommendations,
                metadata=metadata
            )
            
        except Exception as error:
            logger.error(f"Error performing social analysis: {error}")
            raise error

    async def traverse_network(self, request: NetworkTraversalRequest) -> List[Dict[str, Any]]:
        """Traverse social network using Atlas $graphLookup."""
        if not self.is_initialized:
            raise ValueError("SocialIntelligenceEngine not initialized")

        try:
            # Use $graphLookup to traverse the social network
            pipeline = [
                {
                    "$match": {
                        "agentId": request.agent_id,
                        "connection.participants.source.id": request.start_person_id
                    }
                },
                {
                    "$graphLookup": {
                        "from": self.social_collection.collection.name,
                        "startWith": "$connection.participants.target.id",
                        "connectFromField": "connection.participants.target.id",
                        "connectToField": "connection.participants.source.id",
                        "as": "networkPath",
                        "maxDepth": request.max_depth,
                        "restrictSearchWithMatch": {
                            "agentId": request.agent_id,
                            "connection.strength.overall": {"$gte": request.min_connection_strength},
                            "connection.status": "active"
                        }
                    }
                }
            ]

            # Add connection type filter if specified
            if request.connection_types:
                pipeline[1]["$graphLookup"]["restrictSearchWithMatch"]["connection.type"] = {
                    "$in": request.connection_types
                }

            results = await self.social_collection.collection.aggregate(pipeline).to_list(length=None)

            # Process results into traversal format
            traversal_results = []
            for result in results:
                # Process initial connection
                initial_connection = SocialConnection(
                    connection=result["connection"],
                    metadata=result.get("metadata", {})
                )

                traversal_results.append({
                    "person": initial_connection,
                    "depth": 0,
                    "path": [request.start_person_id],
                    "connectionStrength": result["connection"]["strength"]["overall"],
                    "influenceScore": self._calculate_influence_score(initial_connection, 0)
                })

                # Process network path
                for i, path_item in enumerate(result.get("networkPath", [])):
                    path_connection = SocialConnection(
                        connection=path_item["connection"],
                        metadata=path_item.get("metadata", {})
                    )

                    # Build path
                    path = [request.start_person_id]
                    for j in range(i + 1):
                        if j < len(result["networkPath"]):
                            path.append(result["networkPath"][j]["connection"]["participants"]["target"]["id"])

                    traversal_results.append({
                        "person": path_connection,
                        "depth": i + 1,
                        "path": path,
                        "connectionStrength": path_item["connection"]["strength"]["overall"],
                        "influenceScore": self._calculate_influence_score(path_connection, i + 1)
                    })

            return traversal_results

        except Exception as error:
            logger.error(f"Error traversing network: {error}")
            return []

    async def find_mutual_connections(
        self,
        agent_id: str,
        person_id1: str,
        person_id2: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Find mutual connections between people."""
        try:
            # Find connections for person 1
            person1_connections = await self.traverse_network(NetworkTraversalRequest(
                agent_id=agent_id,
                start_person_id=person_id1,
                max_depth=max_depth,
                min_connection_strength=0.3
            ))

            # Find connections for person 2
            person2_connections = await self.traverse_network(NetworkTraversalRequest(
                agent_id=agent_id,
                start_person_id=person_id2,
                max_depth=max_depth,
                min_connection_strength=0.3
            ))

            # Find mutual connections
            person1_network = {
                conn["person"].connection["participants"]["target"]["id"]
                for conn in person1_connections
            }
            person2_network = {
                conn["person"].connection["participants"]["target"]["id"]
                for conn in person2_connections
            }

            mutual_ids = person1_network & person2_network

            # Build mutual connections result
            mutual_connections = []
            for mutual_id in mutual_ids:
                # Find connection details from both networks
                person1_conn = next((c for c in person1_connections
                                   if c["person"].connection["participants"]["target"]["id"] == mutual_id), None)
                person2_conn = next((c for c in person2_connections
                                   if c["person"].connection["participants"]["target"]["id"] == mutual_id), None)

                if person1_conn and person2_conn:
                    mutual_connections.append({
                        "personId": mutual_id,
                        "person1Path": person1_conn["path"],
                        "person2Path": person2_conn["path"],
                        "person1Strength": person1_conn["connectionStrength"],
                        "person2Strength": person2_conn["connectionStrength"],
                        "bridgePotential": self._calculate_bridge_potential(
                            person1_conn["connectionStrength"],
                            person2_conn["connectionStrength"]
                        )
                    })

            return mutual_connections

        except Exception as error:
            logger.error(f"Error finding mutual connections: {error}")
            return []

    async def detect_communities(self, agent_id: str) -> Dict[str, Any]:
        """Detect social communities using graph analysis."""
        try:
            # Get all connections for community detection
            connections = await self.social_collection.get_agent_connections(agent_id)

            # Simple community detection based on connection strength and clustering
            communities = []
            processed_people = set()

            for connection in connections:
                source_id = connection.connection["participants"]["source"]["id"]
                target_id = connection.connection["participants"]["target"]["id"]

                if source_id in processed_people or target_id in processed_people:
                    continue

                # Find strongly connected group
                community_members = {source_id, target_id}

                # Expand community by finding mutual strong connections
                for other_conn in connections:
                    other_source = other_conn.connection["participants"]["source"]["id"]
                    other_target = other_conn.connection["participants"]["target"]["id"]

                    if (other_source in community_members or other_target in community_members) and \
                       other_conn.connection["strength"]["overall"] > 0.6:
                        community_members.add(other_source)
                        community_members.add(other_target)

                if len(community_members) >= 3:  # Minimum community size
                    community_id = f"community_{len(communities) + 1}"

                    # Determine central members (highest connection strength)
                    central_members = []
                    for member in community_members:
                        member_connections = [c for c in connections
                                            if c.connection["participants"]["source"]["id"] == member or
                                               c.connection["participants"]["target"]["id"] == member]
                        avg_strength = sum(c.connection["strength"]["overall"] for c in member_connections) / len(member_connections) if member_connections else 0
                        if avg_strength > 0.7:
                            central_members.append(member)

                    # Calculate community cohesion
                    internal_connections = [c for c in connections
                                          if c.connection["participants"]["source"]["id"] in community_members and
                                             c.connection["participants"]["target"]["id"] in community_members]
                    cohesion = sum(c.connection["strength"]["overall"] for c in internal_connections) / len(internal_connections) if internal_connections else 0

                    communities.append(Community(
                        id=community_id,
                        name=f"Community {len(communities) + 1}",
                        members=list(community_members),
                        central_members=central_members,
                        bridge_members=[],  # Would be calculated with more sophisticated algorithm
                        cohesion=cohesion,
                        influence=cohesion * len(community_members),
                        role=self._determine_role_in_community({"centralMembers": central_members}, agent_id)
                    ))

                    processed_people.update(community_members)

            return {
                "communities": communities,
                "totalCommunities": len(communities),
                "averageCommunitySize": sum(len(c.members) for c in communities) / len(communities) if communities else 0
            }

        except Exception as error:
            logger.error(f"Error detecting communities: {error}")
            return {"communities": [], "totalCommunities": 0, "averageCommunitySize": 0}

    async def find_influencers(
        self,
        agent_id: str,
        criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find social influencers in the network."""
        try:
            min_influence = criteria.get("minInfluence", 0.5)
            min_connections = criteria.get("minConnections", 5)
            max_depth = criteria.get("maxDepth", 2)

            # Get all connections
            connections = await self.social_collection.get_agent_connections(agent_id)

            # Analyze each person's influence
            influencers = []
            person_metrics = {}

            for connection in connections:
                source_id = connection.connection["participants"]["source"]["id"]
                target_id = connection.connection["participants"]["target"]["id"]

                # Calculate metrics for source
                if source_id not in person_metrics:
                    person_connections = [c for c in connections
                                        if c.connection["participants"]["source"]["id"] == source_id or
                                           c.connection["participants"]["target"]["id"] == source_id]

                    if len(person_connections) >= min_connections:
                        avg_strength = sum(c.connection["strength"]["overall"] for c in person_connections) / len(person_connections)
                        influence_score = avg_strength * math.log(len(person_connections) + 1)

                        if influence_score >= min_influence:
                            person_metrics[source_id] = {
                                "personId": source_id,
                                "name": connection.connection["participants"]["source"].get("name", "Unknown"),
                                "influenceScore": influence_score,
                                "connectionCount": len(person_connections),
                                "averageConnectionStrength": avg_strength,
                                "influenceType": self._classify_influence_type(
                                    {"eigenvector": avg_strength}, influence_score
                                )
                            }

            # Sort by influence score
            influencers = sorted(person_metrics.values(), key=lambda x: x["influenceScore"], reverse=True)

            return influencers

        except Exception as error:
            logger.error(f"Error finding influencers: {error}")
            return []

    # Private analysis methods

    async def _analyze_network(self, agent_id: str, connections: List[SocialConnection]) -> NetworkAnalysis:
        """Analyze network structure and metrics."""
        if not connections:
            return NetworkAnalysis(
                total_connections=0,
                active_connections=0,
                network_reach=0,
                average_connection_strength=0.0,
                network_density=0.0,
                clustering_coefficient=0.0
            )

        # Calculate basic metrics
        total_connections = len(connections)
        active_connections = len([c for c in connections if c.connection.get("status") == "active"])

        # Calculate average connection strength
        total_strength = sum(c.connection["strength"]["overall"] for c in connections)
        average_connection_strength = total_strength / total_connections

        # Calculate network reach (unique people)
        unique_people = set()
        for connection in connections:
            unique_people.add(connection.connection["participants"]["source"]["id"])
            unique_people.add(connection.connection["participants"]["target"]["id"])
        network_reach = len(unique_people)

        # Calculate network density
        network_density = self._calculate_network_density(connections)

        # Calculate clustering coefficient
        clustering_coefficient = self._calculate_clustering_coefficient(connections)

        return NetworkAnalysis(
            total_connections=total_connections,
            active_connections=active_connections,
            network_reach=network_reach,
            average_connection_strength=average_connection_strength,
            network_density=network_density,
            clustering_coefficient=clustering_coefficient
        )

    async def _analyze_social_influence(self, agent_id: str, parameters: Dict[str, Any]) -> InfluenceMetrics:
        """Analyze social influence metrics."""
        try:
            # Find influencers in the network
            influencers = await self.find_influencers(agent_id, {
                "minInfluence": 0.3,
                "minConnections": 3,
                "maxDepth": parameters.get("maxDepth", 3)
            })

            # Calculate personal influence (agent's own influence)
            agent_connections = await self.social_collection.collection.find({
                "agentId": agent_id,
                "$or": [
                    {"connection.participants.source.id": agent_id},
                    {"connection.participants.target.id": agent_id}
                ]
            }).to_list(length=None)

            personal_influence = 0.5  # Default
            if agent_connections:
                avg_strength = sum(c["connection"]["strength"]["overall"] for c in agent_connections) / len(agent_connections)
                personal_influence = avg_strength * math.log(len(agent_connections) + 1)

            # Calculate network influence (influence through network)
            network_influence = sum(inf["influenceScore"] for inf in influencers[:5]) / 5 if influencers else 0.3

            # Calculate authority score
            authority_score = (personal_influence + network_influence) / 2

            # Estimate reach
            reach_estimate = len(agent_connections) * 3  # Simplified reach calculation

            # Get top influential connections
            influential_connections = []
            for influencer in influencers[:5]:
                influential_connections.append({
                    "personId": influencer["personId"],
                    "name": influencer["name"],
                    "influenceScore": influencer["influenceScore"],
                    "connectionPath": [agent_id, influencer["personId"]]  # Simplified path
                })

            return InfluenceMetrics(
                personal_influence=personal_influence,
                network_influence=network_influence,
                authority_score=authority_score,
                reach_estimate=reach_estimate,
                influential_connections=influential_connections
            )

        except Exception as error:
            logger.error(f"Error analyzing social influence: {error}")
            return InfluenceMetrics(
                personal_influence=0.5,
                network_influence=0.3,
                authority_score=0.4,
                reach_estimate=10,
                influential_connections=[]
            )

    async def _analyze_relationships(self, agent_id: str, connections: List[SocialConnection]) -> RelationshipAnalysis:
        """Analyze relationship patterns and trends."""
        if not connections:
            return RelationshipAnalysis(
                strong_connections=[],
                weak_connections=[],
                growing_connections=[],
                declining_connections=[],
                mutual_connections=[]
            )

        # Categorize connections by strength
        strong_connections = [c for c in connections if c.connection["strength"]["overall"] > 0.7]
        weak_connections = [c for c in connections if c.connection["strength"]["overall"] < 0.3]

        # Analyze trends (simplified - would need historical data)
        growing_connections = [c for c in connections if c.connection.get("trend", "stable") == "growing"]
        declining_connections = [c for c in connections if c.connection.get("trend", "stable") == "declining"]

        # Find mutual connections (simplified)
        mutual_connections = []
        processed_pairs = set()

        for i, conn1 in enumerate(connections):
            for j, conn2 in enumerate(connections[i+1:], i+1):
                person1 = conn1.connection["participants"]["target"]["id"]
                person2 = conn2.connection["participants"]["target"]["id"]

                if (person1, person2) not in processed_pairs and (person2, person1) not in processed_pairs:
                    # Check if they have mutual connections
                    mutual_friends = await self.find_mutual_connections(agent_id, person1, person2, 2)

                    if mutual_friends:
                        mutual_connections.append({
                            "personId": person1,
                            "mutualFriends": [mf["personId"] for mf in mutual_friends],
                            "connectionStrength": (conn1.connection["strength"]["overall"] + conn2.connection["strength"]["overall"]) / 2
                        })

                    processed_pairs.add((person1, person2))

        return RelationshipAnalysis(
            strong_connections=strong_connections,
            weak_connections=weak_connections,
            growing_connections=growing_connections,
            declining_connections=declining_connections,
            mutual_connections=mutual_connections[:10]  # Limit to top 10
        )

    async def _generate_social_recommendations(
        self,
        agent_id: str,
        network_analysis: NetworkAnalysis,
        influence_metrics: InfluenceMetrics,
        communities: List[Community],
        relationships: RelationshipAnalysis
    ) -> List[SocialRecommendation]:
        """Generate social recommendations based on analysis."""
        recommendations = []

        # Recommendation 1: Strengthen weak connections
        if len(relationships.weak_connections) > 0:
            weak_conn = relationships.weak_connections[0]
            target_person = weak_conn.connection["participants"]["target"]["id"]
            recommendations.append(SocialRecommendation(
                type="strengthen_connection",
                priority="medium",
                description=f"Strengthen connection with {weak_conn.connection['participants']['target'].get('name', 'person')} to improve network stability",
                target_person_id=target_person,
                expected_impact=0.6,
                effort_required="medium",
                timeline="short_term"
            ))

        # Recommendation 2: Expand network if density is low
        if network_analysis.network_density < 0.3:
            recommendations.append(SocialRecommendation(
                type="expand_network",
                priority="high",
                description="Network density is low. Consider connecting with new people in your field",
                target_person_id=None,
                expected_impact=0.8,
                effort_required="high",
                timeline="long_term"
            ))

        # Recommendation 3: Bridge communities
        if len(communities) > 1:
            recommendations.append(SocialRecommendation(
                type="bridge_communities",
                priority="high",
                description="You have access to multiple communities. Consider introducing members to create bridges",
                target_person_id=None,
                expected_impact=0.9,
                effort_required="medium",
                timeline="short_term"
            ))

        # Recommendation 4: Leverage influence
        if influence_metrics.authority_score > 0.7:
            recommendations.append(SocialRecommendation(
                type="leverage_influence",
                priority="medium",
                description="Your authority score is high. Consider mentoring or leading initiatives",
                target_person_id=None,
                expected_impact=0.7,
                effort_required="low",
                timeline="immediate"
            ))

        # Recommendation 5: Nurture growing connections
        if len(relationships.growing_connections) > 0:
            growing_conn = relationships.growing_connections[0]
            target_person = growing_conn.connection["participants"]["target"]["id"]
            recommendations.append(SocialRecommendation(
                type="strengthen_connection",
                priority="high",
                description=f"Connection with {growing_conn.connection['participants']['target'].get('name', 'person')} is growing. Invest more time to solidify",
                target_person_id=target_person,
                expected_impact=0.8,
                effort_required="low",
                timeline="immediate"
            ))

        return recommendations[:5]  # Limit to top 5 recommendations

    # Helper methods

    def _calculate_influence_score(self, person: SocialConnection, depth: int) -> float:
        """Calculate influence score based on connection and depth."""
        base_influence = person.connection.get("network_position", {}).get("influence", {}).get("authority", 0.5)
        depth_penalty = math.pow(0.8, depth)  # Influence decreases with distance
        return base_influence * depth_penalty

    def _calculate_bridge_potential(self, strength1: float, strength2: float) -> float:
        """Calculate bridge potential between two connections."""
        return (strength1 + strength2) / 2 * min(strength1, strength2)

    def _determine_role_in_community(self, community: Dict[str, Any], agent_id: str) -> str:
        """Determine role in community."""
        central_members = community.get("centralMembers", [])
        bridge_members = community.get("bridgeMembers", [])

        if agent_id in central_members:
            return "leader"
        elif agent_id in bridge_members:
            return "bridge"
        else:
            return "member"

    def _classify_influence_type(self, centrality: Dict[str, Any], influence_score: float) -> str:
        """Classify influence type based on metrics."""
        eigenvector = centrality.get("eigenvector", 0.5)

        if eigenvector > 0.7:
            return "authority"
        elif influence_score > 0.8:
            return "connector"
        elif eigenvector > 0.5:
            return "maven"
        else:
            return "salesperson"

    def _calculate_network_density(self, connections: List[SocialConnection]) -> float:
        """Calculate network density."""
        if len(connections) < 2:
            return 0.0

        # Get unique nodes
        unique_nodes = set()
        for connection in connections:
            unique_nodes.add(connection.connection["participants"]["source"]["id"])
            unique_nodes.add(connection.connection["participants"]["target"]["id"])

        n = len(unique_nodes)
        if n < 2:
            return 0.0

        max_possible_edges = n * (n - 1) / 2
        actual_edges = len(connections)

        return actual_edges / max_possible_edges

    def _calculate_clustering_coefficient(self, connections: List[SocialConnection]) -> float:
        """Calculate clustering coefficient."""
        if not connections:
            return 0.0

        # Simplified clustering coefficient calculation
        strong_connections = [c for c in connections if c.connection["strength"]["overall"] > 0.5]
        return len(strong_connections) / len(connections)

    def _calculate_complexity(self, node_count: int, depth: int) -> float:
        """Calculate computational complexity."""
        return math.pow(node_count, depth) / 1000  # Normalized complexity score

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup any resources if needed
        pass
