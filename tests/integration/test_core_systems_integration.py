"""
Integration tests for core cognitive systems.
Tests the interaction between the Universal AI Brain and the first 8 cognitive systems.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext
from ai_brain_python.storage.storage_manager import StorageConfig
from ai_brain_python.storage.mongodb_client import MongoDBConfig
from ai_brain_python.storage.cache_manager import CacheConfig
from ai_brain_python.storage.vector_store import VectorSearchConfig

# Import core cognitive systems
from ai_brain_python.core.cognitive_systems.emotional_intelligence import EmotionalIntelligenceEngine
from ai_brain_python.core.cognitive_systems.goal_hierarchy import GoalHierarchyManager
from ai_brain_python.core.cognitive_systems.confidence_tracking import ConfidenceTrackingEngine
from ai_brain_python.core.cognitive_systems.attention_management import AttentionManagementSystem
from ai_brain_python.core.cognitive_systems.cultural_knowledge import CulturalKnowledgeEngine
from ai_brain_python.core.cognitive_systems.skill_capability import SkillCapabilityManager
from ai_brain_python.core.cognitive_systems.communication_protocol import CommunicationProtocolManager
from ai_brain_python.core.cognitive_systems.temporal_planning import TemporalPlanningEngine


@pytest.fixture
def storage_config():
    """Create test storage configuration."""
    return StorageConfig(
        mongodb=MongoDBConfig(
            host="localhost",
            port=27017,
            database="test_ai_brain_integration",
            use_atlas=False
        ),
        redis=CacheConfig(
            host="localhost",
            port=6379,
            database=2
        ),
        vector_search=VectorSearchConfig(
            embedding_dimension=1536
        )
    )


@pytest.fixture
def ai_brain_config(storage_config):
    """Create test AI Brain configuration."""
    return UniversalAIBrainConfig(
        storage_config=storage_config,
        enable_all_systems=True,
        max_concurrent_processing=3,
        default_timeout=5,
        enable_monitoring=True,
        enable_safety_checks=True
    )


@pytest.fixture
def sample_emotional_input():
    """Create sample input with emotional content."""
    context = CognitiveContext(
        user_id="test_user_emotional",
        session_id="test_session_emotional"
    )
    
    return CognitiveInputData(
        text="I'm feeling really excited about this new AI project! It's going to be amazing.",
        input_type="text",
        context=context,
        requested_systems=["emotional_intelligence", "communication_protocol"],
        processing_priority=8
    )


@pytest.fixture
def sample_goal_input():
    """Create sample input with goal-related content."""
    context = CognitiveContext(
        user_id="test_user_goals",
        session_id="test_session_goals"
    )
    
    return CognitiveInputData(
        text="I want to learn Python programming and complete this AI project by next month.",
        input_type="text",
        context=context,
        requested_systems=["goal_hierarchy", "temporal_planning", "skill_capability"],
        processing_priority=7
    )


@pytest.mark.integration
@pytest.mark.asyncio
class TestCoreSystemsIntegration:
    """Test integration between core cognitive systems."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_emotional_intelligence_integration(self, mock_store_doc, mock_storage_init, ai_brain_config, sample_emotional_input):
        """Test emotional intelligence system integration."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_123"
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(sample_emotional_input)
            
            # Verify response structure
            assert response.success is True
            assert response.confidence > 0.0
            assert "emotional_intelligence" in response.cognitive_results
            assert "communication_protocol" in response.cognitive_results
            
            # Verify emotional intelligence results
            emotional_result = response.cognitive_results["emotional_intelligence"]
            assert emotional_result["status"] == "completed"
            assert "emotional_state" in emotional_result
            assert "empathy_response" in emotional_result
            
            # Verify emotional state structure
            emotional_state = emotional_result["emotional_state"]
            assert "primary_emotion" in emotional_state
            assert "emotion_intensity" in emotional_state
            assert "valence" in emotional_state
            assert "arousal" in emotional_state
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_goal_hierarchy_integration(self, mock_store_doc, mock_storage_init, ai_brain_config, sample_goal_input):
        """Test goal hierarchy system integration."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_456"
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(sample_goal_input)
            
            # Verify response structure
            assert response.success is True
            assert "goal_hierarchy" in response.cognitive_results
            assert "temporal_planning" in response.cognitive_results
            assert "skill_capability" in response.cognitive_results
            
            # Verify goal hierarchy results
            goal_result = response.cognitive_results["goal_hierarchy"]
            assert goal_result["status"] == "completed"
            assert "goal_hierarchy" in goal_result
            assert "extracted_goals" in goal_result
            assert "recommendations" in goal_result
            
            # Verify extracted goals
            extracted_goals = goal_result["extracted_goals"]
            assert isinstance(extracted_goals, list)
            if extracted_goals:
                goal = extracted_goals[0]
                assert "text" in goal
                assert "priority" in goal
                assert "type" in goal
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_individual_cognitive_systems(self, mock_storage_init, ai_brain_config):
        """Test individual cognitive systems can be initialized and used."""
        mock_storage_init.return_value = None
        
        # Test each cognitive system individually
        systems_to_test = [
            EmotionalIntelligenceEngine,
            GoalHierarchyManager,
            ConfidenceTrackingEngine,
            AttentionManagementSystem,
            CulturalKnowledgeEngine,
            SkillCapabilityManager,
            CommunicationProtocolManager,
            TemporalPlanningEngine
        ]
        
        for system_class in systems_to_test:
            system = system_class()
            
            # Test initialization
            await system.initialize()
            assert system.is_initialized
            
            # Test basic properties
            assert system.system_name
            assert system.system_description
            assert len(system.required_capabilities) > 0
            assert len(system.provided_capabilities) > 0
            
            # Test health check
            health = await system.health_check()
            assert health["status"] == "healthy"
            assert health["initialized"] is True
            
            # Test state management
            state = await system.get_state()
            assert state.system_type is not None
            assert state.is_active is True
            
            # Test shutdown
            await system.shutdown()
            assert not system.is_initialized
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_confidence_tracking_integration(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test confidence tracking across multiple systems."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_789"
        
        context = CognitiveContext(
            user_id="test_user_confidence",
            session_id="test_session_confidence"
        )
        
        input_data = CognitiveInputData(
            text="I'm not sure about this approach, but I think it might work.",
            input_type="text",
            context=context,
            requested_systems=["confidence_tracking", "emotional_intelligence"],
            processing_priority=6
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(input_data)
            
            # Verify confidence tracking results
            assert "confidence_tracking" in response.cognitive_results
            confidence_result = response.cognitive_results["confidence_tracking"]
            
            assert "confidence_assessment" in confidence_result
            confidence_assessment = confidence_result["confidence_assessment"]
            
            assert "overall_confidence" in confidence_assessment
            assert "confidence_level" in confidence_assessment
            assert "epistemic_uncertainty" in confidence_assessment
            assert "aleatoric_uncertainty" in confidence_assessment
            
            # Verify confidence values are in valid ranges
            assert 0.0 <= confidence_assessment["overall_confidence"] <= 1.0
            assert 0.0 <= confidence_assessment["epistemic_uncertainty"] <= 1.0
            assert 0.0 <= confidence_assessment["aleatoric_uncertainty"] <= 1.0
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_attention_management_integration(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test attention management system integration."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_attention"
        
        context = CognitiveContext(
            user_id="test_user_attention",
            session_id="test_session_attention"
        )
        
        input_data = CognitiveInputData(
            text="I need to focus on this complex algorithm implementation while also reviewing the documentation and planning the next sprint.",
            input_type="text",
            context=context,
            requested_systems=["attention_management", "temporal_planning"],
            processing_priority=8
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(input_data)
            
            # Verify attention management results
            assert "attention_management" in response.cognitive_results
            attention_result = response.cognitive_results["attention_management"]
            
            assert "attention_state" in attention_result
            attention_state = attention_result["attention_state"]
            
            assert "attention_type" in attention_state
            assert "focus_level" in attention_state
            assert "cognitive_load" in attention_state
            assert "distraction_level" in attention_state
            
            # Verify attention allocation
            assert "attention_allocation" in attention_result
            attention_allocation = attention_result["attention_allocation"]
            assert isinstance(attention_allocation, dict)
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_cultural_knowledge_integration(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test cultural knowledge system integration."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_cultural"
        
        context = CognitiveContext(
            user_id="test_user_cultural",
            session_id="test_session_cultural"
        )
        
        input_data = CognitiveInputData(
            text="Please provide a formal response for our business meeting with the Japanese clients.",
            input_type="text",
            language="en",
            context=context,
            requested_systems=["cultural_knowledge", "communication_protocol"],
            processing_priority=7
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(input_data)
            
            # Verify cultural knowledge results
            assert "cultural_knowledge" in response.cognitive_results
            cultural_result = response.cognitive_results["cultural_knowledge"]
            
            assert "cultural_context" in cultural_result
            cultural_context = cultural_result["cultural_context"]
            
            assert "primary_culture" in cultural_context
            assert "cultural_dimensions" in cultural_context
            assert "adaptation_level" in cultural_context
            assert "cultural_sensitivity" in cultural_context
            
            # Verify communication adaptation
            assert "communication_adaptation" in cultural_result
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_system_coordination(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test coordination between multiple cognitive systems."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_coordination"
        
        context = CognitiveContext(
            user_id="test_user_coordination",
            session_id="test_session_coordination"
        )
        
        input_data = CognitiveInputData(
            text="I'm feeling overwhelmed with my learning goals. I want to master Python programming, improve my communication skills, and complete this project by next month, but I'm not confident I can manage it all.",
            input_type="text",
            context=context,
            requested_systems=[
                "emotional_intelligence",
                "goal_hierarchy", 
                "confidence_tracking",
                "attention_management",
                "skill_capability",
                "temporal_planning"
            ],
            processing_priority=9
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(input_data)
            
            # Verify all requested systems responded
            requested_systems = input_data.requested_systems
            for system in requested_systems:
                assert system in response.cognitive_results
                system_result = response.cognitive_results[system]
                assert system_result["status"] == "completed"
                assert system_result["confidence"] > 0.0
            
            # Verify system coordination
            # Emotional intelligence should detect overwhelm
            emotional_result = response.cognitive_results["emotional_intelligence"]
            assert "emotional_state" in emotional_result
            
            # Goal hierarchy should extract multiple goals
            goal_result = response.cognitive_results["goal_hierarchy"]
            assert "extracted_goals" in goal_result
            
            # Confidence tracking should detect low confidence
            confidence_result = response.cognitive_results["confidence_tracking"]
            assert "confidence_assessment" in confidence_result
            
            # Attention management should detect high cognitive load
            attention_result = response.cognitive_results["attention_management"]
            assert "cognitive_load" in attention_result
            
            # Skill capability should identify learning goals
            skill_result = response.cognitive_results["skill_capability"]
            assert "skill_analysis" in skill_result
            
            # Temporal planning should identify time constraints
            temporal_result = response.cognitive_results["temporal_planning"]
            assert "temporal_analysis" in temporal_result
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_system_error_handling(self, mock_storage_init, ai_brain_config):
        """Test error handling in cognitive systems."""
        mock_storage_init.return_value = None
        
        context = CognitiveContext(
            user_id="test_user_error",
            session_id="test_session_error"
        )
        
        # Test with empty input
        empty_input = CognitiveInputData(
            text="",
            input_type="text",
            context=context,
            requested_systems=["emotional_intelligence"],
            processing_priority=5
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(empty_input)
            
            # Should still complete successfully even with empty input
            assert response.success is True
            assert "emotional_intelligence" in response.cognitive_results
            
            # System should handle empty input gracefully
            emotional_result = response.cognitive_results["emotional_intelligence"]
            assert emotional_result["status"] == "completed"
