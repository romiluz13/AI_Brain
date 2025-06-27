"""
Comprehensive integration tests for all 16 cognitive systems.
Tests the complete AI Brain with all cognitive systems working together.
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

# Import all 16 cognitive systems
from ai_brain_python.core.cognitive_systems.emotional_intelligence import EmotionalIntelligenceEngine
from ai_brain_python.core.cognitive_systems.goal_hierarchy import GoalHierarchyManager
from ai_brain_python.core.cognitive_systems.confidence_tracking import ConfidenceTrackingEngine
from ai_brain_python.core.cognitive_systems.attention_management import AttentionManagementSystem
from ai_brain_python.core.cognitive_systems.cultural_knowledge import CulturalKnowledgeEngine
from ai_brain_python.core.cognitive_systems.skill_capability import SkillCapabilityManager
from ai_brain_python.core.cognitive_systems.communication_protocol import CommunicationProtocolManager
from ai_brain_python.core.cognitive_systems.temporal_planning import TemporalPlanningEngine
from ai_brain_python.core.cognitive_systems.semantic_memory import SemanticMemoryEngine
from ai_brain_python.core.cognitive_systems.safety_guardrails import SafetyGuardrailsEngine
from ai_brain_python.core.cognitive_systems.self_improvement import SelfImprovementEngine
from ai_brain_python.core.cognitive_systems.monitoring import MonitoringEngine
from ai_brain_python.core.cognitive_systems.tool_interface import AdvancedToolInterface
from ai_brain_python.core.cognitive_systems.workflow_orchestration import WorkflowOrchestrationEngine
from ai_brain_python.core.cognitive_systems.multimodal_processing import MultiModalProcessingEngine
from ai_brain_python.core.cognitive_systems.human_feedback import HumanFeedbackIntegrationEngine


@pytest.fixture
def storage_config():
    """Create test storage configuration."""
    return StorageConfig(
        mongodb=MongoDBConfig(
            host="localhost",
            port=27017,
            database="test_ai_brain_complete",
            use_atlas=False
        ),
        redis=CacheConfig(
            host="localhost",
            port=6379,
            database=3
        ),
        vector_search=VectorSearchConfig(
            embedding_dimension=1536
        )
    )


@pytest.fixture
def ai_brain_config(storage_config):
    """Create test AI Brain configuration with all systems enabled."""
    return UniversalAIBrainConfig(
        storage_config=storage_config,
        enable_all_systems=True,
        max_concurrent_processing=5,
        default_timeout=10,
        enable_monitoring=True,
        enable_safety_checks=True
    )


@pytest.fixture
def comprehensive_input():
    """Create comprehensive input that engages all systems."""
    context = CognitiveContext(
        user_id="test_user_comprehensive",
        session_id="test_session_comprehensive"
    )
    
    return CognitiveInputData(
        text="I'm feeling excited but also nervous about my goal to learn AI programming and complete this complex project by next month. I want to improve my technical skills while maintaining good communication with my international team. Please help me plan this carefully and safely, and I'd like to use tools to calculate the time needed and search for resources.",
        input_type="text",
        language="en",
        context=context,
        requested_systems=[
            "emotional_intelligence",
            "goal_hierarchy", 
            "confidence_tracking",
            "attention_management",
            "cultural_knowledge",
            "skill_capability",
            "communication_protocol",
            "temporal_planning",
            "semantic_memory",
            "safety_guardrails",
            "self_improvement",
            "monitoring",
            "tool_interface",
            "workflow_orchestration",
            "multimodal_processing",
            "human_feedback"
        ],
        processing_priority=9
    )


@pytest.mark.integration
@pytest.mark.asyncio
class TestAllCognitiveSystems:
    """Test all 16 cognitive systems working together."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_all_systems_initialization(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test that all 16 cognitive systems can be initialized."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_all_systems"
        
        # Test individual system initialization
        all_systems = [
            EmotionalIntelligenceEngine,
            GoalHierarchyManager,
            ConfidenceTrackingEngine,
            AttentionManagementSystem,
            CulturalKnowledgeEngine,
            SkillCapabilityManager,
            CommunicationProtocolManager,
            TemporalPlanningEngine,
            SemanticMemoryEngine,
            SafetyGuardrailsEngine,
            SelfImprovementEngine,
            MonitoringEngine,
            AdvancedToolInterface,
            WorkflowOrchestrationEngine,
            MultiModalProcessingEngine,
            HumanFeedbackIntegrationEngine
        ]
        
        for system_class in all_systems:
            system = system_class()
            
            # Test initialization
            await system.initialize()
            assert system.is_initialized
            
            # Test basic properties
            assert system.system_name
            assert system.system_description
            assert len(system.required_capabilities) >= 0
            assert len(system.provided_capabilities) > 0
            
            # Test health check
            health = await system.health_check()
            assert health["status"] == "healthy"
            
            # Test shutdown
            await system.shutdown()
            assert not system.is_initialized
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_comprehensive_processing(self, mock_store_doc, mock_storage_init, ai_brain_config, comprehensive_input):
        """Test comprehensive processing with all systems engaged."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_comprehensive"
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(comprehensive_input)
            
            # Verify response structure
            assert response.success is True
            assert response.confidence > 0.0
            assert len(response.cognitive_results) > 0
            
            # Verify all requested systems responded
            requested_systems = comprehensive_input.requested_systems
            for system in requested_systems:
                assert system in response.cognitive_results
                system_result = response.cognitive_results[system]
                assert system_result["status"] == "completed"
                assert system_result["confidence"] > 0.0
            
            # Verify system-specific results
            
            # Emotional Intelligence
            emotional_result = response.cognitive_results["emotional_intelligence"]
            assert "emotional_state" in emotional_result
            assert "empathy_response" in emotional_result
            
            # Goal Hierarchy
            goal_result = response.cognitive_results["goal_hierarchy"]
            assert "extracted_goals" in goal_result
            assert "goal_hierarchy" in goal_result
            
            # Confidence Tracking
            confidence_result = response.cognitive_results["confidence_tracking"]
            assert "confidence_assessment" in confidence_result
            
            # Attention Management
            attention_result = response.cognitive_results["attention_management"]
            assert "attention_state" in attention_result
            assert "attention_allocation" in attention_result
            
            # Cultural Knowledge
            cultural_result = response.cognitive_results["cultural_knowledge"]
            assert "cultural_context" in cultural_result
            
            # Skill Capability
            skill_result = response.cognitive_results["skill_capability"]
            assert "skill_analysis" in skill_result
            
            # Communication Protocol
            comm_result = response.cognitive_results["communication_protocol"]
            assert "communication_protocol" in comm_result
            
            # Temporal Planning
            temporal_result = response.cognitive_results["temporal_planning"]
            assert "temporal_analysis" in temporal_result
            
            # Semantic Memory
            memory_result = response.cognitive_results["semantic_memory"]
            assert "memory_operations" in memory_result
            
            # Safety Guardrails
            safety_result = response.cognitive_results["safety_guardrails"]
            assert "safety_assessment" in safety_result
            
            # Self Improvement
            improvement_result = response.cognitive_results["self_improvement"]
            assert "performance_analysis" in improvement_result
            
            # Monitoring
            monitoring_result = response.cognitive_results["monitoring"]
            assert "monitoring_metrics" in monitoring_result
            
            # Tool Interface
            tool_result = response.cognitive_results["tool_interface"]
            assert "tool_analysis" in tool_result
            
            # Workflow Orchestration
            workflow_result = response.cognitive_results["workflow_orchestration"]
            assert "workflow_analysis" in workflow_result
            
            # Multimodal Processing
            multimodal_result = response.cognitive_results["multimodal_processing"]
            assert "multimodal_analysis" in multimodal_result
            
            # Human Feedback
            feedback_result = response.cognitive_results["human_feedback"]
            assert "feedback_analysis" in feedback_result
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_system_coordination_and_data_flow(self, mock_store_doc, mock_storage_init, ai_brain_config, comprehensive_input):
        """Test coordination and data flow between systems."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_coordination"
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(comprehensive_input)
            
            # Verify system coordination
            assert response.success is True
            
            # Check that systems are sharing context appropriately
            cognitive_results = response.cognitive_results
            
            # Safety should be checked first
            safety_result = cognitive_results["safety_guardrails"]
            assert safety_result["safety_assessment"]["is_safe"] is True
            
            # Emotional intelligence should detect mixed emotions
            emotional_result = cognitive_results["emotional_intelligence"]
            emotional_state = emotional_result["emotional_state"]
            assert "primary_emotion" in emotional_state
            
            # Goal hierarchy should extract learning goals
            goal_result = cognitive_results["goal_hierarchy"]
            extracted_goals = goal_result["extracted_goals"]
            assert len(extracted_goals) > 0
            
            # Temporal planning should identify time constraints
            temporal_result = cognitive_results["temporal_planning"]
            assert "temporal_analysis" in temporal_result
            
            # Workflow orchestration should coordinate multiple systems
            workflow_result = cognitive_results["workflow_orchestration"]
            workflow_plan = workflow_result["workflow_plan"]
            assert workflow_plan["total_tasks"] > 0
            
            # Monitoring should track all system performance
            monitoring_result = cognitive_results["monitoring"]
            performance_summary = monitoring_result["performance_summary"]
            assert "overall_health" in performance_summary
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_error_handling_and_resilience(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test error handling and system resilience."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_error_handling"
        
        # Test with problematic input
        context = CognitiveContext(
            user_id="test_user_error",
            session_id="test_session_error"
        )
        
        problematic_input = CognitiveInputData(
            text="",  # Empty text
            input_type="text",
            context=context,
            requested_systems=["emotional_intelligence", "goal_hierarchy"],
            processing_priority=5
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(problematic_input)
            
            # Should still complete successfully even with empty input
            assert response.success is True
            
            # Systems should handle empty input gracefully
            for system_id, result in response.cognitive_results.items():
                assert result["status"] in ["completed", "warning"]
                assert "confidence" in result
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_performance_and_scalability(self, mock_store_doc, mock_storage_init, ai_brain_config, comprehensive_input):
        """Test performance and scalability with multiple concurrent requests."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_performance"
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            # Process multiple requests concurrently
            tasks = []
            for i in range(5):  # 5 concurrent requests
                # Create unique input for each request
                context = CognitiveContext(
                    user_id=f"test_user_perf_{i}",
                    session_id=f"test_session_perf_{i}"
                )
                
                input_data = CognitiveInputData(
                    text=f"Request {i}: {comprehensive_input.text}",
                    input_type="text",
                    context=context,
                    requested_systems=["emotional_intelligence", "goal_hierarchy", "safety_guardrails"],
                    processing_priority=7
                )
                
                tasks.append(brain.process_input(input_data))
            
            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks)
            
            # Verify all requests completed successfully
            for i, response in enumerate(responses):
                assert response.success is True
                assert len(response.cognitive_results) > 0
                assert response.processing_time_ms > 0
                
                # Verify each response is unique (different user context)
                assert response.user_id == f"test_user_perf_{i}"
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_multimodal_capabilities(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test multimodal processing capabilities."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_multimodal"
        
        context = CognitiveContext(
            user_id="test_user_multimodal",
            session_id="test_session_multimodal"
        )
        
        # Create multimodal input
        multimodal_input = CognitiveInputData(
            text="Please analyze this image and audio content",
            input_type="multimodal",
            image_data=b"fake_image_data_for_testing",
            audio_data=b"fake_audio_data_for_testing",
            context=context,
            requested_systems=["multimodal_processing", "semantic_memory", "safety_guardrails"],
            processing_priority=8
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(multimodal_input)
            
            # Verify multimodal processing
            assert response.success is True
            assert "multimodal_processing" in response.cognitive_results
            
            multimodal_result = response.cognitive_results["multimodal_processing"]
            multimodal_analysis = multimodal_result["multimodal_analysis"]
            
            # Should detect multiple modalities
            assert len(multimodal_analysis["detected_modalities"]) > 1
            assert "text" in multimodal_analysis["detected_modalities"]
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_learning_and_adaptation(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test learning and adaptation capabilities."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_learning"
        
        context = CognitiveContext(
            user_id="test_user_learning",
            session_id="test_session_learning"
        )
        
        learning_input = CognitiveInputData(
            text="I want to improve my performance and learn from feedback",
            input_type="text",
            context=context,
            requested_systems=["self_improvement", "human_feedback", "skill_capability"],
            processing_priority=7
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(learning_input)
            
            # Verify learning systems
            assert response.success is True
            
            # Self-improvement should provide recommendations
            improvement_result = response.cognitive_results["self_improvement"]
            assert "learning_recommendations" in improvement_result
            
            # Human feedback should analyze feedback requirements
            feedback_result = response.cognitive_results["human_feedback"]
            assert "feedback_analysis" in feedback_result
            
            # Skill capability should track learning progress
            skill_result = response.cognitive_results["skill_capability"]
            assert "skill_analysis" in skill_result
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_safety_and_compliance(self, mock_store_doc, mock_storage_init, ai_brain_config):
        """Test safety and compliance across all systems."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_safety"
        
        context = CognitiveContext(
            user_id="test_user_safety",
            session_id="test_session_safety"
        )
        
        # Test with potentially sensitive content
        safety_input = CognitiveInputData(
            text="Please help me with this sensitive task that requires careful handling",
            input_type="text",
            context=context,
            requested_systems=["safety_guardrails", "confidence_tracking", "human_feedback"],
            processing_priority=9
        )
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            response = await brain.process_input(safety_input)
            
            # Verify safety processing
            assert response.success is True
            
            # Safety guardrails should assess content
            safety_result = response.cognitive_results["safety_guardrails"]
            safety_assessment = safety_result["safety_assessment"]
            assert "is_safe" in safety_assessment
            assert "risk_score" in safety_assessment
            
            # Confidence tracking should assess uncertainty
            confidence_result = response.cognitive_results["confidence_tracking"]
            confidence_assessment = confidence_result["confidence_assessment"]
            assert "overall_confidence" in confidence_assessment
            
            # Human feedback should determine if approval is needed
            feedback_result = response.cognitive_results["human_feedback"]
            feedback_analysis = feedback_result["feedback_analysis"]
            assert "approval_needed" in feedback_analysis
