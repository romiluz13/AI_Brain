"""
Unit tests for Universal AI Brain core functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ai_brain_python.core.universal_ai_brain import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models.base_models import CognitiveInputData, CognitiveContext, ProcessingStatus
from ai_brain_python.storage.storage_manager import StorageConfig
from ai_brain_python.storage.mongodb_client import MongoDBConfig
from ai_brain_python.storage.cache_manager import CacheConfig
from ai_brain_python.storage.vector_store import VectorSearchConfig


@pytest.fixture
def storage_config():
    """Create test storage configuration."""
    return StorageConfig(
        mongodb=MongoDBConfig(
            host="localhost",
            port=27017,
            database="test_ai_brain",
            use_atlas=False  # Use local for testing
        ),
        redis=CacheConfig(
            host="localhost",
            port=6379,
            database=1  # Use test database
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
        max_concurrent_processing=5,
        default_timeout=10,
        enable_monitoring=True,
        enable_safety_checks=True
    )


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    context = CognitiveContext(
        user_id="test_user_123",
        session_id="test_session_456"
    )
    
    return CognitiveInputData(
        text="I'm feeling excited about this new AI project!",
        input_type="text",
        context=context,
        requested_systems=["emotional_intelligence", "goal_hierarchy"],
        processing_priority=7
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestUniversalAIBrain:
    """Test Universal AI Brain functionality."""
    
    async def test_initialization(self, ai_brain_config):
        """Test AI Brain initialization."""
        brain = UniversalAIBrain(ai_brain_config)
        
        # Check initial state
        assert not brain._is_initialized
        assert brain.config == ai_brain_config
        assert len(brain._enabled_systems) == 16  # All systems enabled
        assert brain._processing_stats["total_requests"] == 0
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_initialize_success(self, mock_storage_init, ai_brain_config):
        """Test successful initialization."""
        mock_storage_init.return_value = None
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        assert brain._is_initialized
        assert len(brain._cognitive_systems) == 16
        mock_storage_init.assert_called_once()
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_initialize_failure(self, mock_storage_init, ai_brain_config):
        """Test initialization failure."""
        mock_storage_init.side_effect = Exception("Storage initialization failed")
        
        brain = UniversalAIBrain(ai_brain_config)
        
        with pytest.raises(Exception, match="Storage initialization failed"):
            await brain.initialize()
        
        assert not brain._is_initialized
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.shutdown')
    async def test_shutdown(self, mock_storage_shutdown, mock_storage_init, ai_brain_config):
        """Test AI Brain shutdown."""
        mock_storage_init.return_value = None
        mock_storage_shutdown.return_value = None
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        assert brain._is_initialized
        
        await brain.shutdown()
        
        assert not brain._is_initialized
        mock_storage_shutdown.assert_called_once()
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_process_input_not_initialized(self, mock_storage_init, ai_brain_config, sample_input_data):
        """Test processing input when not initialized."""
        brain = UniversalAIBrain(ai_brain_config)
        
        with pytest.raises(RuntimeError, match="AI Brain not initialized"):
            await brain.process_input(sample_input_data)
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_document')
    async def test_process_input_success(self, mock_store_doc, mock_storage_init, ai_brain_config, sample_input_data):
        """Test successful input processing."""
        mock_storage_init.return_value = None
        mock_store_doc.return_value = "doc_id_123"
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        response = await brain.process_input(sample_input_data)
        
        # Check response
        assert response.status == ProcessingStatus.COMPLETED
        assert response.success is True
        assert response.confidence >= 0.0
        assert response.processing_metadata.processing_time_ms > 0
        assert len(response.cognitive_results) > 0
        
        # Check stats updated
        assert brain._processing_stats["total_requests"] == 1
        assert brain._processing_stats["successful_requests"] == 1
        assert brain._processing_stats["failed_requests"] == 0
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_process_input_with_requested_systems(self, mock_storage_init, ai_brain_config, sample_input_data):
        """Test processing with specific requested systems."""
        mock_storage_init.return_value = None
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        # Request only specific systems
        requested_systems = ["emotional_intelligence", "safety_guardrails"]
        response = await brain.process_input(sample_input_data, requested_systems=requested_systems)
        
        assert response.status == ProcessingStatus.COMPLETED
        assert response.success is True
        
        # Check that only requested systems were engaged
        engaged_systems = set(response.cognitive_results.keys())
        assert engaged_systems.issubset(set(requested_systems))
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_process_input_timeout(self, mock_storage_init, ai_brain_config, sample_input_data):
        """Test processing timeout."""
        mock_storage_init.return_value = None
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        # Mock a system that takes too long
        original_process = brain._process_single_system
        
        async def slow_process(system_name, input_data):
            if system_name == "emotional_intelligence":
                await asyncio.sleep(2)  # Longer than timeout
            return await original_process(system_name, input_data)
        
        brain._process_single_system = slow_process
        
        response = await brain.process_input(sample_input_data, timeout=1)  # 1 second timeout
        
        # Should still complete but may have some systems that timed out
        assert response.processing_metadata.processing_time_ms > 0
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.get_cognitive_state')
    async def test_get_system_state(self, mock_get_state, mock_storage_init, ai_brain_config):
        """Test getting system state."""
        mock_storage_init.return_value = None
        mock_get_state.return_value = {"test_state": "value"}
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        state = await brain.get_system_state("emotional_intelligence", "test_user")
        
        assert state == {"test_state": "value"}
        mock_get_state.assert_called_once_with("emotional_intelligence", "test_user")
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.store_cognitive_state')
    async def test_update_system_state(self, mock_store_state, mock_storage_init, ai_brain_config):
        """Test updating system state."""
        mock_storage_init.return_value = None
        mock_store_state.return_value = "state_id_123"
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        new_state = {"emotion": "happy", "confidence": 0.9}
        success = await brain.update_system_state("emotional_intelligence", new_state, "test_user")
        
        assert success is True
        mock_store_state.assert_called_once_with("emotional_intelligence", "test_user", new_state)
        
        # Check state is cached
        cache_key = "emotional_intelligence:test_user"
        assert brain._system_states[cache_key] == new_state
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.get_statistics')
    async def test_get_performance_stats(self, mock_get_stats, mock_storage_init, ai_brain_config):
        """Test getting performance statistics."""
        mock_storage_init.return_value = None
        mock_get_stats.return_value = {"storage_stat": "value"}
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        # Update some stats
        brain._processing_stats["total_requests"] = 10
        brain._processing_stats["successful_requests"] = 8
        brain._processing_stats["failed_requests"] = 2
        
        stats = await brain.get_performance_stats()
        
        assert stats["total_requests"] == 10
        assert stats["successful_requests"] == 8
        assert stats["failed_requests"] == 2
        assert stats["enabled_systems"] == list(brain._enabled_systems)
        assert stats["total_systems"] == 16
        assert "storage" in stats
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    @patch('ai_brain_python.storage.storage_manager.StorageManager.health_check')
    async def test_health_check(self, mock_storage_health, mock_storage_init, ai_brain_config):
        """Test health check."""
        mock_storage_init.return_value = None
        mock_storage_health.return_value = {"status": "healthy"}
        
        brain = UniversalAIBrain(ai_brain_config)
        await brain.initialize()
        
        health = await brain.health_check()
        
        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert "components" in health
        assert "storage" in health["components"]
        assert "cognitive_systems" in health["components"]
        assert "performance" in health["components"]
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_context_manager(self, mock_storage_init, ai_brain_config):
        """Test async context manager."""
        mock_storage_init.return_value = None
        
        async with UniversalAIBrain(ai_brain_config) as brain:
            assert brain._is_initialized
            assert isinstance(brain, UniversalAIBrain)
        
        # Should be shutdown after context exit
        assert not brain._is_initialized
    
    def test_determine_systems_to_process(self, ai_brain_config, sample_input_data):
        """Test system selection logic."""
        brain = UniversalAIBrain(ai_brain_config)
        
        # Test with explicit requested systems
        requested = ["emotional_intelligence", "goal_hierarchy"]
        systems = brain._determine_systems_to_process(sample_input_data, requested)
        assert systems == set(requested)
        
        # Test auto-determination
        systems = brain._determine_systems_to_process(sample_input_data, None)
        assert "safety_guardrails" in systems  # Always included
        assert "monitoring" in systems  # Always included
        assert "emotional_intelligence" in systems  # Text input
        assert "semantic_memory" in systems  # Text input
        assert "goal_hierarchy" in systems  # Has user_id
    
    async def test_safety_check(self, ai_brain_config):
        """Test safety check functionality."""
        brain = UniversalAIBrain(ai_brain_config)
        
        # Test normal input
        context = CognitiveContext(user_id="test_user")
        normal_input = CognitiveInputData(text="Hello world", context=context)
        
        result = await brain._perform_safety_check(normal_input)
        assert result.is_valid is True
        assert len(result.violations) == 0
        
        # Test input that's too long
        long_text = "x" * 200000  # Exceeds 100k limit
        long_input = CognitiveInputData(text=long_text, context=context)
        
        result = await brain._perform_safety_check(long_input)
        assert result.is_valid is False
        assert len(result.violations) > 0
    
    def test_config_validation(self, storage_config):
        """Test configuration validation."""
        # Test with specific enabled systems
        config = UniversalAIBrainConfig(
            storage_config=storage_config,
            enable_all_systems=False,
            enabled_systems={"emotional_intelligence", "safety_guardrails", "invalid_system"}
        )
        
        brain = UniversalAIBrain(config)
        
        # Should only include valid systems
        assert "emotional_intelligence" in brain._enabled_systems
        assert "safety_guardrails" in brain._enabled_systems
        assert "invalid_system" not in brain._enabled_systems
        assert len(brain._enabled_systems) == 2
