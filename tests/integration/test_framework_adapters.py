"""
Comprehensive integration tests for all framework adapters.
Tests the AI Brain integration with CrewAI, Pydantic AI, Agno, LangChain, and LangGraph.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from ai_brain_python.core.universal_ai_brain import UniversalAIBrainConfig
from ai_brain_python.storage.storage_manager import StorageConfig
from ai_brain_python.storage.mongodb_client import MongoDBConfig
from ai_brain_python.storage.cache_manager import CacheConfig
from ai_brain_python.storage.vector_store import VectorSearchConfig

# Import adapter functionality
from ai_brain_python.adapters import (
    get_available_frameworks,
    get_registered_adapters,
    create_adapter,
    initialize_adapter,
    get_framework_info,
    get_installation_instructions,
    AdapterManager,
    FRAMEWORK_AVAILABILITY
)


@pytest.fixture
def ai_brain_config():
    """Create test AI Brain configuration."""
    storage_config = StorageConfig(
        mongodb=MongoDBConfig(
            host="localhost",
            port=27017,
            database="test_ai_brain_adapters",
            use_atlas=False
        ),
        redis=CacheConfig(
            host="localhost",
            port=6379,
            database=4
        ),
        vector_search=VectorSearchConfig(
            embedding_dimension=1536
        )
    )
    
    return UniversalAIBrainConfig(
        storage_config=storage_config,
        enable_all_systems=True,
        max_concurrent_processing=3,
        default_timeout=10
    )


@pytest.mark.integration
class TestFrameworkAdapters:
    """Test framework adapter functionality."""
    
    def test_framework_detection(self):
        """Test framework availability detection."""
        available_frameworks = get_available_frameworks()
        registered_adapters = get_registered_adapters()
        
        # Should detect available frameworks
        assert isinstance(available_frameworks, list)
        assert isinstance(registered_adapters, list)
        
        # Registered adapters should be subset of available frameworks
        for adapter in registered_adapters:
            assert adapter in FRAMEWORK_AVAILABILITY
    
    def test_framework_info(self):
        """Test framework information retrieval."""
        # Test getting info for all frameworks
        all_info = get_framework_info()
        
        assert "available_frameworks" in all_info
        assert "registered_adapters" in all_info
        assert "framework_availability" in all_info
        assert "frameworks" in all_info
        
        # Test getting info for specific framework (if any available)
        available_frameworks = get_available_frameworks()
        if available_frameworks:
            framework = available_frameworks[0]
            framework_info = get_framework_info(framework)
            
            assert "framework" in framework_info
            assert "available" in framework_info
            assert "cognitive_features" in framework_info
    
    def test_installation_instructions(self):
        """Test installation instructions for missing frameworks."""
        instructions = get_installation_instructions()
        
        assert isinstance(instructions, dict)
        
        # Should provide instructions for unavailable frameworks
        for framework, available in FRAMEWORK_AVAILABILITY.items():
            if not available:
                assert framework in instructions
                assert "pip install" in instructions[framework]
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_adapter_manager(self, mock_storage_init, ai_brain_config):
        """Test adapter manager functionality."""
        mock_storage_init.return_value = None
        
        manager = AdapterManager(ai_brain_config)
        
        # Test status before initialization
        status = manager.get_status()
        assert status["adapter_count"] == 0
        assert len(status["initialized_adapters"]) == 0
        
        # Test initialization of available adapters
        available_frameworks = get_available_frameworks()
        if available_frameworks:
            # Try to initialize first available framework
            framework = available_frameworks[0]
            
            try:
                adapter = await manager.initialize_adapter(framework)
                assert adapter is not None
                assert framework in manager.initialized_adapters
                
                # Test getting adapter
                retrieved_adapter = manager.get_adapter(framework)
                assert retrieved_adapter is adapter
                
                # Test status after initialization
                status = manager.get_status()
                assert status["adapter_count"] == 1
                assert framework in status["initialized_adapters"]
                
            except ImportError:
                # Framework not actually available
                pass
        
        # Test shutdown
        await manager.shutdown_all()
        assert len(manager.adapters) == 0
        assert len(manager.initialized_adapters) == 0


@pytest.mark.integration
@pytest.mark.skipif(not FRAMEWORK_AVAILABILITY.get("crewai", False), reason="CrewAI not available")
class TestCrewAIAdapter:
    """Test CrewAI adapter integration."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_crewai_adapter_creation(self, mock_storage_init, ai_brain_config):
        """Test CrewAI adapter creation and initialization."""
        mock_storage_init.return_value = None
        
        try:
            adapter = await initialize_adapter("crewai", ai_brain_config)
            
            assert adapter.framework_name == "crewai"
            assert adapter.is_initialized
            
            # Test framework info
            info = adapter.get_framework_info()
            assert info["framework"] == "CrewAI"
            assert "cognitive_features" in info
            
            # Test usage stats
            stats = adapter.get_usage_stats()
            assert "framework" in stats
            assert "is_initialized" in stats
            
            await adapter.shutdown()
            
        except ImportError:
            pytest.skip("CrewAI dependencies not available")
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_cognitive_agent_creation(self, mock_storage_init, ai_brain_config):
        """Test cognitive agent creation."""
        mock_storage_init.return_value = None
        
        try:
            from ai_brain_python.adapters.crewai_adapter import CrewAIAdapter
            
            adapter = CrewAIAdapter(ai_brain_config)
            await adapter.initialize()
            
            # Create cognitive agent
            agent = adapter.create_cognitive_agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory"
            )
            
            assert agent.role == "Test Agent"
            assert agent.goal == "Test goal"
            assert agent.agent_id.startswith("crewai_agent_")
            
            await adapter.shutdown()
            
        except ImportError:
            pytest.skip("CrewAI dependencies not available")


@pytest.mark.integration
@pytest.mark.skipif(not FRAMEWORK_AVAILABILITY.get("pydantic_ai", False), reason="Pydantic AI not available")
class TestPydanticAIAdapter:
    """Test Pydantic AI adapter integration."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_pydantic_ai_adapter_creation(self, mock_storage_init, ai_brain_config):
        """Test Pydantic AI adapter creation and initialization."""
        mock_storage_init.return_value = None
        
        try:
            adapter = await initialize_adapter("pydantic_ai", ai_brain_config)
            
            assert adapter.framework_name == "pydantic_ai"
            assert adapter.is_initialized
            
            # Test framework info
            info = adapter.get_framework_info()
            assert info["framework"] == "Pydantic AI"
            assert info["type_safety"] is True
            
            await adapter.shutdown()
            
        except ImportError:
            pytest.skip("Pydantic AI dependencies not available")


@pytest.mark.integration
@pytest.mark.skipif(not FRAMEWORK_AVAILABILITY.get("langchain", False), reason="LangChain not available")
class TestLangChainAdapter:
    """Test LangChain adapter integration."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_langchain_adapter_creation(self, mock_storage_init, ai_brain_config):
        """Test LangChain adapter creation and initialization."""
        mock_storage_init.return_value = None
        
        try:
            adapter = await initialize_adapter("langchain", ai_brain_config)
            
            assert adapter.framework_name == "langchain"
            assert adapter.is_initialized
            
            # Test framework info
            info = adapter.get_framework_info()
            assert info["framework"] == "LangChain"
            assert "components" in info
            
            await adapter.shutdown()
            
        except ImportError:
            pytest.skip("LangChain dependencies not available")


@pytest.mark.integration
@pytest.mark.skipif(not FRAMEWORK_AVAILABILITY.get("langgraph", False), reason="LangGraph not available")
class TestLangGraphAdapter:
    """Test LangGraph adapter integration."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_langgraph_adapter_creation(self, mock_storage_init, ai_brain_config):
        """Test LangGraph adapter creation and initialization."""
        mock_storage_init.return_value = None
        
        try:
            adapter = await initialize_adapter("langgraph", ai_brain_config)
            
            assert adapter.framework_name == "langgraph"
            assert adapter.is_initialized
            
            # Test framework info
            info = adapter.get_framework_info()
            assert info["framework"] == "LangGraph"
            assert info["stateful"] is True
            
            await adapter.shutdown()
            
        except ImportError:
            pytest.skip("LangGraph dependencies not available")


@pytest.mark.integration
@pytest.mark.skipif(not FRAMEWORK_AVAILABILITY.get("agno", False), reason="Agno not available")
class TestAgnoAdapter:
    """Test Agno adapter integration."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_agno_adapter_creation(self, mock_storage_init, ai_brain_config):
        """Test Agno adapter creation and initialization."""
        mock_storage_init.return_value = None
        
        try:
            adapter = await initialize_adapter("agno", ai_brain_config)
            
            assert adapter.framework_name == "agno"
            assert adapter.is_initialized
            
            # Test framework info
            info = adapter.get_framework_info()
            assert info["framework"] == "Agno"
            assert "cognitive_features" in info
            
            await adapter.shutdown()
            
        except ImportError:
            pytest.skip("Agno dependencies not available")


@pytest.mark.integration
class TestAdapterErrorHandling:
    """Test adapter error handling and edge cases."""
    
    def test_invalid_framework(self, ai_brain_config):
        """Test handling of invalid framework names."""
        with pytest.raises(ValueError, match="Unknown framework"):
            create_adapter("invalid_framework", ai_brain_config)
    
    def test_unavailable_framework(self, ai_brain_config):
        """Test handling of unavailable frameworks."""
        # Find a framework that's not available
        unavailable_frameworks = [
            framework for framework, available in FRAMEWORK_AVAILABILITY.items()
            if not available
        ]
        
        if unavailable_frameworks:
            framework = unavailable_frameworks[0]
            with pytest.raises(ImportError, match="not available"):
                create_adapter(framework, ai_brain_config)
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_adapter_health_check(self, mock_storage_init, ai_brain_config):
        """Test adapter health check functionality."""
        mock_storage_init.return_value = None
        
        available_frameworks = get_available_frameworks()
        if available_frameworks:
            framework = available_frameworks[0]
            
            try:
                adapter = await initialize_adapter(framework, ai_brain_config)
                
                # Test health check
                health = await adapter.health_check()
                
                assert "adapter_name" in health
                assert "is_initialized" in health
                assert "status" in health
                assert health["adapter_name"] == framework
                assert health["is_initialized"] is True
                
                await adapter.shutdown()
                
            except ImportError:
                pytest.skip(f"{framework} dependencies not available")


@pytest.mark.integration
class TestCrossFrameworkCompatibility:
    """Test compatibility and interaction between different framework adapters."""
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_multiple_adapters_simultaneously(self, mock_storage_init, ai_brain_config):
        """Test running multiple framework adapters simultaneously."""
        mock_storage_init.return_value = None
        
        manager = AdapterManager(ai_brain_config)
        available_frameworks = get_available_frameworks()
        
        if len(available_frameworks) >= 2:
            # Initialize multiple adapters
            initialized_adapters = []
            
            for framework in available_frameworks[:2]:  # Test with first 2 available
                try:
                    adapter = await manager.initialize_adapter(framework)
                    initialized_adapters.append((framework, adapter))
                except ImportError:
                    continue
            
            # Verify they can coexist
            if len(initialized_adapters) >= 2:
                for framework, adapter in initialized_adapters:
                    assert adapter.is_initialized
                    
                    # Test basic functionality
                    health = await adapter.health_check()
                    assert health["status"] in ["healthy", "degraded"]
            
            # Cleanup
            await manager.shutdown_all()
    
    @patch('ai_brain_python.storage.storage_manager.StorageManager.initialize')
    async def test_adapter_isolation(self, mock_storage_init, ai_brain_config):
        """Test that adapters are properly isolated from each other."""
        mock_storage_init.return_value = None
        
        available_frameworks = get_available_frameworks()
        
        if len(available_frameworks) >= 2:
            framework1, framework2 = available_frameworks[:2]
            
            try:
                # Create two separate adapters
                adapter1 = await initialize_adapter(framework1, ai_brain_config)
                adapter2 = await initialize_adapter(framework2, ai_brain_config)
                
                # Verify they have different identities
                assert adapter1.framework_name != adapter2.framework_name
                
                # Verify they maintain separate state
                stats1 = adapter1.get_usage_stats()
                stats2 = adapter2.get_usage_stats()
                
                assert stats1["framework"] != stats2["framework"]
                
                # Cleanup
                await adapter1.shutdown()
                await adapter2.shutdown()
                
            except ImportError:
                pytest.skip("Required framework dependencies not available")
