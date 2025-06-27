"""
Pytest configuration and fixtures for AI Brain Python tests.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

# AI Brain imports
from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext
from ai_brain_python.database.mongodb_client import MongoDBClient

# Configure test logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> UniversalAIBrainConfig:
    """Create a test configuration for AI Brain."""
    return UniversalAIBrainConfig(
        mongodb_uri="mongodb://localhost:27017",
        database_name="ai_brain_test",
        enable_safety_systems=True,
        cognitive_systems_config={
            "emotional_intelligence": {"sensitivity": 0.8},
            "goal_hierarchy": {"max_goals": 5},
            "confidence_tracking": {"min_confidence": 0.7}
        }
    )


@pytest.fixture
async def mock_mongodb_client() -> AsyncGenerator[MongoDBClient, None]:
    """Create a mock MongoDB client for testing."""
    client = MagicMock(spec=MongoDBClient)
    
    # Mock async methods
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.store_memory = AsyncMock(return_value="test_memory_id")
    client.search_memories = AsyncMock(return_value=[])
    client.get_user_profile = AsyncMock(return_value=None)
    client.update_user_profile = AsyncMock()
    client.store_conversation = AsyncMock(return_value="test_conversation_id")
    client.get_conversation_history = AsyncMock(return_value=[])
    
    # Mock properties
    client.is_connected = True
    client.database_name = "ai_brain_test"
    
    yield client


@pytest.fixture
async def ai_brain(test_config: UniversalAIBrainConfig) -> AsyncGenerator[UniversalAIBrain, None]:
    """Create an AI Brain instance for testing."""
    brain = UniversalAIBrain(test_config)
    
    # Mock the MongoDB client to avoid actual database connections in tests
    brain.mongodb_client = MagicMock(spec=MongoDBClient)
    brain.mongodb_client.initialize = AsyncMock()
    brain.mongodb_client.close = AsyncMock()
    brain.mongodb_client.is_connected = True
    
    await brain.initialize()
    yield brain
    await brain.shutdown()


@pytest.fixture
def sample_cognitive_input() -> CognitiveInputData:
    """Create sample cognitive input data for testing."""
    return CognitiveInputData(
        text="I'm feeling excited about this new AI project!",
        input_type="user_message",
        context=CognitiveContext(
            user_id="test_user",
            session_id="test_session"
        ),
        requested_systems=["emotional_intelligence", "goal_hierarchy"],
        processing_priority=7
    )


@pytest.fixture
def sample_emotional_input() -> CognitiveInputData:
    """Create sample emotional input for testing."""
    return CognitiveInputData(
        text="I'm feeling overwhelmed and anxious about my workload",
        input_type="emotional_analysis",
        context=CognitiveContext(
            user_id="test_user",
            session_id="emotional_test"
        ),
        requested_systems=["emotional_intelligence", "empathy_response"],
        processing_priority=8
    )


@pytest.fixture
def sample_goal_input() -> CognitiveInputData:
    """Create sample goal input for testing."""
    return CognitiveInputData(
        text="I want to learn Python programming to advance my career in data science",
        input_type="goal_analysis",
        context=CognitiveContext(
            user_id="test_user",
            session_id="goal_test"
        ),
        requested_systems=["goal_hierarchy", "temporal_planning"],
        processing_priority=7
    )


@pytest.fixture
def mock_safety_system():
    """Create a mock safety system for testing."""
    safety_system = MagicMock()
    
    # Mock async methods
    safety_system.initialize = AsyncMock()
    safety_system.comprehensive_safety_check = AsyncMock(return_value={
        "overall_safe": True,
        "processing_time_ms": 50.0,
        "safety_guardrails": {
            "is_safe": True,
            "violations": []
        },
        "hallucination_detection": {
            "has_hallucinations": False,
            "detections": []
        }
    })
    safety_system.shutdown = AsyncMock()
    
    return safety_system


@pytest.fixture
def mock_cognitive_response():
    """Create a mock cognitive response for testing."""
    from ai_brain_python.core.models import CognitiveResponse, EmotionalState, GoalHierarchy
    
    return CognitiveResponse(
        confidence=0.85,
        processing_time_ms=150.0,
        cognitive_results={
            "emotional_intelligence": {
                "primary_emotion": "excitement",
                "emotion_intensity": 0.8,
                "confidence": 0.9
            },
            "goal_hierarchy": {
                "primary_goal": "Learn AI programming",
                "goal_priority": 8,
                "confidence": 0.8
            }
        },
        emotional_state=EmotionalState(
            primary_emotion="excitement",
            emotion_intensity=0.8,
            emotional_valence="positive",
            emotion_explanation="User shows enthusiasm for AI learning",
            empathy_response="That's wonderful! Your excitement will help you learn faster."
        ),
        goal_hierarchy=GoalHierarchy(
            primary_goal="Learn AI programming",
            goal_priority=8,
            goal_category="professional_development",
            sub_goals=["Learn Python", "Study machine learning", "Build projects"],
            goal_dependencies=["Basic programming knowledge"],
            estimated_timeline="6-12 months"
        )
    )


# Test data fixtures
@pytest.fixture
def test_texts():
    """Provide various test texts for different scenarios."""
    return {
        "positive_emotional": "I'm thrilled about my new job opportunity!",
        "negative_emotional": "I'm feeling stressed and overwhelmed with work",
        "neutral": "The weather is nice today",
        "goal_oriented": "I want to become a data scientist within 2 years",
        "complex": "I'm excited about learning AI but worried about the complexity",
        "pii_content": "My email is john.doe@example.com and phone is 555-123-4567",
        "potentially_harmful": "This content might be flagged by safety systems",
        "long_text": "This is a very long text " * 100,
        "empty": "",
        "special_chars": "Text with émojis 🚀 and spëcial characters!",
    }


@pytest.fixture
def framework_availability():
    """Mock framework availability for testing."""
    return {
        "crewai": True,
        "pydantic_ai": True,
        "agno": True,
        "langchain": True,
        "langgraph": True
    }


# Async test helpers
@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 30.0  # 30 seconds


# Performance test fixtures
@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "max_response_time_ms": 5000,
        "max_memory_usage_mb": 512,
        "concurrent_requests": 10,
        "test_duration_seconds": 60
    }


# Error simulation fixtures
@pytest.fixture
def mock_database_error():
    """Mock database error for testing error handling."""
    from pymongo.errors import ConnectionFailure
    return ConnectionFailure("Mock database connection error")


@pytest.fixture
def mock_cognitive_system_error():
    """Mock cognitive system error for testing."""
    return Exception("Mock cognitive system processing error")


# Integration test fixtures
@pytest.fixture
async def integration_test_brain(test_config: UniversalAIBrainConfig) -> AsyncGenerator[UniversalAIBrain, None]:
    """Create an AI Brain for integration tests (may use real MongoDB if available)."""
    brain = UniversalAIBrain(test_config)
    
    try:
        await brain.initialize()
        yield brain
    except Exception as e:
        # If initialization fails (e.g., no MongoDB), create a mock
        logger.warning(f"Integration test brain initialization failed: {e}")
        brain.mongodb_client = MagicMock(spec=MongoDBClient)
        brain.mongodb_client.initialize = AsyncMock()
        brain.mongodb_client.close = AsyncMock()
        brain.mongodb_client.is_connected = True
        await brain.initialize()
        yield brain
    finally:
        await brain.shutdown()


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_mongodb: mark test as requiring MongoDB"
    )
    config.addinivalue_line(
        "markers", "requires_frameworks: mark test as requiring AI frameworks"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add markers based on test names
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        if "mongodb" in item.name:
            item.add_marker(pytest.mark.requires_mongodb)
        if any(framework in item.name for framework in ["crewai", "pydantic", "agno", "langchain", "langgraph"]):
            item.add_marker(pytest.mark.requires_frameworks)
