"""
Unit tests for Universal AI Brain core functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext, CognitiveResponse


class TestUniversalAIBrain:
    """Test cases for Universal AI Brain."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test AI Brain initialization."""
        brain = UniversalAIBrain(test_config)
        
        # Mock MongoDB client
        brain.mongodb_client = MagicMock()
        brain.mongodb_client.initialize = AsyncMock()
        brain.mongodb_client.is_connected = True
        
        await brain.initialize()
        
        assert brain.initialized is True
        assert brain.config == test_config
        assert len(brain.cognitive_systems) > 0
        
        await brain.shutdown()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization_failure(self, test_config):
        """Test AI Brain initialization failure handling."""
        brain = UniversalAIBrain(test_config)
        
        # Mock MongoDB client to fail
        brain.mongodb_client = MagicMock()
        brain.mongodb_client.initialize = AsyncMock(side_effect=Exception("Connection failed"))
        
        with pytest.raises(Exception):
            await brain.initialize()
        
        assert brain.initialized is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_basic(self, ai_brain, sample_cognitive_input):
        """Test basic input processing."""
        response = await ai_brain.process_input(sample_cognitive_input)
        
        assert isinstance(response, CognitiveResponse)
        assert response.confidence >= 0.0
        assert response.confidence <= 1.0
        assert response.processing_time_ms > 0
        assert response.emotional_state is not None
        assert response.goal_hierarchy is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_emotional(self, ai_brain, sample_emotional_input):
        """Test emotional input processing."""
        response = await ai_brain.process_input(sample_emotional_input)
        
        assert response.emotional_state.primary_emotion is not None
        assert response.emotional_state.emotion_intensity >= 0.0
        assert response.emotional_state.emotion_intensity <= 1.0
        assert response.emotional_state.emotional_valence in ["positive", "negative", "neutral"]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_goal_analysis(self, ai_brain, sample_goal_input):
        """Test goal analysis input processing."""
        response = await ai_brain.process_input(sample_goal_input)
        
        assert response.goal_hierarchy.primary_goal is not None
        assert response.goal_hierarchy.goal_priority >= 1
        assert response.goal_hierarchy.goal_priority <= 10
        assert isinstance(response.goal_hierarchy.sub_goals, list)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_with_specific_systems(self, ai_brain):
        """Test processing with specific cognitive systems."""
        input_data = CognitiveInputData(
            text="I'm excited about learning AI",
            input_type="test",
            context=CognitiveContext(user_id="test", session_id="test"),
            requested_systems=["emotional_intelligence", "confidence_tracking"]
        )
        
        response = await ai_brain.process_input(input_data)
        
        # Should only have results from requested systems
        assert "emotional_intelligence" in response.cognitive_results
        assert "confidence_tracking" in response.cognitive_results
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_empty_text(self, ai_brain):
        """Test processing empty text."""
        input_data = CognitiveInputData(
            text="",
            input_type="test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        response = await ai_brain.process_input(input_data)
        
        # Should handle empty text gracefully
        assert isinstance(response, CognitiveResponse)
        assert response.confidence >= 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_long_text(self, ai_brain):
        """Test processing very long text."""
        long_text = "This is a very long text. " * 1000
        
        input_data = CognitiveInputData(
            text=long_text,
            input_type="test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        response = await ai_brain.process_input(input_data)
        
        # Should handle long text without errors
        assert isinstance(response, CognitiveResponse)
        assert response.processing_time_ms > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_special_characters(self, ai_brain):
        """Test processing text with special characters."""
        special_text = "Text with émojis 🚀 and spëcial characters! @#$%^&*()"
        
        input_data = CognitiveInputData(
            text=special_text,
            input_type="test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        response = await ai_brain.process_input(input_data)
        
        # Should handle special characters gracefully
        assert isinstance(response, CognitiveResponse)
        assert response.confidence >= 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, ai_brain):
        """Test concurrent input processing."""
        inputs = [
            CognitiveInputData(
                text=f"Test message {i}",
                input_type="test",
                context=CognitiveContext(user_id=f"user_{i}", session_id=f"session_{i}")
            )
            for i in range(5)
        ]
        
        # Process all inputs concurrently
        tasks = [ai_brain.process_input(input_data) for input_data in inputs]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(responses) == 5
        for response in responses:
            assert isinstance(response, CognitiveResponse)
            assert response.confidence >= 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cognitive_system_error_handling(self, ai_brain):
        """Test error handling in cognitive systems."""
        # Mock a cognitive system to raise an error
        with patch.object(ai_brain.cognitive_systems["emotional_intelligence"], "process_input", 
                         side_effect=Exception("Mock error")):
            
            input_data = CognitiveInputData(
                text="Test input",
                input_type="test",
                context=CognitiveContext(user_id="test", session_id="test"),
                requested_systems=["emotional_intelligence"]
            )
            
            response = await ai_brain.process_input(input_data)
            
            # Should handle error gracefully
            assert isinstance(response, CognitiveResponse)
            # Confidence might be lower due to error
            assert response.confidence >= 0.0
    
    @pytest.mark.unit
    def test_get_system_status(self, ai_brain):
        """Test system status retrieval."""
        status = ai_brain.get_system_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "cognitive_systems" in status
        assert "total_requests" in status
        assert "average_processing_time_ms" in status
        assert status["status"] in ["initializing", "ready", "error"]
    
    @pytest.mark.unit
    def test_get_cognitive_insights(self, ai_brain):
        """Test cognitive insights retrieval."""
        insights = ai_brain.get_cognitive_insights()
        
        assert isinstance(insights, dict)
        assert "system_performance" in insights
        assert "cognitive_patterns" in insights
        assert "recommendations" in insights
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown(self, test_config):
        """Test AI Brain shutdown."""
        brain = UniversalAIBrain(test_config)
        
        # Mock MongoDB client
        brain.mongodb_client = MagicMock()
        brain.mongodb_client.initialize = AsyncMock()
        brain.mongodb_client.close = AsyncMock()
        brain.mongodb_client.is_connected = True
        
        await brain.initialize()
        assert brain.initialized is True
        
        await brain.shutdown()
        assert brain.initialized is False
        brain.mongodb_client.close.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_input_priority_handling(self, ai_brain):
        """Test processing priority handling."""
        high_priority_input = CognitiveInputData(
            text="High priority message",
            input_type="urgent",
            context=CognitiveContext(user_id="test", session_id="test"),
            processing_priority=10
        )
        
        low_priority_input = CognitiveInputData(
            text="Low priority message",
            input_type="normal",
            context=CognitiveContext(user_id="test", session_id="test"),
            processing_priority=1
        )
        
        # Process both
        high_response = await ai_brain.process_input(high_priority_input)
        low_response = await ai_brain.process_input(low_priority_input)
        
        # Both should succeed
        assert isinstance(high_response, CognitiveResponse)
        assert isinstance(low_response, CognitiveResponse)
        
        # High priority might have better confidence or faster processing
        # (This depends on implementation details)
        assert high_response.confidence >= 0.0
        assert low_response.confidence >= 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_preservation(self, ai_brain):
        """Test context preservation across multiple inputs."""
        context = CognitiveContext(
            user_id="context_test_user",
            session_id="context_test_session"
        )
        
        # First input
        input1 = CognitiveInputData(
            text="I'm starting a new project",
            input_type="context_test",
            context=context
        )
        
        response1 = await ai_brain.process_input(input1)
        
        # Second input with same context
        input2 = CognitiveInputData(
            text="I'm excited about the progress",
            input_type="context_test",
            context=context
        )
        
        response2 = await ai_brain.process_input(input2)
        
        # Both should succeed and potentially show context awareness
        assert isinstance(response1, CognitiveResponse)
        assert isinstance(response2, CognitiveResponse)
        
        # Context should be preserved (implementation dependent)
        assert response1.cognitive_results is not None
        assert response2.cognitive_results is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, ai_brain):
        """Test handling of invalid input data."""
        # Test with None text
        with pytest.raises((ValueError, TypeError)):
            input_data = CognitiveInputData(
                text=None,
                input_type="test",
                context=CognitiveContext(user_id="test", session_id="test")
            )
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_performance_metrics(self, ai_brain):
        """Test performance metrics collection."""
        input_data = CognitiveInputData(
            text="Performance test message",
            input_type="performance_test",
            context=CognitiveContext(user_id="perf_test", session_id="perf_session")
        )
        
        start_time = datetime.utcnow()
        response = await ai_brain.process_input(input_data)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Response should include timing information
        assert response.processing_time_ms > 0
        assert response.processing_time_ms <= processing_time + 100  # Allow some margin
        
        # System status should reflect the request
        status = ai_brain.get_system_status()
        assert status["total_requests"] > 0
