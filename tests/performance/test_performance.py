"""
Performance tests for AI Brain Python.
"""

import pytest
import asyncio
import time
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

from ai_brain_python import UniversalAIBrain, UniversalAIBrainConfig
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext


class TestPerformance:
    """Performance tests for AI Brain core functionality."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_request_performance(self, ai_brain, sample_cognitive_input):
        """Test performance of single request processing."""
        
        start_time = time.time()
        response = await ai_brain.process_input(sample_cognitive_input)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Performance assertions
        assert processing_time < 5000  # Should complete within 5 seconds
        assert response.processing_time_ms < 5000
        assert response.confidence > 0.0
        
        print(f"Single request processing time: {processing_time:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, ai_brain):
        """Test performance with concurrent requests."""
        
        # Create multiple test inputs
        inputs = [
            CognitiveInputData(
                text=f"Test message {i} for concurrent processing",
                input_type="performance_test",
                context=CognitiveContext(
                    user_id=f"perf_user_{i}",
                    session_id=f"perf_session_{i}"
                )
            )
            for i in range(10)
        ]
        
        start_time = time.time()
        
        # Process all inputs concurrently
        tasks = [ai_brain.process_input(input_data) for input_data in inputs]
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Performance assertions
        assert len(responses) == 10
        assert total_time < 15000  # Should complete within 15 seconds
        
        # All responses should be valid
        for response in responses:
            assert response.confidence > 0.0
            assert response.processing_time_ms > 0
        
        avg_response_time = statistics.mean([r.processing_time_ms for r in responses])
        
        print(f"Concurrent requests total time: {total_time:.2f}ms")
        print(f"Average response time: {avg_response_time:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage(self, ai_brain):
        """Test memory usage during processing."""
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple requests
        for i in range(20):
            input_data = CognitiveInputData(
                text=f"Memory test message {i}",
                input_type="memory_test",
                context=CognitiveContext(
                    user_id=f"memory_user_{i}",
                    session_id=f"memory_session_{i}"
                )
            )
            
            response = await ai_brain.process_input(input_data)
            assert response.confidence > 0.0
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not increase excessively
        assert memory_increase < 100  # Should not increase by more than 100MB
        
        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Final memory: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput(self, ai_brain):
        """Test system throughput (requests per second)."""
        
        request_count = 50
        start_time = time.time()
        
        # Create and process requests
        tasks = []
        for i in range(request_count):
            input_data = CognitiveInputData(
                text=f"Throughput test {i}",
                input_type="throughput_test",
                context=CognitiveContext(
                    user_id=f"throughput_user_{i}",
                    session_id=f"throughput_session_{i}"
                )
            )
            tasks.append(ai_brain.process_input(input_data))
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = request_count / total_time
        
        # Performance assertions
        assert len(responses) == request_count
        assert throughput > 1.0  # Should handle at least 1 request per second
        
        # All responses should be valid
        for response in responses:
            assert response.confidence > 0.0
        
        print(f"Processed {request_count} requests in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load(self, ai_brain):
        """Test performance under sustained load."""
        
        duration_seconds = 30
        request_interval = 0.1  # 10 requests per second
        
        start_time = time.time()
        responses = []
        
        while (time.time() - start_time) < duration_seconds:
            input_data = CognitiveInputData(
                text=f"Sustained load test at {time.time()}",
                input_type="sustained_load_test",
                context=CognitiveContext(
                    user_id="sustained_user",
                    session_id="sustained_session"
                )
            )
            
            response = await ai_brain.process_input(input_data)
            responses.append(response)
            
            await asyncio.sleep(request_interval)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Performance assertions
        assert len(responses) > 0
        assert actual_duration >= duration_seconds * 0.9  # Allow 10% variance
        
        # Calculate performance metrics
        response_times = [r.processing_time_ms for r in responses]
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # Performance should remain stable
        assert avg_response_time < 5000  # Average should be under 5 seconds
        assert max_response_time < 10000  # Max should be under 10 seconds
        
        print(f"Sustained load test duration: {actual_duration:.2f}s")
        print(f"Total requests: {len(responses)}")
        print(f"Average response time: {avg_response_time:.2f}ms")
        print(f"Min response time: {min_response_time:.2f}ms")
        print(f"Max response time: {max_response_time:.2f}ms")


class TestCognitiveSystemPerformance:
    """Performance tests for individual cognitive systems."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_emotional_intelligence_performance(self, ai_brain):
        """Test emotional intelligence system performance."""
        
        emotional_texts = [
            "I'm feeling extremely happy about this wonderful news!",
            "I'm devastated and heartbroken by this terrible situation.",
            "I'm anxious and worried about the upcoming presentation.",
            "I'm excited and enthusiastic about the new project.",
            "I'm frustrated and annoyed by these constant interruptions."
        ]
        
        start_time = time.time()
        
        for text in emotional_texts:
            input_data = CognitiveInputData(
                text=text,
                input_type="emotional_performance_test",
                context=CognitiveContext(user_id="emotional_perf", session_id="emotional_session"),
                requested_systems=["emotional_intelligence"]
            )
            
            response = await ai_brain.process_input(input_data)
            assert response.emotional_state.primary_emotion is not None
            assert response.processing_time_ms < 3000  # Should be fast for single system
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        avg_time_per_request = total_time / len(emotional_texts)
        
        print(f"Emotional intelligence average time: {avg_time_per_request:.2f}ms")
        assert avg_time_per_request < 2000  # Should average under 2 seconds
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_goal_hierarchy_performance(self, ai_brain):
        """Test goal hierarchy system performance."""
        
        goal_texts = [
            "I want to learn Python programming to advance my career",
            "I need to finish my project by the deadline next week",
            "I plan to start exercising regularly to improve my health",
            "I want to save money for a vacation next summer",
            "I need to improve my communication skills for better relationships"
        ]
        
        start_time = time.time()
        
        for text in goal_texts:
            input_data = CognitiveInputData(
                text=text,
                input_type="goal_performance_test",
                context=CognitiveContext(user_id="goal_perf", session_id="goal_session"),
                requested_systems=["goal_hierarchy"]
            )
            
            response = await ai_brain.process_input(input_data)
            assert response.goal_hierarchy.primary_goal is not None
            assert response.processing_time_ms < 3000
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        avg_time_per_request = total_time / len(goal_texts)
        
        print(f"Goal hierarchy average time: {avg_time_per_request:.2f}ms")
        assert avg_time_per_request < 2000
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multiple_systems_performance(self, ai_brain):
        """Test performance when using multiple cognitive systems."""
        
        test_text = "I'm excited about learning AI but worried about the complexity"
        
        # Test with increasing number of systems
        system_combinations = [
            ["emotional_intelligence"],
            ["emotional_intelligence", "goal_hierarchy"],
            ["emotional_intelligence", "goal_hierarchy", "confidence_tracking"],
            ["emotional_intelligence", "goal_hierarchy", "confidence_tracking", "attention_management"]
        ]
        
        for systems in system_combinations:
            start_time = time.time()
            
            input_data = CognitiveInputData(
                text=test_text,
                input_type="multi_system_performance_test",
                context=CognitiveContext(user_id="multi_perf", session_id="multi_session"),
                requested_systems=systems
            )
            
            response = await ai_brain.process_input(input_data)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            # Performance should scale reasonably with number of systems
            expected_max_time = len(systems) * 2000  # 2 seconds per system
            assert processing_time < expected_max_time
            
            print(f"{len(systems)} systems processing time: {processing_time:.2f}ms")


class TestSafetySystemPerformance:
    """Performance tests for safety systems."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_safety_check_performance(self):
        """Test safety check performance."""
        from ai_brain_python.safety import IntegratedSafetySystem
        
        safety_system = IntegratedSafetySystem()
        
        # Mock components for performance testing
        safety_system.compliance_logger.initialize = AsyncMock()
        safety_system.real_time_monitor.start = AsyncMock()
        safety_system.compliance_logger.log_event = AsyncMock(return_value="test_event")
        safety_system.real_time_monitor.record_request = MagicMock()
        safety_system.real_time_monitor.add_custom_metric = MagicMock()
        
        await safety_system.initialize()
        
        test_texts = [
            "This is a normal safe message",
            "My email is test@example.com",
            "This content might be questionable",
            "I'm feeling great about this project",
            "Here's some technical information about AI"
        ]
        
        start_time = time.time()
        
        for text in test_texts:
            result = await safety_system.comprehensive_safety_check(
                text=text,
                user_id="perf_user",
                session_id="perf_session"
            )
            assert "overall_safe" in result
            assert result["processing_time_ms"] < 5000  # Should be fast
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        avg_time = total_time / len(test_texts)
        
        print(f"Safety check average time: {avg_time:.2f}ms")
        assert avg_time < 3000  # Should average under 3 seconds
        
        await safety_system.shutdown()


class TestDatabasePerformance:
    """Performance tests for database operations."""
    
    @pytest.mark.performance
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_mongodb_performance(self, test_config):
        """Test MongoDB performance."""
        from ai_brain_python.database.mongodb_client import MongoDBClient
        
        try:
            client = MongoDBClient(test_config.mongodb_uri)
            await client.initialize()
            
            # Test memory storage performance
            start_time = time.time()
            
            memory_ids = []
            for i in range(10):
                memory_id = await client.store_memory(
                    user_id=f"perf_user_{i}",
                    content=f"Performance test memory {i}",
                    memory_type="performance_test",
                    metadata={"test_index": i}
                )
                memory_ids.append(memory_id)
            
            end_time = time.time()
            storage_time = (end_time - start_time) * 1000
            
            # Test memory search performance
            start_time = time.time()
            
            results = await client.search_memories(
                user_id="perf_user_1",
                query="performance test",
                limit=5
            )
            
            end_time = time.time()
            search_time = (end_time - start_time) * 1000
            
            # Performance assertions
            assert len(memory_ids) == 10
            assert storage_time < 10000  # Should store 10 memories in under 10 seconds
            assert search_time < 5000  # Should search in under 5 seconds
            
            print(f"Memory storage time (10 items): {storage_time:.2f}ms")
            print(f"Memory search time: {search_time:.2f}ms")
            
            await client.close()
            
        except Exception as e:
            pytest.skip(f"MongoDB not available: {e}")


class TestScalabilityLimits:
    """Test scalability limits and edge cases."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_text_processing(self, ai_brain):
        """Test processing of very large text inputs."""
        
        # Create progressively larger texts
        text_sizes = [1000, 5000, 10000, 20000]  # Character counts
        
        for size in text_sizes:
            large_text = "This is a test sentence. " * (size // 25)  # Approximate size
            
            start_time = time.time()
            
            input_data = CognitiveInputData(
                text=large_text,
                input_type="large_text_test",
                context=CognitiveContext(user_id="large_text_user", session_id="large_text_session")
            )
            
            response = await ai_brain.process_input(input_data)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            # Should handle large texts but may take longer
            max_expected_time = size * 2  # 2ms per character (very generous)
            assert processing_time < max_expected_time
            assert response.confidence > 0.0
            
            print(f"Text size {size} chars: {processing_time:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_concurrency_limits(self, ai_brain):
        """Test system behavior under high concurrency."""
        
        # Test with high number of concurrent requests
        concurrent_requests = 50
        
        inputs = [
            CognitiveInputData(
                text=f"High concurrency test {i}",
                input_type="concurrency_test",
                context=CognitiveContext(
                    user_id=f"concurrent_user_{i}",
                    session_id=f"concurrent_session_{i}"
                )
            )
            for i in range(concurrent_requests)
        ]
        
        start_time = time.time()
        
        # Use semaphore to limit actual concurrency if needed
        semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent
        
        async def process_with_semaphore(input_data):
            async with semaphore:
                return await ai_brain.process_input(input_data)
        
        tasks = [process_with_semaphore(input_data) for input_data in inputs]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Count successful responses
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        
        # Should handle most requests successfully
        success_rate = len(successful_responses) / len(responses)
        assert success_rate > 0.8  # At least 80% success rate
        
        print(f"High concurrency test ({concurrent_requests} requests): {total_time:.2f}ms")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Successful responses: {len(successful_responses)}/{len(responses)}")
        
        # Check that successful responses are valid
        for response in successful_responses:
            assert response.confidence > 0.0
