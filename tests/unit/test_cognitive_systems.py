"""
Unit tests for cognitive systems.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ai_brain_python.cognitive_systems.emotional_intelligence import EmotionalIntelligenceEngine
from ai_brain_python.cognitive_systems.goal_hierarchy import GoalHierarchyManager
from ai_brain_python.cognitive_systems.confidence_tracking import ConfidenceTrackingEngine
from ai_brain_python.cognitive_systems.attention_management import AttentionManagementSystem
from ai_brain_python.core.models import CognitiveInputData, CognitiveContext


class TestEmotionalIntelligenceEngine:
    """Test cases for Emotional Intelligence Engine."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test emotional intelligence engine initialization."""
        engine = EmotionalIntelligenceEngine(test_config.cognitive_systems_config.get("emotional_intelligence", {}))
        
        # Mock MongoDB client
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        
        await engine.initialize()
        
        assert engine.initialized is True
        assert engine.config is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_positive_emotion_detection(self, test_config):
        """Test detection of positive emotions."""
        engine = EmotionalIntelligenceEngine(test_config.cognitive_systems_config.get("emotional_intelligence", {}))
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        await engine.initialize()
        
        input_data = CognitiveInputData(
            text="I'm so excited about this amazing opportunity!",
            input_type="emotional_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await engine.process_input(input_data)
        
        assert result["emotional_state"]["primary_emotion"] in ["excitement", "joy", "happiness"]
        assert result["emotional_state"]["emotional_valence"] == "positive"
        assert result["emotional_state"]["emotion_intensity"] > 0.5
        assert result["confidence"] > 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_emotion_detection(self, test_config):
        """Test detection of negative emotions."""
        engine = EmotionalIntelligenceEngine(test_config.cognitive_systems_config.get("emotional_intelligence", {}))
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        await engine.initialize()
        
        input_data = CognitiveInputData(
            text="I'm feeling really stressed and overwhelmed with everything",
            input_type="emotional_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await engine.process_input(input_data)
        
        assert result["emotional_state"]["primary_emotion"] in ["stress", "anxiety", "overwhelm"]
        assert result["emotional_state"]["emotional_valence"] == "negative"
        assert result["emotional_state"]["emotion_intensity"] > 0.5
        assert result["confidence"] > 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_neutral_emotion_detection(self, test_config):
        """Test detection of neutral emotions."""
        engine = EmotionalIntelligenceEngine(test_config.cognitive_systems_config.get("emotional_intelligence", {}))
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        await engine.initialize()
        
        input_data = CognitiveInputData(
            text="The weather is nice today. It's partly cloudy.",
            input_type="emotional_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await engine.process_input(input_data)
        
        assert result["emotional_state"]["emotional_valence"] == "neutral"
        assert result["emotional_state"]["emotion_intensity"] <= 0.5
        assert result["confidence"] > 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empathy_response_generation(self, test_config):
        """Test empathy response generation."""
        engine = EmotionalIntelligenceEngine(test_config.cognitive_systems_config.get("emotional_intelligence", {}))
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        await engine.initialize()
        
        input_data = CognitiveInputData(
            text="I'm feeling sad because my project was cancelled",
            input_type="emotional_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await engine.process_input(input_data)
        
        assert "empathy_response" in result["emotional_state"]
        assert len(result["emotional_state"]["empathy_response"]) > 0
        assert isinstance(result["emotional_state"]["empathy_response"], str)


class TestGoalHierarchyManager:
    """Test cases for Goal Hierarchy Manager."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test goal hierarchy manager initialization."""
        manager = GoalHierarchyManager(test_config.cognitive_systems_config.get("goal_hierarchy", {}))
        
        # Mock MongoDB client
        manager.mongodb_client = MagicMock()
        manager.mongodb_client.initialize = AsyncMock()
        
        await manager.initialize()
        
        assert manager.initialized is True
        assert manager.config is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_goal_extraction(self, test_config):
        """Test goal extraction from text."""
        manager = GoalHierarchyManager(test_config.cognitive_systems_config.get("goal_hierarchy", {}))
        manager.mongodb_client = MagicMock()
        manager.mongodb_client.initialize = AsyncMock()
        await manager.initialize()
        
        input_data = CognitiveInputData(
            text="I want to learn Python programming to advance my career in data science",
            input_type="goal_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await manager.process_input(input_data)
        
        assert "extracted_goals" in result
        assert "goal_hierarchy" in result
        assert result["goal_hierarchy"]["primary_goal"] is not None
        assert result["goal_hierarchy"]["goal_priority"] >= 1
        assert result["goal_hierarchy"]["goal_priority"] <= 10
        assert isinstance(result["goal_hierarchy"]["sub_goals"], list)
        assert result["confidence"] > 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_goal_prioritization(self, test_config):
        """Test goal prioritization."""
        manager = GoalHierarchyManager(test_config.cognitive_systems_config.get("goal_hierarchy", {}))
        manager.mongodb_client = MagicMock()
        manager.mongodb_client.initialize = AsyncMock()
        await manager.initialize()
        
        input_data = CognitiveInputData(
            text="I need to finish my urgent project, learn new skills, and plan my vacation",
            input_type="goal_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await manager.process_input(input_data)
        
        assert len(result["extracted_goals"]) > 1
        assert result["goal_hierarchy"]["goal_priority"] >= 1
        # Urgent project should likely have higher priority
        assert "urgent" in result["goal_hierarchy"]["primary_goal"].lower() or result["goal_hierarchy"]["goal_priority"] >= 7
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_goals_detected(self, test_config):
        """Test handling when no goals are detected."""
        manager = GoalHierarchyManager(test_config.cognitive_systems_config.get("goal_hierarchy", {}))
        manager.mongodb_client = MagicMock()
        manager.mongodb_client.initialize = AsyncMock()
        await manager.initialize()
        
        input_data = CognitiveInputData(
            text="The weather is nice today. I like coffee.",
            input_type="goal_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await manager.process_input(input_data)
        
        assert result["goal_hierarchy"]["primary_goal"] in [None, "No specific goal detected", ""]
        assert len(result["extracted_goals"]) == 0 or result["extracted_goals"] == []


class TestConfidenceTrackingEngine:
    """Test cases for Confidence Tracking Engine."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test confidence tracking engine initialization."""
        engine = ConfidenceTrackingEngine(test_config.cognitive_systems_config.get("confidence_tracking", {}))
        
        # Mock MongoDB client
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        
        await engine.initialize()
        
        assert engine.initialized is True
        assert engine.config is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, test_config):
        """Test confidence calculation."""
        engine = ConfidenceTrackingEngine(test_config.cognitive_systems_config.get("confidence_tracking", {}))
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        await engine.initialize()
        
        input_data = CognitiveInputData(
            text="I'm confident about my programming skills",
            input_type="confidence_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await engine.process_input(input_data)
        
        assert "confidence_metrics" in result
        assert "overall_confidence" in result["confidence_metrics"]
        assert result["confidence_metrics"]["overall_confidence"] >= 0.0
        assert result["confidence_metrics"]["overall_confidence"] <= 1.0
        assert result["confidence"] > 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_uncertainty_detection(self, test_config):
        """Test uncertainty detection."""
        engine = ConfidenceTrackingEngine(test_config.cognitive_systems_config.get("confidence_tracking", {}))
        engine.mongodb_client = MagicMock()
        engine.mongodb_client.initialize = AsyncMock()
        await engine.initialize()
        
        input_data = CognitiveInputData(
            text="I'm not sure if I should take this job offer. Maybe it's good, but I don't know.",
            input_type="confidence_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await engine.process_input(input_data)
        
        assert "uncertainty_indicators" in result["confidence_metrics"]
        assert len(result["confidence_metrics"]["uncertainty_indicators"]) > 0
        # Confidence should be lower for uncertain text
        assert result["confidence_metrics"]["overall_confidence"] < 0.7


class TestAttentionManagementSystem:
    """Test cases for Attention Management System."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test attention management system initialization."""
        system = AttentionManagementSystem(test_config.cognitive_systems_config.get("attention_management", {}))
        
        # Mock MongoDB client
        system.mongodb_client = MagicMock()
        system.mongodb_client.initialize = AsyncMock()
        
        await system.initialize()
        
        assert system.initialized is True
        assert system.config is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_attention_focus_detection(self, test_config):
        """Test attention focus detection."""
        system = AttentionManagementSystem(test_config.cognitive_systems_config.get("attention_management", {}))
        system.mongodb_client = MagicMock()
        system.mongodb_client.initialize = AsyncMock()
        await system.initialize()
        
        input_data = CognitiveInputData(
            text="I need to focus on completing my important project deadline tomorrow",
            input_type="attention_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await system.process_input(input_data)
        
        assert "attention_analysis" in result
        assert "focus_areas" in result["attention_analysis"]
        assert "attention_priority" in result["attention_analysis"]
        assert len(result["attention_analysis"]["focus_areas"]) > 0
        assert result["confidence"] > 0.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_distraction_detection(self, test_config):
        """Test distraction detection."""
        system = AttentionManagementSystem(test_config.cognitive_systems_config.get("attention_management", {}))
        system.mongodb_client = MagicMock()
        system.mongodb_client.initialize = AsyncMock()
        await system.initialize()
        
        input_data = CognitiveInputData(
            text="I should work on my project but I keep getting distracted by social media and emails",
            input_type="attention_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await system.process_input(input_data)
        
        assert "distraction_factors" in result["attention_analysis"]
        assert len(result["attention_analysis"]["distraction_factors"]) > 0
        # Should detect social media and emails as distractions
        distractions = " ".join(result["attention_analysis"]["distraction_factors"]).lower()
        assert "social media" in distractions or "email" in distractions
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_attention_recommendations(self, test_config):
        """Test attention management recommendations."""
        system = AttentionManagementSystem(test_config.cognitive_systems_config.get("attention_management", {}))
        system.mongodb_client = MagicMock()
        system.mongodb_client.initialize = AsyncMock()
        await system.initialize()
        
        input_data = CognitiveInputData(
            text="I have trouble concentrating on my work",
            input_type="attention_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        result = await system.process_input(input_data)
        
        assert "attention_recommendations" in result["attention_analysis"]
        assert len(result["attention_analysis"]["attention_recommendations"]) > 0
        assert isinstance(result["attention_analysis"]["attention_recommendations"], list)


class TestCognitiveSystemIntegration:
    """Test integration between cognitive systems."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_emotional_goal_integration(self, test_config):
        """Test integration between emotional intelligence and goal hierarchy."""
        # Initialize both systems
        emotional_engine = EmotionalIntelligenceEngine(test_config.cognitive_systems_config.get("emotional_intelligence", {}))
        goal_manager = GoalHierarchyManager(test_config.cognitive_systems_config.get("goal_hierarchy", {}))
        
        # Mock MongoDB clients
        for system in [emotional_engine, goal_manager]:
            system.mongodb_client = MagicMock()
            system.mongodb_client.initialize = AsyncMock()
            await system.initialize()
        
        input_data = CognitiveInputData(
            text="I'm excited about learning AI to advance my career",
            input_type="integration_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        # Process with both systems
        emotional_result = await emotional_engine.process_input(input_data)
        goal_result = await goal_manager.process_input(input_data)
        
        # Both should detect relevant information
        assert emotional_result["emotional_state"]["primary_emotion"] in ["excitement", "enthusiasm"]
        assert "learn" in goal_result["goal_hierarchy"]["primary_goal"].lower() or "career" in goal_result["goal_hierarchy"]["primary_goal"].lower()
        
        # Both should have reasonable confidence
        assert emotional_result["confidence"] > 0.5
        assert goal_result["confidence"] > 0.5
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_confidence_attention_integration(self, test_config):
        """Test integration between confidence tracking and attention management."""
        # Initialize both systems
        confidence_engine = ConfidenceTrackingEngine(test_config.cognitive_systems_config.get("confidence_tracking", {}))
        attention_system = AttentionManagementSystem(test_config.cognitive_systems_config.get("attention_management", {}))
        
        # Mock MongoDB clients
        for system in [confidence_engine, attention_system]:
            system.mongodb_client = MagicMock()
            system.mongodb_client.initialize = AsyncMock()
            await system.initialize()
        
        input_data = CognitiveInputData(
            text="I'm not sure if I can focus on this complex task",
            input_type="integration_test",
            context=CognitiveContext(user_id="test", session_id="test")
        )
        
        # Process with both systems
        confidence_result = await confidence_engine.process_input(input_data)
        attention_result = await attention_system.process_input(input_data)
        
        # Confidence should be low due to uncertainty
        assert confidence_result["confidence_metrics"]["overall_confidence"] < 0.7
        
        # Attention system should detect focus challenges
        assert "focus" in " ".join(attention_result["attention_analysis"]["focus_areas"]).lower()
        
        # Both should provide relevant insights
        assert len(confidence_result["confidence_metrics"]["uncertainty_indicators"]) > 0
        assert len(attention_result["attention_analysis"]["attention_recommendations"]) > 0
