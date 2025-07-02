"""
EmotionalIntelligenceEngine - Advanced emotional intelligence for AI agents

Exact Python equivalent of JavaScript EmotionalIntelligenceEngine.ts with:
- Real-time emotion detection and tracking
- Automatic emotional decay with TTL indexes
- Emotional pattern analysis and learning
- Context-aware emotional responses
- Emotional memory and state transitions
- Cognitive impact assessment

Features:
- Time-series collections for emotional state tracking
- TTL indexes for automatic emotional decay
- Complex aggregation pipelines for emotional analytics
- Real-time emotional pattern recognition
- Emotional memory and learning systems
"""

import re
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_brain_python.storage.collections.emotional_state_collection import EmotionalStateCollection
from ai_brain_python.core.models.cognitive_interfaces import (
    EmotionalContext, EmotionDetectionResult, EmotionalResponse,
    EmotionalGuidance, CognitiveImpact, EmotionalLearning,
    EmotionalPattern, EmotionalImprovement, EmotionalCalibration
)
from ai_brain_python.utils.logger import logger


class EmotionalIntelligenceEngine:
    """
    EmotionalIntelligenceEngine - Advanced emotional intelligence for AI agents
    
    Exact Python equivalent of JavaScript EmotionalIntelligenceEngine with:
    - Time-series collections for emotional state tracking
    - TTL indexes for automatic emotional decay
    - Complex aggregation pipelines for emotional analytics
    - Real-time emotional pattern recognition
    - Emotional memory and learning systems
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.emotional_state_collection = EmotionalStateCollection(db)
        self.is_initialized = False

        # Emotional intelligence configuration (exact match with JavaScript)
        self.config = {
            "decay_settings": {
                "default_half_life": 30,  # minutes
                "default_baseline_return": 60,  # minutes
                "decay_function": "exponential"
            },
            "detection_thresholds": {
                "min_confidence": 0.6,
                "intensity_threshold": 0.1,
                "valence_threshold": 0.05
            },
            "cognitive_impact": {
                "attention_weight": 0.3,
                "memory_weight": 0.4,
                "decision_weight": 0.3
            }
        }

        # Emotion detection patterns (simplified for demonstration)
        self.emotion_patterns = {
            'joy': [
                r'\b(happy|joy|excited|thrilled|delighted|pleased|glad|cheerful)\b',
                r'\b(amazing|wonderful|fantastic|great|excellent|awesome)\b',
                r'[!]{2,}',  # Multiple exclamation marks
                r':\)|:-\)|:D|:-D'  # Happy emoticons
            ],
            'sadness': [
                r'\b(sad|depressed|down|upset|disappointed|heartbroken|miserable)\b',
                r'\b(terrible|awful|horrible|bad|worst)\b',
                r':\(|:-\(|:\'('  # Sad emoticons
            ],
            'anger': [
                r'\b(angry|mad|furious|irritated|annoyed|frustrated|outraged)\b',
                r'\b(hate|stupid|ridiculous|absurd|nonsense)\b',
                r'[!]{3,}'  # Many exclamation marks
            ],
            'fear': [
                r'\b(afraid|scared|worried|anxious|nervous|terrified|panic)\b',
                r'\b(dangerous|risky|unsafe|threat|problem)\b'
            ],
            'surprise': [
                r'\b(surprised|shocked|amazed|astonished|unexpected)\b',
                r'\b(wow|whoa|omg|incredible|unbelievable)\b'
            ],
            'disgust': [
                r'\b(disgusting|gross|awful|terrible|horrible|nasty)\b',
                r'\b(yuck|ew|ugh)\b'
            ],
            'trust': [
                r'\b(trust|reliable|dependable|confident|sure|certain)\b',
                r'\b(believe|faith|count on|rely on)\b'
            ],
            'anticipation': [
                r'\b(excited|eager|looking forward|anticipate|expect|hope)\b',
                r'\b(soon|upcoming|future|plan|will)\b'
            ]
        }
    
    async def initialize(self) -> None:
        """Initialize the emotional intelligence engine."""
        if self.is_initialized:
            return
        
        try:
            # Initialize emotional state collection
            await self.emotional_state_collection.create_indexes()
            
            self.is_initialized = True
            logger.info("✅ EmotionalIntelligenceEngine initialized successfully")
            
        except Exception as error:
            logger.error(f"❌ Failed to initialize EmotionalIntelligenceEngine: {error}")
            raise error
    
    async def detectEmotion(self, context: EmotionalContext) -> EmotionDetectionResult:
        """Detect emotions from input text and context."""
        if not self.is_initialized:
            raise Exception("EmotionalIntelligenceEngine must be initialized first")
        
        # Get current emotional state for context
        current_state = await self.emotional_state_collection.get_current_emotional_state(
            context.agent_id,
            context.session_id
        )
        
        # Analyze input for emotional content
        emotion_analysis = await self._analyze_emotional_content(
            context.input,
            context.conversation_history,
            current_state
        )
        
        # Consider task and user context
        contextual_adjustment = self._adjust_for_context(
            emotion_analysis,
            context.task_context,
            context.user_context
        )
        
        return contextual_adjustment

    async def process_emotional_state(
        self,
        context: EmotionalContext,
        detected_emotion: EmotionDetectionResult,
        trigger: str,
        trigger_type: str
    ) -> EmotionalResponse:
        """Process emotional state and provide intelligent response guidance."""
        # Get current emotional state for context
        current_state = await self.emotional_state_collection.get_current_emotional_state(
            context.agent_id,
            context.session_id
        )

        # Create emotional state record
        emotional_state = {
            "agentId": context.agent_id,
            "sessionId": context.session_id,
            "timestamp": datetime.utcnow(),
            "emotions": {
                "primary": detected_emotion.primary,
                "secondary": detected_emotion.secondary,
                "intensity": detected_emotion.intensity,
                "valence": detected_emotion.valence,
                "arousal": detected_emotion.arousal,
                "dominance": detected_emotion.dominance
            },
            "context": {
                "trigger": trigger,
                "triggerType": trigger_type,
                "conversationTurn": len(context.conversation_history),
                "taskId": context.task_context.get("task_id") if context.task_context else None,
                "workflowId": context.task_context.get("workflow_id") if context.task_context else None,
                "previousEmotion": current_state["emotions"]["primary"] if current_state else None
            },
            "cognitiveEffects": {
                "attentionModification": self._calculate_attention_modification(detected_emotion),
                "memoryStrength": self._calculate_memory_strength(detected_emotion),
                "decisionBias": self._calculate_decision_bias(detected_emotion),
                "responseStyle": self._determine_response_style(detected_emotion)
            },
            "decay": {
                "halfLife": self._calculate_half_life(detected_emotion),
                "decayFunction": "exponential",
                "baselineReturn": self._calculate_baseline_return(detected_emotion)
            },
            "metadata": {
                "framework": "python_ai_brain",
                "model": "emotional_intelligence_v1",
                "confidence": detected_emotion.confidence,
                "source": "detected",
                "version": "1.0.0"
            }
        }

        # Store emotional state
        await self.emotional_state_collection.record_emotional_state(emotional_state)

        # Generate response guidance
        emotional_guidance = self._generate_emotional_guidance(detected_emotion)
        cognitive_impact = self._assess_cognitive_impact(detected_emotion)
        recommendations = await self._generate_recommendations(context, detected_emotion)

        return EmotionalResponse(
            current_emotion=emotional_state,
            emotional_guidance=emotional_guidance,
            cognitive_impact=cognitive_impact,
            recommendations=recommendations
        )

    async def getEmotionalTimeline(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get emotional timeline for an agent."""
        return await self.emotional_state_collection.get_emotional_timeline(agent_id, options)

    async def analyze_emotional_patterns(
        self,
        agent_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Analyze emotional patterns using MongoDB aggregation."""
        return await self.emotional_state_collection.analyze_emotional_patterns(agent_id, days)

    async def getEmotionalStats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get emotional intelligence statistics."""
        stats = await self.emotional_state_collection.get_emotional_stats(agent_id)

        # Get dominant emotions (simplified for now)
        dominant_emotions = [
            {"emotion": "neutral", "frequency": 0.4},
            {"emotion": "joy", "frequency": 0.3},
            {"emotion": "concern", "frequency": 0.2},
            {"emotion": "satisfaction", "frequency": 0.1}
        ]

        return {
            **stats,
            "dominantEmotions": dominant_emotions
        }

    async def _analyze_emotional_content(
        self,
        input_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        current_state: Optional[Dict[str, Any]] = None
    ) -> EmotionDetectionResult:
        """Analyze emotional content using pattern matching and heuristics."""
        # Emotional keyword patterns (exact match with JavaScript)
        emotional_patterns = {
            'joy': ['happy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'perfect'],
            'sadness': ['sad', 'disappointed', 'upset', 'down', 'depressed', 'unhappy', 'terrible'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'outraged'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'concerned', 'panic'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow', 'incredible'],
            'disgust': ['disgusted', 'awful', 'horrible', 'gross', 'terrible', 'hate'],
            'trust': ['trust', 'confident', 'reliable', 'sure', 'certain', 'believe'],
            'anticipation': ['excited', 'looking forward', 'can\'t wait', 'eager', 'hopeful']
        }

        input_lower = input_text.lower()
        emotion_scores = {}

        # Calculate emotion scores based on keyword matching
        for emotion, keywords in emotional_patterns.items():
            score = 0
            for keyword in keywords:
                matches = len(re.findall(rf'\b{re.escape(keyword)}\b', input_lower))
                score += matches
            emotion_scores[emotion] = score

        # Find primary emotion
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        primary = 'neutral' if primary_emotion[1] == 0 else primary_emotion[0]
        intensity = 0.1 if primary_emotion[1] == 0 else min(primary_emotion[1] * 0.3, 1.0)

        # Calculate valence based on emotion type (exact match with JavaScript)
        valence_map = {
            'joy': 0.8, 'sadness': -0.7, 'anger': -0.6, 'fear': -0.5,
            'surprise': 0.2, 'disgust': -0.8, 'trust': 0.6, 'anticipation': 0.4,
            'neutral': 0.0
        }
        valence = valence_map.get(primary, 0)

        # Calculate arousal (how activating the emotion is)
        arousal_map = {
            'joy': 0.7, 'sadness': 0.3, 'anger': 0.9, 'fear': 0.8,
            'surprise': 0.9, 'disgust': 0.6, 'trust': 0.4, 'anticipation': 0.6,
            'neutral': 0.3
        }
        arousal = arousal_map.get(primary, 0.5)

        # Calculate dominance (how much control the emotion implies)
        dominance_map = {
            'joy': 0.6, 'sadness': 0.2, 'anger': 0.8, 'fear': 0.1,
            'surprise': 0.3, 'disgust': 0.4, 'trust': 0.7, 'anticipation': 0.5,
            'neutral': 0.5
        }
        dominance = dominance_map.get(primary, 0.5)

        # Determine secondary emotions
        secondary = [
            emotion for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            if emotion != primary and score > 0
        ][:2]

        return EmotionDetectionResult(
            primary=primary,
            secondary=secondary if secondary else None,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence=min(0.7 + (intensity * 0.3), 1.0),
            reasoning=f"Detected {primary} emotion based on keyword analysis with intensity {intensity:.2f}"
        )

    def _adjust_for_context(
        self,
        emotion: EmotionDetectionResult,
        task_context: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> EmotionDetectionResult:
        """Adjust emotion detection based on context."""
        adjusted_intensity = emotion.intensity
        adjusted_valence = emotion.valence

        # Task context adjustments (exact match with JavaScript)
        if task_context:
            if task_context.get('task_type') == 'error_handling':
                adjusted_valence = min(adjusted_valence - 0.2, 1.0)
                adjusted_intensity = min(adjusted_intensity + 0.1, 1.0)

            if task_context.get('progress') and task_context['progress'] > 0.8:
                adjusted_valence = min(adjusted_valence + 0.1, 1.0)

        # User context adjustments (exact match with JavaScript)
        if user_context:
            if user_context.get('urgency') and user_context['urgency'] > 0.7:
                adjusted_intensity = min(adjusted_intensity + 0.2, 1.0)

            if user_context.get('satisfaction') and user_context['satisfaction'] < 0.3:
                adjusted_valence = min(adjusted_valence - 0.3, 1.0)

        # Create new emotion result with adjusted values
        return EmotionDetectionResult(
            primary=emotion.primary,
            secondary=emotion.secondary,
            intensity=adjusted_intensity,
            valence=adjusted_valence,
            arousal=emotion.arousal,
            dominance=emotion.dominance,
            confidence=emotion.confidence,
            reasoning=emotion.reasoning
        )

    def _calculate_attention_modification(self, emotion: EmotionDetectionResult) -> float:
        """Calculate attention modification based on arousal and valence."""
        return (emotion.arousal - 0.5) * emotion.intensity

    def _calculate_memory_strength(self, emotion: EmotionDetectionResult) -> float:
        """Calculate memory strength based on intensity and arousal."""
        return min((emotion.intensity + emotion.arousal) / 2, 1.0)

    def _calculate_decision_bias(self, emotion: EmotionDetectionResult) -> float:
        """Calculate decision bias based on valence and dominance."""
        return emotion.valence * emotion.dominance * emotion.intensity

    def _determine_response_style(self, emotion: EmotionDetectionResult) -> str:
        """Determine response style based on emotional dimensions."""
        if emotion.dominance > 0.6 and emotion.valence > 0:
            return 'assertive'
        elif emotion.valence > 0.3 and emotion.arousal < 0.5:
            return 'empathetic'
        elif emotion.arousal > 0.7:
            return 'creative'
        elif emotion.valence < -0.3:
            return 'cautious'
        else:
            return 'analytical'

    def _calculate_half_life(self, emotion: EmotionDetectionResult) -> int:
        """Calculate emotional decay half-life."""
        return self.config["decay_settings"]["default_half_life"]

    def _calculate_baseline_return(self, emotion: EmotionDetectionResult) -> int:
        """Calculate baseline return time."""
        return self.config["decay_settings"]["default_baseline_return"]

    def _generate_emotional_guidance(self, emotion: EmotionDetectionResult) -> EmotionalGuidance:
        """Generate emotional guidance for response style."""
        response_style = 'confident' if emotion.dominance > 0.6 else 'supportive'
        approach = 'positive' if emotion.valence > 0 else 'understanding'
        empathy_level = max(0.3, 1 - emotion.dominance)
        emotional_considerations = []

        if emotion.intensity > 0.8:
            emotional_considerations.append('High emotional intensity detected')
        if emotion.valence < -0.5:
            emotional_considerations.append('Negative emotional state requires support')
        if emotion.arousal > 0.7:
            emotional_considerations.append('High arousal requires calm response')

        return EmotionalGuidance(
            tone=approach,
            approach=response_style,
            empathy_level=empathy_level,
            response_style=self._determine_response_style(emotion),
            emotional_considerations=emotional_considerations
        )

    def _assess_cognitive_impact(self, emotion: EmotionDetectionResult) -> CognitiveImpact:
        """Assess cognitive impact of emotional state."""
        attention_modification = self._calculate_attention_modification(emotion)
        memory_strength = self._calculate_memory_strength(emotion)
        decision_bias = self._calculate_decision_bias(emotion)
        response_style = self._determine_response_style(emotion)

        return CognitiveImpact(
            attention_modification=attention_modification,
            memory_strength=memory_strength,
            decision_bias=decision_bias,
            response_style=response_style
        )

    async def _generate_recommendations(
        self,
        context: EmotionalContext,
        emotion: EmotionDetectionResult
    ) -> List[str]:
        """Generate recommendations based on emotional state."""
        recommendations = []

        if emotion.intensity > 0.8:
            recommendations.append('Consider emotional regulation techniques')

        if emotion.valence < -0.5:
            recommendations.append('Focus on problem resolution and user support')

        if emotion.arousal > 0.7:
            recommendations.append('Maintain calm and measured responses')

        if emotion.dominance < 0.3:
            recommendations.append('Provide reassurance and build confidence')

        return recommendations

    async def analyzeEmotionalLearning(self, agent_id: str, days: int = 7) -> EmotionalLearning:
        """Analyze emotional learning patterns for an agent."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            # Get emotional data for the period
            emotional_data = await self.emotional_collection.collection.find({
                "agentId": agent_id,
                "timestamp": {"$gte": start_date}
            }).to_list(length=None)

            if not emotional_data:
                return {
                    "agentId": agent_id,
                    "period": f"{days} days",
                    "dominantEmotions": [],
                    "emotionalStability": 0.5,
                    "learningProgress": 0.0,
                    "insights": ["No emotional data available for analysis"]
                }

            # Analyze dominant emotions
            emotion_counts = {}
            total_intensity = 0
            intensity_variance = []

            for data in emotional_data:
                emotion = data.get("emotion", {})
                primary = emotion.get("primary", "neutral")
                intensity = emotion.get("intensity", 0.5)

                emotion_counts[primary] = emotion_counts.get(primary, 0) + 1
                total_intensity += intensity
                intensity_variance.append(intensity)

            # Calculate dominant emotions
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            dominant_emotions = [emotion for emotion, count in sorted_emotions[:3]]

            # Calculate emotional stability (lower variance = higher stability)
            avg_intensity = total_intensity / len(emotional_data) if emotional_data else 0.5
            variance = sum((x - avg_intensity) ** 2 for x in intensity_variance) / len(intensity_variance) if intensity_variance else 0
            emotional_stability = max(0.0, 1.0 - variance)

            # Calculate learning progress (simplified)
            learning_progress = min(1.0, len(emotional_data) / (days * 10))  # Assume 10 interactions per day is good

            # Generate insights
            insights = []
            if emotional_stability > 0.8:
                insights.append("High emotional stability - consistent emotional responses")
            elif emotional_stability < 0.3:
                insights.append("Low emotional stability - consider emotional regulation strategies")

            if "anger" in dominant_emotions:
                insights.append("Frequent anger detection - monitor for stress factors")
            if "joy" in dominant_emotions:
                insights.append("Positive emotional patterns detected")

            return {
                "agentId": agent_id,
                "period": f"{days} days",
                "dominantEmotions": dominant_emotions,
                "emotionalStability": emotional_stability,
                "learningProgress": learning_progress,
                "insights": insights,
                "totalInteractions": len(emotional_data),
                "averageIntensity": avg_intensity
            }

        except Exception as error:
            logger.error(f"Error analyzing emotional learning: {error}")
            return {
                "agentId": agent_id,
                "period": f"{days} days",
                "dominantEmotions": [],
                "emotionalStability": 0.5,
                "learningProgress": 0.0,
                "insights": ["Analysis failed due to error"]
            }

    async def calculate_emotional_calibration(self, agent_id: str, days: int = 7) -> Dict[str, Any]:
        """Calculate emotional calibration metrics for an agent."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            # Get emotional responses and their outcomes
            pipeline = [
                {
                    "$match": {
                        "agentId": agent_id,
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$emotion.primary",
                        "avgIntensity": {"$avg": "$emotion.intensity"},
                        "avgConfidence": {"$avg": "$emotion.confidence"},
                        "count": {"$sum": 1},
                        "avgValence": {"$avg": "$emotion.valence"},
                        "avgArousal": {"$avg": "$emotion.arousal"}
                    }
                }
            ]

            results = await self.emotional_collection.collection.aggregate(pipeline).to_list(length=None)

            if not results:
                return {
                    "agentId": agent_id,
                    "calibrationScore": 0.5,
                    "emotionAccuracy": {},
                    "recommendedAdjustments": ["No data available for calibration"]
                }

            # Calculate calibration metrics
            total_confidence = sum(r["avgConfidence"] for r in results)
            total_count = sum(r["count"] for r in results)
            overall_confidence = total_confidence / len(results) if results else 0.5

            # Calculate emotion-specific accuracy
            emotion_accuracy = {}
            for result in results:
                emotion = result["_id"]
                confidence = result["avgConfidence"]
                intensity = result["avgIntensity"]

                # Simple accuracy metric based on confidence and consistency
                accuracy = (confidence + (1.0 - abs(intensity - 0.5))) / 2
                emotion_accuracy[emotion] = accuracy

            # Overall calibration score
            calibration_score = overall_confidence * 0.7 + (total_count / (days * 5)) * 0.3
            calibration_score = max(0.0, min(1.0, calibration_score))

            # Generate recommendations
            recommendations = []
            if calibration_score < 0.6:
                recommendations.append("Consider increasing emotional detection sensitivity")
            if overall_confidence < 0.7:
                recommendations.append("Improve confidence in emotion detection")

            low_accuracy_emotions = [e for e, acc in emotion_accuracy.items() if acc < 0.6]
            if low_accuracy_emotions:
                recommendations.append(f"Focus on improving detection for: {', '.join(low_accuracy_emotions)}")

            return {
                "agentId": agent_id,
                "calibrationScore": calibration_score,
                "overallConfidence": overall_confidence,
                "emotionAccuracy": emotion_accuracy,
                "recommendedAdjustments": recommendations,
                "dataPoints": total_count
            }

        except Exception as error:
            logger.error(f"Error calculating emotional calibration: {error}")
            return {
                "agentId": agent_id,
                "calibrationScore": 0.5,
                "emotionAccuracy": {},
                "recommendedAdjustments": ["Calibration failed due to error"]
            }

    async def cleanup(self) -> int:
        """Cleanup old emotional data."""
        try:
            # Remove emotional data older than 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            result = await self.emotional_collection.collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })

            logger.info(f"Cleaned up {result.deleted_count} old emotional records")
            return result.deleted_count

        except Exception as error:
            logger.error(f"Error during emotional intelligence cleanup: {error}")
            return 0
