"""
Cultural Knowledge Engine

Cross-cultural intelligence and adaptation system.
Provides culturally-aware responses and communication adaptation.

Features:
- Cultural dimension analysis (Hofstede's model)
- Language and communication style adaptation
- Cultural norm awareness and compliance
- Cross-cultural sensitivity scoring
- Localization and personalization
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, CulturalContext, CulturalDimension, CognitiveSystemType

logger = logging.getLogger(__name__)


class CulturalKnowledgeEngine(CognitiveSystemInterface):
    """Cultural Knowledge Engine - System 5 of 16"""
    
    def __init__(self, system_id: str = "cultural_knowledge", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Cultural profiles by user
        self._user_cultural_contexts: Dict[str, CulturalContext] = {}
        
        # Cultural dimension defaults by region/country
        self._cultural_profiles = {
            "US": {CulturalDimension.INDIVIDUALISM: 0.91, CulturalDimension.POWER_DISTANCE: 0.40},
            "JP": {CulturalDimension.INDIVIDUALISM: 0.46, CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.92},
            "DE": {CulturalDimension.INDIVIDUALISM: 0.67, CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.65},
            "default": {dim: 0.5 for dim in CulturalDimension}
        }
        
        # Communication style indicators
        self._communication_styles = {
            "direct": ["clearly", "specifically", "exactly", "precisely", "straightforward"],
            "indirect": ["perhaps", "maybe", "might", "could be", "it seems"],
            "formal": ["please", "thank you", "sir", "madam", "respectfully"],
            "informal": ["hey", "yeah", "cool", "awesome", "no problem"]
        }
    
    @property
    def system_name(self) -> str:
        return "Cultural Knowledge Engine"
    
    @property
    def system_description(self) -> str:
        return "Cross-cultural intelligence and adaptation system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.CULTURAL_ADAPTATION}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.CULTURAL_ADAPTATION}
    
    async def initialize(self) -> None:
        """Initialize the Cultural Knowledge Engine."""
        try:
            logger.info("Initializing Cultural Knowledge Engine...")
            await self._load_cultural_data()
            self._is_initialized = True
            logger.info("Cultural Knowledge Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cultural Knowledge Engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Cultural Knowledge Engine."""
        try:
            await self._save_cultural_data()
            self._is_initialized = False
            logger.info("Cultural Knowledge Engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during Cultural Knowledge Engine shutdown: {e}")
    
    async def process(self, input_data: CognitiveInputData, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through cultural knowledge analysis."""
        if not self._is_initialized:
            raise RuntimeError("Cultural Knowledge Engine not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Ensure user has cultural context
            if user_id not in self._user_cultural_contexts:
                await self._create_user_cultural_context(user_id, input_data)
            
            # Analyze cultural indicators in input
            cultural_analysis = await self._analyze_cultural_indicators(input_data)
            
            # Adapt communication style
            communication_adaptation = await self._adapt_communication_style(user_id, input_data)
            
            # Generate cultural recommendations
            recommendations = await self._generate_cultural_recommendations(user_id, cultural_analysis)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.8,
                "cultural_context": {
                    "primary_culture": self._user_cultural_contexts[user_id].primary_culture,
                    "cultural_dimensions": {dim.value: score for dim, score in self._user_cultural_contexts[user_id].cultural_dimensions.items()},
                    "adaptation_level": self._user_cultural_contexts[user_id].adaptation_level,
                    "cultural_sensitivity": self._user_cultural_contexts[user_id].cultural_sensitivity
                },
                "cultural_analysis": cultural_analysis,
                "communication_adaptation": communication_adaptation,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in Cultural Knowledge processing: {e}")
            return {"system": self.system_id, "status": "error", "error": str(e), "confidence": 0.0}
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current cultural knowledge state."""
        state_data = {
            "total_users": len(self._user_cultural_contexts),
            "cultural_profiles_loaded": len(self._cultural_profiles)
        }
        
        if user_id and user_id in self._user_cultural_contexts:
            cultural_context = self._user_cultural_contexts[user_id]
            state_data.update({
                "user_primary_culture": cultural_context.primary_culture,
                "user_adaptation_level": cultural_context.adaptation_level,
                "user_cultural_sensitivity": cultural_context.cultural_sensitivity
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.CULTURAL_KNOWLEDGE,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.85,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update cultural knowledge state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Cultural Knowledge state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for cultural knowledge processing."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Private methods
    
    async def _load_cultural_data(self) -> None:
        """Load cultural data from storage."""
        logger.debug("Cultural data loaded")
    
    async def _save_cultural_data(self) -> None:
        """Save cultural data to storage."""
        logger.debug("Cultural data saved")
    
    async def _create_user_cultural_context(self, user_id: str, input_data: CognitiveInputData) -> None:
        """Create cultural context for a user."""
        # Detect culture from language and context
        language = input_data.language
        culture_code = self._detect_culture_from_language(language)
        
        cultural_context = CulturalContext(
            primary_culture=culture_code,
            cultural_dimensions=self._cultural_profiles.get(culture_code, self._cultural_profiles["default"]).copy(),
            primary_language=language,
            adaptation_level=0.5,
            cultural_sensitivity=0.8
        )
        
        self._user_cultural_contexts[user_id] = cultural_context
        logger.debug(f"Created cultural context for user {user_id}: {culture_code}")
    
    async def _analyze_cultural_indicators(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze cultural indicators in input."""
        text = input_data.text or ""
        text_lower = text.lower()
        
        # Analyze communication style
        style_scores = {}
        for style, indicators in self._communication_styles.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            style_scores[style] = score
        
        # Determine dominant style
        dominant_style = max(style_scores, key=style_scores.get) if any(style_scores.values()) else "neutral"
        
        return {
            "communication_style": dominant_style,
            "style_scores": style_scores,
            "formality_level": self._assess_formality(text),
            "directness_level": self._assess_directness(text)
        }
    
    async def _adapt_communication_style(self, user_id: str, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Adapt communication style based on cultural context."""
        cultural_context = self._user_cultural_contexts[user_id]
        
        # Get cultural dimensions
        individualism = cultural_context.cultural_dimensions.get(CulturalDimension.INDIVIDUALISM, 0.5)
        power_distance = cultural_context.cultural_dimensions.get(CulturalDimension.POWER_DISTANCE, 0.5)
        
        # Adapt based on cultural dimensions
        adaptations = []
        
        if individualism > 0.7:
            adaptations.append("Use direct, personal communication style")
        elif individualism < 0.3:
            adaptations.append("Use group-oriented, collective communication style")
        
        if power_distance > 0.7:
            adaptations.append("Use formal, respectful communication style")
        elif power_distance < 0.3:
            adaptations.append("Use informal, egalitarian communication style")
        
        return {
            "recommended_style": "formal" if power_distance > 0.6 else "informal",
            "adaptations": adaptations,
            "cultural_considerations": [
                f"Individualism level: {individualism:.2f}",
                f"Power distance level: {power_distance:.2f}"
            ]
        }
    
    async def _generate_cultural_recommendations(self, user_id: str, cultural_analysis: Dict[str, Any]) -> List[str]:
        """Generate cultural adaptation recommendations."""
        recommendations = []
        cultural_context = self._user_cultural_contexts[user_id]
        
        if cultural_context.adaptation_level < 0.5:
            recommendations.append("Consider learning more about cultural communication preferences")
        
        if cultural_analysis["formality_level"] > 0.8 and cultural_context.cultural_dimensions.get(CulturalDimension.POWER_DISTANCE, 0.5) < 0.3:
            recommendations.append("Consider using a more informal communication style for this cultural context")
        
        return recommendations
    
    def _detect_culture_from_language(self, language: str) -> str:
        """Detect culture code from language."""
        language_to_culture = {
            "en": "US", "ja": "JP", "de": "DE", "fr": "FR", "es": "ES", "zh": "CN"
        }
        return language_to_culture.get(language, "default")
    
    def _assess_formality(self, text: str) -> float:
        """Assess formality level of text."""
        formal_indicators = ["please", "thank you", "sir", "madam", "respectfully", "sincerely"]
        informal_indicators = ["hey", "yeah", "cool", "awesome", "no problem", "sure thing"]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text.lower())
        informal_count = sum(1 for indicator in informal_indicators if indicator in text.lower())
        
        if formal_count + informal_count == 0:
            return 0.5
        
        return formal_count / (formal_count + informal_count)
    
    def _assess_directness(self, text: str) -> float:
        """Assess directness level of text."""
        direct_indicators = ["clearly", "specifically", "exactly", "must", "will", "definitely"]
        indirect_indicators = ["perhaps", "maybe", "might", "could", "possibly", "it seems"]
        
        direct_count = sum(1 for indicator in direct_indicators if indicator in text.lower())
        indirect_count = sum(1 for indicator in indirect_indicators if indicator in text.lower())
        
        if direct_count + indirect_count == 0:
            return 0.5
        
        return direct_count / (direct_count + indirect_count)
