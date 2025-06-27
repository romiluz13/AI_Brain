"""
Communication Protocol Manager

Multi-protocol communication management system.
Adapts communication style and protocol based on context and user preferences.

Features:
- Dynamic protocol selection and adaptation
- Communication style optimization
- Context-aware formality adjustment
- User preference learning and application
- Multi-modal communication support
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, CommunicationProtocol, CognitiveSystemType

logger = logging.getLogger(__name__)


class CommunicationProtocolManager(CognitiveSystemInterface):
    """Communication Protocol Manager - System 7 of 16"""
    
    def __init__(self, system_id: str = "communication_protocol", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Communication protocols by user
        self._user_protocols: Dict[str, CommunicationProtocol] = {}
        
        # Available communication protocols
        self._available_protocols = {
            "formal_business": {
                "formality_level": 0.9,
                "verbosity_level": 0.7,
                "style": "professional",
                "tone": "respectful"
            },
            "casual_friendly": {
                "formality_level": 0.3,
                "verbosity_level": 0.5,
                "style": "conversational",
                "tone": "friendly"
            },
            "technical_precise": {
                "formality_level": 0.6,
                "verbosity_level": 0.8,
                "style": "technical",
                "tone": "precise"
            },
            "empathetic_supportive": {
                "formality_level": 0.4,
                "verbosity_level": 0.6,
                "style": "supportive",
                "tone": "caring"
            },
            "concise_direct": {
                "formality_level": 0.5,
                "verbosity_level": 0.3,
                "style": "direct",
                "tone": "efficient"
            }
        }
        
        # Context indicators for protocol selection
        self._context_indicators = {
            "formal": ["business", "professional", "meeting", "presentation", "official"],
            "casual": ["chat", "friendly", "informal", "relaxed", "personal"],
            "technical": ["technical", "code", "algorithm", "implementation", "system"],
            "emotional": ["feeling", "emotion", "sad", "happy", "worried", "excited"],
            "urgent": ["urgent", "asap", "quickly", "immediately", "rush"]
        }
        
        # Communication effectiveness tracking
        self._effectiveness_history: Dict[str, List[Dict[str, Any]]] = {}
    
    @property
    def system_name(self) -> str:
        return "Communication Protocol Manager"
    
    @property
    def system_description(self) -> str:
        return "Multi-protocol communication management system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.COMMUNICATION_OPTIMIZATION}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.COMMUNICATION_OPTIMIZATION}
    
    async def initialize(self) -> None:
        """Initialize the Communication Protocol Manager."""
        try:
            logger.info("Initializing Communication Protocol Manager...")
            await self._load_communication_data()
            self._is_initialized = True
            logger.info("Communication Protocol Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Communication Protocol Manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Communication Protocol Manager."""
        try:
            await self._save_communication_data()
            self._is_initialized = False
            logger.info("Communication Protocol Manager shutdown complete")
        except Exception as e:
            logger.error(f"Error during Communication Protocol Manager shutdown: {e}")
    
    async def process(self, input_data: CognitiveInputData, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through communication protocol analysis."""
        if not self._is_initialized:
            raise RuntimeError("Communication Protocol Manager not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Ensure user has communication protocol
            if user_id not in self._user_protocols:
                await self._create_user_protocol(user_id, input_data)
            
            # Analyze communication context
            context_analysis = await self._analyze_communication_context(input_data)
            
            # Select optimal protocol
            optimal_protocol = await self._select_optimal_protocol(user_id, context_analysis)
            
            # Adapt communication style
            style_adaptation = await self._adapt_communication_style(user_id, context_analysis, optimal_protocol)
            
            # Generate communication recommendations
            recommendations = await self._generate_communication_recommendations(user_id, context_analysis)
            
            # Update protocol effectiveness
            await self._update_protocol_effectiveness(user_id, optimal_protocol, context or {})
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.85,
                "communication_protocol": {
                    "active_protocol": optimal_protocol,
                    "formality_level": self._user_protocols[user_id].formality_level,
                    "verbosity_level": self._user_protocols[user_id].verbosity_level,
                    "effectiveness": self._user_protocols[user_id].communication_effectiveness
                },
                "context_analysis": context_analysis,
                "style_adaptation": style_adaptation,
                "recommendations": recommendations,
                "available_protocols": list(self._available_protocols.keys())
            }
            
        except Exception as e:
            logger.error(f"Error in Communication Protocol processing: {e}")
            return {"system": self.system_id, "status": "error", "error": str(e), "confidence": 0.0}
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current communication protocol state."""
        state_data = {
            "total_users": len(self._user_protocols),
            "available_protocols": len(self._available_protocols)
        }
        
        if user_id and user_id in self._user_protocols:
            protocol = self._user_protocols[user_id]
            state_data.update({
                "user_active_protocol": protocol.active_protocol,
                "user_formality_level": protocol.formality_level,
                "user_effectiveness": protocol.communication_effectiveness
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.COMMUNICATION_PROTOCOL,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.9,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update communication protocol state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Communication Protocol state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for communication protocol processing."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Public communication methods
    
    async def set_user_protocol(self, user_id: str, protocol_name: str) -> bool:
        """Set communication protocol for a user."""
        if protocol_name not in self._available_protocols:
            return False
        
        if user_id not in self._user_protocols:
            await self._create_user_protocol(user_id, None)
        
        protocol_config = self._available_protocols[protocol_name]
        user_protocol = self._user_protocols[user_id]
        
        user_protocol.active_protocol = protocol_name
        user_protocol.formality_level = protocol_config["formality_level"]
        user_protocol.verbosity_level = protocol_config["verbosity_level"]
        
        return True
    
    async def get_communication_style(self, user_id: str) -> Dict[str, Any]:
        """Get current communication style for a user."""
        if user_id not in self._user_protocols:
            return {"protocol": "casual_friendly", "formality": 0.3, "verbosity": 0.5}
        
        protocol = self._user_protocols[user_id]
        return {
            "protocol": protocol.active_protocol,
            "formality": protocol.formality_level,
            "verbosity": protocol.verbosity_level,
            "effectiveness": protocol.communication_effectiveness
        }
    
    # Private methods
    
    async def _load_communication_data(self) -> None:
        """Load communication data from storage."""
        logger.debug("Communication data loaded")
    
    async def _save_communication_data(self) -> None:
        """Save communication data to storage."""
        logger.debug("Communication data saved")
    
    async def _create_user_protocol(self, user_id: str, input_data: Optional[CognitiveInputData]) -> None:
        """Create communication protocol for a user."""
        # Default to casual_friendly protocol
        default_protocol = "casual_friendly"
        protocol_config = self._available_protocols[default_protocol]
        
        # Analyze input to determine better initial protocol
        if input_data and input_data.text:
            context_analysis = await self._analyze_communication_context(input_data)
            if context_analysis["context_type"] == "formal":
                default_protocol = "formal_business"
            elif context_analysis["context_type"] == "technical":
                default_protocol = "technical_precise"
            elif context_analysis["context_type"] == "emotional":
                default_protocol = "empathetic_supportive"
        
        protocol_config = self._available_protocols[default_protocol]
        
        communication_protocol = CommunicationProtocol(
            active_protocol=default_protocol,
            available_protocols=list(self._available_protocols.keys()),
            formality_level=protocol_config["formality_level"],
            verbosity_level=protocol_config["verbosity_level"],
            communication_effectiveness=0.8,
            user_satisfaction=0.8
        )
        
        self._user_protocols[user_id] = communication_protocol
        logger.debug(f"Created communication protocol for user {user_id}: {default_protocol}")
    
    async def _analyze_communication_context(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze communication context from input."""
        text = input_data.text or ""
        text_lower = text.lower()
        
        # Analyze context indicators
        context_scores = {}
        for context_type, indicators in self._context_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                context_scores[context_type] = score
        
        # Determine primary context
        primary_context = max(context_scores, key=context_scores.get) if context_scores else "casual"
        
        # Analyze communication characteristics
        word_count = len(text.split()) if text else 0
        sentence_count = text.count('.') + text.count('!') + text.count('?') if text else 0
        
        # Assess urgency
        urgency_level = context_scores.get("urgent", 0) / max(1, word_count / 10)
        
        # Assess emotional content
        emotional_level = context_scores.get("emotional", 0) / max(1, word_count / 10)
        
        return {
            "context_type": primary_context,
            "context_scores": context_scores,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "urgency_level": min(1.0, urgency_level),
            "emotional_level": min(1.0, emotional_level),
            "complexity_level": self._assess_text_complexity(text)
        }
    
    async def _select_optimal_protocol(self, user_id: str, context_analysis: Dict[str, Any]) -> str:
        """Select optimal communication protocol based on context."""
        current_protocol = self._user_protocols[user_id].active_protocol
        context_type = context_analysis["context_type"]
        
        # Protocol selection logic
        if context_type == "formal":
            optimal_protocol = "formal_business"
        elif context_type == "technical":
            optimal_protocol = "technical_precise"
        elif context_type == "emotional":
            optimal_protocol = "empathetic_supportive"
        elif context_analysis["urgency_level"] > 0.7:
            optimal_protocol = "concise_direct"
        else:
            optimal_protocol = "casual_friendly"
        
        # Update user protocol if different
        if optimal_protocol != current_protocol:
            await self.set_user_protocol(user_id, optimal_protocol)
        
        return optimal_protocol
    
    async def _adapt_communication_style(
        self, 
        user_id: str, 
        context_analysis: Dict[str, Any], 
        protocol: str
    ) -> Dict[str, Any]:
        """Adapt communication style based on context and protocol."""
        protocol_config = self._available_protocols[protocol]
        user_protocol = self._user_protocols[user_id]
        
        # Adjust formality based on context
        base_formality = protocol_config["formality_level"]
        if context_analysis["context_type"] == "formal":
            adjusted_formality = min(1.0, base_formality + 0.2)
        elif context_analysis["urgency_level"] > 0.7:
            adjusted_formality = max(0.3, base_formality - 0.1)
        else:
            adjusted_formality = base_formality
        
        # Adjust verbosity based on context
        base_verbosity = protocol_config["verbosity_level"]
        if context_analysis["urgency_level"] > 0.7:
            adjusted_verbosity = max(0.2, base_verbosity - 0.3)
        elif context_analysis["complexity_level"] > 0.7:
            adjusted_verbosity = min(1.0, base_verbosity + 0.2)
        else:
            adjusted_verbosity = base_verbosity
        
        # Update user protocol
        user_protocol.formality_level = adjusted_formality
        user_protocol.verbosity_level = adjusted_verbosity
        
        return {
            "protocol": protocol,
            "adjusted_formality": adjusted_formality,
            "adjusted_verbosity": adjusted_verbosity,
            "style_recommendations": self._get_style_recommendations(protocol, context_analysis)
        }
    
    async def _generate_communication_recommendations(self, user_id: str, context_analysis: Dict[str, Any]) -> List[str]:
        """Generate communication recommendations."""
        recommendations = []
        user_protocol = self._user_protocols[user_id]
        
        if context_analysis["urgency_level"] > 0.7:
            recommendations.append("Use concise, direct communication for urgent matters")
        
        if context_analysis["emotional_level"] > 0.5:
            recommendations.append("Consider using empathetic language and acknowledgment")
        
        if context_analysis["complexity_level"] > 0.7:
            recommendations.append("Break down complex information into digestible parts")
        
        if user_protocol.communication_effectiveness < 0.6:
            recommendations.append("Consider adjusting communication style based on user feedback")
        
        return recommendations
    
    async def _update_protocol_effectiveness(self, user_id: str, protocol: str, context: Dict[str, Any]) -> None:
        """Update protocol effectiveness based on context."""
        # Simple effectiveness tracking - in production would use user feedback
        if user_id not in self._effectiveness_history:
            self._effectiveness_history[user_id] = []
        
        effectiveness_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "protocol": protocol,
            "context_type": context.get("context_type", "unknown"),
            "effectiveness": 0.8  # Default effectiveness
        }
        
        self._effectiveness_history[user_id].append(effectiveness_entry)
        
        # Keep only recent history
        if len(self._effectiveness_history[user_id]) > 50:
            self._effectiveness_history[user_id] = self._effectiveness_history[user_id][-50:]
    
    def _assess_text_complexity(self, text: str) -> float:
        """Assess text complexity."""
        if not text:
            return 0.0
        
        word_count = len(text.split())
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        avg_words_per_sentence = word_count / sentence_count
        
        # Simple complexity assessment
        complexity = 0.0
        if avg_words_per_sentence > 20:
            complexity += 0.4
        elif avg_words_per_sentence > 15:
            complexity += 0.3
        elif avg_words_per_sentence > 10:
            complexity += 0.2
        
        # Check for technical terms
        technical_terms = ["algorithm", "implementation", "configuration", "optimization"]
        technical_count = sum(1 for term in technical_terms if term in text.lower())
        complexity += min(0.3, technical_count * 0.1)
        
        return min(1.0, complexity)
    
    def _get_style_recommendations(self, protocol: str, context_analysis: Dict[str, Any]) -> List[str]:
        """Get style recommendations for a protocol."""
        protocol_config = self._available_protocols[protocol]
        recommendations = []
        
        if protocol_config["style"] == "professional":
            recommendations.extend(["Use formal language", "Be respectful and courteous"])
        elif protocol_config["style"] == "conversational":
            recommendations.extend(["Use friendly tone", "Be approachable and warm"])
        elif protocol_config["style"] == "technical":
            recommendations.extend(["Be precise and accurate", "Use appropriate technical terminology"])
        elif protocol_config["style"] == "supportive":
            recommendations.extend(["Show empathy and understanding", "Offer helpful guidance"])
        elif protocol_config["style"] == "direct":
            recommendations.extend(["Be concise and clear", "Focus on key points"])
        
        return recommendations
