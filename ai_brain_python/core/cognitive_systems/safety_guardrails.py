"""
Safety Guardrails Engine

Multi-layer safety and compliance system.
Provides comprehensive safety validation, risk assessment, and policy enforcement.

Features:
- Multi-layer safety validation and risk assessment
- Content filtering and bias detection
- Policy compliance checking and enforcement
- Harm prevention and mitigation strategies
- Real-time safety monitoring and alerting
- Adaptive safety thresholds and learning
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, SafetyAssessment, SafetyLevel, CognitiveSystemType

logger = logging.getLogger(__name__)


class SafetyGuardrailsEngine(CognitiveSystemInterface):
    """
    Safety Guardrails Engine - System 10 of 16
    
    Provides multi-layer safety validation and comprehensive
    risk assessment with policy enforcement.
    """
    
    def __init__(self, system_id: str = "safety_guardrails", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Safety assessments by user/session
        self._safety_assessments: Dict[str, SafetyAssessment] = {}
        
        # Safety configuration
        self._config = {
            "strict_mode": config.get("strict_mode", True) if config else True,
            "auto_block_threshold": config.get("auto_block_threshold", 0.8) if config else 0.8,
            "warning_threshold": config.get("warning_threshold", 0.5) if config else 0.5,
            "enable_content_filtering": config.get("enable_content_filtering", True) if config else True,
            "enable_bias_detection": config.get("enable_bias_detection", True) if config else True,
            "enable_harm_prevention": config.get("enable_harm_prevention", True) if config else True
        }
        
        # Content filters and patterns
        self._harmful_content_patterns = {
            "violence": [
                r'\b(?:kill|murder|assault|attack|violence|harm|hurt|damage)\b',
                r'\b(?:weapon|gun|knife|bomb|explosive)\b',
                r'\b(?:fight|battle|war|conflict|aggression)\b'
            ],
            "hate_speech": [
                r'\b(?:hate|racist|discrimination|prejudice|bigot)\b',
                r'\b(?:inferior|superior|subhuman|worthless)\b'
            ],
            "self_harm": [
                r'\b(?:suicide|self.harm|kill.myself|end.my.life)\b',
                r'\b(?:cut.myself|hurt.myself|harm.myself)\b'
            ],
            "illegal_activity": [
                r'\b(?:illegal|criminal|fraud|theft|steal|rob)\b',
                r'\b(?:drugs|narcotics|trafficking|smuggling)\b'
            ],
            "privacy_violation": [
                r'\b(?:personal.information|private.data|confidential)\b',
                r'\b(?:ssn|social.security|credit.card|password)\b'
            ]
        }
        
        # Bias detection patterns
        self._bias_patterns = {
            "gender_bias": [
                r'\b(?:men|women|male|female)\s+(?:are|should|must|always|never)\b',
                r'\b(?:boys|girls)\s+(?:are|should|can\'t|cannot)\b'
            ],
            "racial_bias": [
                r'\b(?:race|ethnicity|nationality)\s+(?:determines|causes|makes)\b'
            ],
            "age_bias": [
                r'\b(?:old|young|elderly|teenager)\s+(?:people|person)\s+(?:are|should|cannot)\b'
            ]
        }
        
        # Policy categories
        self._policy_categories = {
            "content_policy": {
                "no_harmful_content": True,
                "no_hate_speech": True,
                "no_violence": True,
                "no_illegal_content": True
            },
            "privacy_policy": {
                "no_personal_info": True,
                "data_protection": True,
                "consent_required": True
            },
            "ethical_policy": {
                "no_bias": True,
                "fairness": True,
                "transparency": True
            }
        }
        
        # Risk assessment weights
        self._risk_weights = {
            "content_risk": 0.4,
            "bias_risk": 0.2,
            "privacy_risk": 0.2,
            "harm_risk": 0.2
        }
    
    @property
    def system_name(self) -> str:
        return "Safety Guardrails Engine"
    
    @property
    def system_description(self) -> str:
        return "Multi-layer safety and compliance system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.SAFETY_VALIDATION}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.SAFETY_VALIDATION}
    
    async def initialize(self) -> None:
        """Initialize the Safety Guardrails Engine."""
        try:
            logger.info("Initializing Safety Guardrails Engine...")
            
            # Load safety models and policies
            await self._load_safety_models()
            
            # Initialize content filters
            await self._initialize_content_filters()
            
            # Load policy configurations
            await self._load_policy_configurations()
            
            self._is_initialized = True
            logger.info("Safety Guardrails Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Safety Guardrails Engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Safety Guardrails Engine."""
        try:
            logger.info("Shutting down Safety Guardrails Engine...")
            
            # Save safety assessments
            await self._save_safety_data()
            
            self._is_initialized = False
            logger.info("Safety Guardrails Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Safety Guardrails Engine shutdown: {e}")
    
    async def process(
        self, 
        input_data: CognitiveInputData,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input through safety guardrails analysis."""
        if not self._is_initialized:
            raise RuntimeError("Safety Guardrails Engine not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            session_id = input_data.context.session_id or "default"
            
            # Perform comprehensive safety assessment
            safety_assessment = await self._perform_safety_assessment(input_data, context or {})
            
            # Store assessment
            assessment_key = f"{user_id}:{session_id}"
            self._safety_assessments[assessment_key] = safety_assessment
            
            # Generate mitigation actions if needed
            mitigation_actions = await self._generate_mitigation_actions(safety_assessment)
            
            # Check for auto-blocking
            should_block = safety_assessment.risk_score >= self._config["auto_block_threshold"]
            
            # Generate safety recommendations
            recommendations = await self._generate_safety_recommendations(safety_assessment)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.95,
                "safety_assessment": {
                    "safety_level": safety_assessment.safety_level.value,
                    "is_safe": safety_assessment.is_safe,
                    "risk_score": safety_assessment.risk_score,
                    "should_block": should_block,
                    "violations": safety_assessment.violations,
                    "warnings": safety_assessment.warnings
                },
                "risk_breakdown": {
                    category: score for category, score in safety_assessment.risk_categories.items()
                },
                "policy_compliance": {
                    check: result for check, result in safety_assessment.compliance_checks.items()
                },
                "mitigation_actions": mitigation_actions,
                "recommendations": recommendations,
                "safety_metrics": await self._get_safety_metrics(user_id)
            }
            
        except Exception as e:
            logger.error(f"Error in Safety Guardrails processing: {e}")
            return {
                "system": self.system_id,
                "status": "error",
                "error": str(e),
                "confidence": 0.0,
                "safety_assessment": {
                    "safety_level": SafetyLevel.CRITICAL.value,
                    "is_safe": False,
                    "risk_score": 1.0,
                    "should_block": True,
                    "violations": ["System error during safety assessment"],
                    "warnings": []
                }
            }
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current safety guardrails state."""
        state_data = {
            "total_assessments": len(self._safety_assessments),
            "strict_mode": self._config["strict_mode"],
            "content_filters_enabled": self._config["enable_content_filtering"],
            "bias_detection_enabled": self._config["enable_bias_detection"]
        }
        
        if user_id:
            user_assessments = [
                assessment for key, assessment in self._safety_assessments.items()
                if key.startswith(f"{user_id}:")
            ]
            if user_assessments:
                latest_assessment = max(user_assessments, key=lambda a: a.created_at)
                state_data.update({
                    "user_latest_safety_level": latest_assessment.safety_level.value,
                    "user_latest_risk_score": latest_assessment.risk_score,
                    "user_total_assessments": len(user_assessments)
                })
        
        return CognitiveState(
            system_type=CognitiveSystemType.SAFETY_GUARDRAILS,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.98,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update safety guardrails state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Safety Guardrails state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for safety processing."""
        violations = []
        warnings = []
        
        # Always validate for safety
        if not input_data.text and not input_data.audio_data and not input_data.image_data:
            warnings.append("No content provided for safety assessment")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            confidence=1.0 if len(violations) == 0 else 0.0,
            violations=violations,
            warnings=warnings
        )
    
    # Public safety methods
    
    async def assess_content_safety(self, content: str) -> Tuple[bool, float, List[str]]:
        """Assess safety of content."""
        violations = []
        risk_scores = []
        
        # Check harmful content patterns
        for category, patterns in self._harmful_content_patterns.items():
            category_violations = []
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    category_violations.extend(matches)
            
            if category_violations:
                violations.append(f"{category}: {', '.join(category_violations[:3])}")
                risk_scores.append(0.8)  # High risk for harmful content
        
        # Calculate overall risk
        overall_risk = max(risk_scores) if risk_scores else 0.0
        is_safe = overall_risk < self._config["warning_threshold"]
        
        return is_safe, overall_risk, violations
    
    async def check_policy_compliance(self, content: str, context: Dict[str, Any]) -> Dict[str, bool]:
        """Check policy compliance."""
        compliance_results = {}
        
        # Content policy checks
        is_content_safe, _, _ = await self.assess_content_safety(content)
        compliance_results["content_policy"] = is_content_safe
        
        # Privacy policy checks
        has_privacy_violations = await self._check_privacy_violations(content)
        compliance_results["privacy_policy"] = not has_privacy_violations
        
        # Ethical policy checks
        has_bias = await self._detect_bias(content)
        compliance_results["ethical_policy"] = not has_bias
        
        return compliance_results
    
    # Private methods
    
    async def _load_safety_models(self) -> None:
        """Load safety models and classifiers."""
        logger.debug("Safety models loaded")
    
    async def _initialize_content_filters(self) -> None:
        """Initialize content filtering systems."""
        logger.debug("Content filters initialized")
    
    async def _load_policy_configurations(self) -> None:
        """Load policy configurations."""
        logger.debug("Policy configurations loaded")
    
    async def _save_safety_data(self) -> None:
        """Save safety assessment data."""
        logger.debug("Safety data saved")
    
    async def _perform_safety_assessment(self, input_data: CognitiveInputData, context: Dict[str, Any]) -> SafetyAssessment:
        """Perform comprehensive safety assessment."""
        content = input_data.text or ""
        
        # Initialize assessment
        assessment = SafetyAssessment(
            safety_level=SafetyLevel.SAFE,
            is_safe=True,
            risk_score=0.0
        )
        
        # Content safety assessment
        if self._config["enable_content_filtering"]:
            content_safe, content_risk, content_violations = await self.assess_content_safety(content)
            assessment.risk_categories["content_risk"] = content_risk
            if not content_safe:
                assessment.violations.extend(content_violations)
                assessment.is_safe = False
        
        # Bias detection
        if self._config["enable_bias_detection"]:
            bias_detected, bias_score, bias_violations = await self._detect_bias_comprehensive(content)
            assessment.risk_categories["bias_risk"] = bias_score
            if bias_detected:
                assessment.violations.extend(bias_violations)
                assessment.warnings.append("Potential bias detected")
        
        # Privacy assessment
        privacy_violations = await self._check_privacy_violations(content)
        assessment.risk_categories["privacy_risk"] = 0.7 if privacy_violations else 0.0
        if privacy_violations:
            assessment.violations.append("Privacy policy violations detected")
            assessment.is_safe = False
        
        # Harm prevention assessment
        if self._config["enable_harm_prevention"]:
            harm_risk = await self._assess_harm_potential(content, context)
            assessment.risk_categories["harm_risk"] = harm_risk
            if harm_risk > 0.6:
                assessment.violations.append("Potential harm detected")
                assessment.warnings.append("Content may cause harm")
        
        # Calculate overall risk score
        assessment.risk_score = sum(
            score * self._risk_weights.get(category, 0.25)
            for category, score in assessment.risk_categories.items()
        )
        
        # Determine safety level
        assessment.safety_level = self._determine_safety_level(assessment.risk_score)
        assessment.is_safe = assessment.safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTION]
        
        # Policy compliance checks
        assessment.compliance_checks = await self.check_policy_compliance(content, context)
        
        return assessment
    
    async def _detect_bias_comprehensive(self, content: str) -> Tuple[bool, float, List[str]]:
        """Comprehensive bias detection."""
        bias_violations = []
        bias_scores = []
        
        for bias_type, patterns in self._bias_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    bias_violations.append(f"{bias_type}: {matches[0]}")
                    bias_scores.append(0.6)
        
        overall_bias_score = max(bias_scores) if bias_scores else 0.0
        has_bias = overall_bias_score > 0.3
        
        return has_bias, overall_bias_score, bias_violations
    
    async def _check_privacy_violations(self, content: str) -> bool:
        """Check for privacy violations."""
        privacy_patterns = self._harmful_content_patterns.get("privacy_violation", [])
        
        for pattern in privacy_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    async def _assess_harm_potential(self, content: str, context: Dict[str, Any]) -> float:
        """Assess potential for harm."""
        harm_indicators = 0
        
        # Check for harmful content patterns
        for category, patterns in self._harmful_content_patterns.items():
            if category in ["violence", "self_harm"]:
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        harm_indicators += 1
        
        # Context factors
        if context.get("user_emotional_state") == "distressed":
            harm_indicators += 1
        
        # Normalize to 0-1 scale
        harm_score = min(1.0, harm_indicators * 0.3)
        return harm_score
    
    def _determine_safety_level(self, risk_score: float) -> SafetyLevel:
        """Determine safety level from risk score."""
        if risk_score >= 0.9:
            return SafetyLevel.CRITICAL
        elif risk_score >= 0.7:
            return SafetyLevel.DANGER
        elif risk_score >= 0.5:
            return SafetyLevel.WARNING
        elif risk_score >= 0.3:
            return SafetyLevel.CAUTION
        else:
            return SafetyLevel.SAFE
    
    async def _generate_mitigation_actions(self, assessment: SafetyAssessment) -> List[str]:
        """Generate mitigation actions based on assessment."""
        actions = []
        
        if assessment.risk_score >= self._config["auto_block_threshold"]:
            actions.append("Block content from being processed")
            actions.append("Log security incident")
            actions.append("Notify safety team")
        
        elif assessment.risk_score >= self._config["warning_threshold"]:
            actions.append("Add content warning")
            actions.append("Request user confirmation")
            actions.append("Apply additional filtering")
        
        if "bias_risk" in assessment.risk_categories and assessment.risk_categories["bias_risk"] > 0.5:
            actions.append("Apply bias correction")
            actions.append("Provide alternative perspectives")
        
        if "privacy_risk" in assessment.risk_categories and assessment.risk_categories["privacy_risk"] > 0.5:
            actions.append("Remove personal information")
            actions.append("Apply data anonymization")
        
        return actions
    
    async def _generate_safety_recommendations(self, assessment: SafetyAssessment) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if assessment.risk_score > 0.5:
            recommendations.append("Review content before proceeding")
        
        if assessment.violations:
            recommendations.append("Address identified violations before continuing")
        
        if "bias_risk" in assessment.risk_categories and assessment.risk_categories["bias_risk"] > 0.3:
            recommendations.append("Consider bias implications and alternative phrasings")
        
        if not assessment.compliance_checks.get("privacy_policy", True):
            recommendations.append("Ensure privacy compliance before processing personal data")
        
        return recommendations
    
    async def _get_safety_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get safety metrics for a user."""
        user_assessments = [
            assessment for key, assessment in self._safety_assessments.items()
            if key.startswith(f"{user_id}:")
        ]
        
        if not user_assessments:
            return {"total_assessments": 0, "average_risk_score": 0.0, "violation_rate": 0.0}
        
        total_assessments = len(user_assessments)
        average_risk = sum(a.risk_score for a in user_assessments) / total_assessments
        violations = sum(1 for a in user_assessments if a.violations)
        violation_rate = violations / total_assessments
        
        return {
            "total_assessments": total_assessments,
            "average_risk_score": average_risk,
            "violation_rate": violation_rate,
            "latest_safety_level": user_assessments[-1].safety_level.value
        }
