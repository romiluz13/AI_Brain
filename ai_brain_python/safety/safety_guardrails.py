"""
Safety Guardrails System for AI Brain.

Provides comprehensive safety checks including content filtering,
PII detection, harmful content detection, and compliance monitoring.
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    """Safety levels for content filtering."""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class ThreatType(str, Enum):
    """Types of safety threats."""
    PII_EXPOSURE = "pii_exposure"
    HARMFUL_CONTENT = "harmful_content"
    BIAS_DETECTED = "bias_detected"
    MISINFORMATION = "misinformation"
    INAPPROPRIATE_LANGUAGE = "inappropriate_language"
    PRIVACY_VIOLATION = "privacy_violation"
    SECURITY_RISK = "security_risk"


@dataclass
class SafetyViolation:
    """Represents a safety violation."""
    threat_type: ThreatType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    confidence: float
    location: Optional[str] = None
    suggested_action: Optional[str] = None


class SafetyConfig(BaseModel):
    """Configuration for safety guardrails."""
    
    safety_level: SafetyLevel = Field(default=SafetyLevel.MODERATE)
    enable_pii_detection: bool = Field(default=True)
    enable_harmful_content_detection: bool = Field(default=True)
    enable_bias_detection: bool = Field(default=True)
    enable_compliance_logging: bool = Field(default=True)
    
    # PII Detection settings
    pii_patterns: Dict[str, str] = Field(default_factory=lambda: {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    })
    
    # Harmful content keywords
    harmful_keywords: Set[str] = Field(default_factory=lambda: {
        "violence", "hate", "discrimination", "harassment", "threat",
        "illegal", "dangerous", "harmful", "toxic", "abuse"
    })
    
    # Bias detection keywords
    bias_keywords: Set[str] = Field(default_factory=lambda: {
        "stereotype", "prejudice", "discrimination", "bias", "unfair"
    })
    
    # Compliance requirements
    compliance_standards: List[str] = Field(default_factory=lambda: [
        "GDPR", "CCPA", "HIPAA", "SOX", "PCI_DSS"
    ])


class PIIDetector:
    """Detects Personally Identifiable Information in text."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in config.pii_patterns.items()
        }
    
    async def detect_pii(self, text: str) -> List[SafetyViolation]:
        """Detect PII in the given text."""
        violations = []
        
        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            
            for match in matches:
                violation = SafetyViolation(
                    threat_type=ThreatType.PII_EXPOSURE,
                    severity="high",
                    description=f"Detected {pii_type}: {match[:10]}...",
                    confidence=0.9,
                    location=f"Position: {text.find(match)}",
                    suggested_action=f"Remove or mask {pii_type}"
                )
                violations.append(violation)
        
        return violations
    
    def mask_pii(self, text: str) -> str:
        """Mask PII in text with placeholder values."""
        masked_text = text
        
        for pii_type, pattern in self.compiled_patterns.items():
            if pii_type == "email":
                masked_text = pattern.sub("[EMAIL_REDACTED]", masked_text)
            elif pii_type == "phone":
                masked_text = pattern.sub("[PHONE_REDACTED]", masked_text)
            elif pii_type == "ssn":
                masked_text = pattern.sub("[SSN_REDACTED]", masked_text)
            elif pii_type == "credit_card":
                masked_text = pattern.sub("[CARD_REDACTED]", masked_text)
            elif pii_type == "ip_address":
                masked_text = pattern.sub("[IP_REDACTED]", masked_text)
        
        return masked_text


class HarmfulContentDetector:
    """Detects harmful or inappropriate content."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.harmful_keywords = config.harmful_keywords
    
    async def detect_harmful_content(self, text: str) -> List[SafetyViolation]:
        """Detect harmful content in text."""
        violations = []
        text_lower = text.lower()
        
        for keyword in self.harmful_keywords:
            if keyword in text_lower:
                violation = SafetyViolation(
                    threat_type=ThreatType.HARMFUL_CONTENT,
                    severity=self._assess_severity(keyword),
                    description=f"Detected harmful keyword: {keyword}",
                    confidence=0.8,
                    suggested_action="Review and potentially filter content"
                )
                violations.append(violation)
        
        # Advanced pattern detection
        violations.extend(await self._detect_advanced_patterns(text))
        
        return violations
    
    def _assess_severity(self, keyword: str) -> str:
        """Assess severity of harmful keyword."""
        high_severity = {"violence", "threat", "illegal", "dangerous"}
        medium_severity = {"hate", "discrimination", "harassment"}
        
        if keyword in high_severity:
            return "high"
        elif keyword in medium_severity:
            return "medium"
        else:
            return "low"
    
    async def _detect_advanced_patterns(self, text: str) -> List[SafetyViolation]:
        """Detect advanced harmful patterns."""
        violations = []
        
        # Detect potential threats
        threat_patterns = [
            r'\b(kill|hurt|harm|attack|destroy)\s+\w+',
            r'\b(bomb|weapon|gun|knife)\b',
            r'\b(suicide|self-harm)\b'
        ]
        
        for pattern in threat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                violation = SafetyViolation(
                    threat_type=ThreatType.HARMFUL_CONTENT,
                    severity="critical",
                    description=f"Detected potential threat pattern: {match}",
                    confidence=0.85,
                    suggested_action="Immediate review required"
                )
                violations.append(violation)
        
        return violations


class BiasDetector:
    """Detects potential bias in content."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.bias_keywords = config.bias_keywords
    
    async def detect_bias(self, text: str) -> List[SafetyViolation]:
        """Detect potential bias in text."""
        violations = []
        
        # Keyword-based detection
        text_lower = text.lower()
        for keyword in self.bias_keywords:
            if keyword in text_lower:
                violation = SafetyViolation(
                    threat_type=ThreatType.BIAS_DETECTED,
                    severity="medium",
                    description=f"Potential bias indicator: {keyword}",
                    confidence=0.6,
                    suggested_action="Review for bias and fairness"
                )
                violations.append(violation)
        
        # Pattern-based detection
        violations.extend(await self._detect_bias_patterns(text))
        
        return violations
    
    async def _detect_bias_patterns(self, text: str) -> List[SafetyViolation]:
        """Detect bias patterns in text."""
        violations = []
        
        # Detect stereotypical language patterns
        stereotype_patterns = [
            r'\b(all|every|most)\s+(men|women|people)\s+(are|do|have)\b',
            r'\b(typical|usually|always)\s+\w+\s+(person|people)\b'
        ]
        
        for pattern in stereotype_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                violation = SafetyViolation(
                    threat_type=ThreatType.BIAS_DETECTED,
                    severity="medium",
                    description=f"Potential stereotypical language: {match}",
                    confidence=0.7,
                    suggested_action="Consider more inclusive language"
                )
                violations.append(violation)
        
        return violations


class SafetyGuardrails:
    """Main safety guardrails system."""
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.pii_detector = PIIDetector(self.config)
        self.harmful_content_detector = HarmfulContentDetector(self.config)
        self.bias_detector = BiasDetector(self.config)
        
        # Safety metrics
        self.total_checks = 0
        self.violations_detected = 0
        self.violations_by_type: Dict[ThreatType, int] = {}
    
    async def check_safety(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive safety check on text."""
        self.total_checks += 1
        start_time = datetime.utcnow()
        
        all_violations = []
        
        # Run all safety checks concurrently
        tasks = []
        
        if self.config.enable_pii_detection:
            tasks.append(self.pii_detector.detect_pii(text))
        
        if self.config.enable_harmful_content_detection:
            tasks.append(self.harmful_content_detector.detect_harmful_content(text))
        
        if self.config.enable_bias_detection:
            tasks.append(self.bias_detector.detect_bias(text))
        
        # Execute all checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect violations
        for result in results:
            if isinstance(result, list):
                all_violations.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Safety check error: {result}")
        
        # Update metrics
        if all_violations:
            self.violations_detected += 1
            for violation in all_violations:
                self.violations_by_type[violation.threat_type] = (
                    self.violations_by_type.get(violation.threat_type, 0) + 1
                )
        
        # Determine overall safety status
        safety_status = self._determine_safety_status(all_violations)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "is_safe": safety_status["is_safe"],
            "safety_level": safety_status["level"],
            "violations": [
                {
                    "threat_type": v.threat_type.value,
                    "severity": v.severity,
                    "description": v.description,
                    "confidence": v.confidence,
                    "location": v.location,
                    "suggested_action": v.suggested_action
                }
                for v in all_violations
            ],
            "masked_text": self.pii_detector.mask_pii(text) if self.config.enable_pii_detection else text,
            "processing_time_ms": processing_time,
            "context": context or {}
        }
    
    def _determine_safety_status(self, violations: List[SafetyViolation]) -> Dict[str, Any]:
        """Determine overall safety status based on violations."""
        if not violations:
            return {"is_safe": True, "level": "safe"}
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            return {"is_safe": False, "level": "critical"}
        
        # Check for high severity violations
        high_violations = [v for v in violations if v.severity == "high"]
        if high_violations:
            return {"is_safe": False, "level": "high_risk"}
        
        # Check safety level configuration
        if self.config.safety_level == SafetyLevel.STRICT:
            return {"is_safe": False, "level": "requires_review"}
        elif self.config.safety_level == SafetyLevel.MODERATE:
            medium_violations = [v for v in violations if v.severity == "medium"]
            if len(medium_violations) > 2:
                return {"is_safe": False, "level": "requires_review"}
        
        return {"is_safe": True, "level": "acceptable_risk"}
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety system metrics."""
        return {
            "total_checks": self.total_checks,
            "violations_detected": self.violations_detected,
            "violation_rate": self.violations_detected / max(1, self.total_checks),
            "violations_by_type": {
                threat_type.value: count
                for threat_type, count in self.violations_by_type.items()
            },
            "safety_level": self.config.safety_level.value
        }
