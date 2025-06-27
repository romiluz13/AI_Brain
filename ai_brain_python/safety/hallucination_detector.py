"""
Hallucination Detection System for AI Brain.

Detects and mitigates AI hallucinations through fact-checking,
consistency analysis, and confidence assessment.
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HallucinationType(str, Enum):
    """Types of hallucinations."""
    FACTUAL_ERROR = "factual_error"
    INCONSISTENCY = "inconsistency"
    FABRICATED_REFERENCE = "fabricated_reference"
    IMPOSSIBLE_CLAIM = "impossible_claim"
    TEMPORAL_ERROR = "temporal_error"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class ConfidenceLevel(str, Enum):
    """Confidence levels for hallucination detection."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class HallucinationDetection:
    """Represents a detected hallucination."""
    hallucination_type: HallucinationType
    confidence: float
    description: str
    location: str
    evidence: List[str]
    severity: str  # "low", "medium", "high", "critical"
    suggested_correction: Optional[str] = None


class HallucinationConfig(BaseModel):
    """Configuration for hallucination detection."""
    
    enable_fact_checking: bool = Field(default=True)
    enable_consistency_checking: bool = Field(default=True)
    enable_reference_validation: bool = Field(default=True)
    enable_temporal_validation: bool = Field(default=True)
    enable_logical_validation: bool = Field(default=True)
    
    # Confidence thresholds
    min_confidence_threshold: float = Field(default=0.7)
    hallucination_threshold: float = Field(default=0.6)
    
    # Known fact patterns
    fact_patterns: Dict[str, List[str]] = Field(default_factory=lambda: {
        "dates": [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b'
        ],
        "numbers": [
            r'\b\d+(\.\d+)?\s*(million|billion|trillion|thousand)\b',
            r'\b\d+(\.\d+)?%\b'
        ],
        "locations": [
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # City, State
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'    # Country names
        ]
    })
    
    # Impossible claim patterns
    impossible_patterns: List[str] = Field(default_factory=lambda: [
        r'\b(before|after)\s+(birth|death)\b',
        r'\b(negative|minus)\s+\d+\s+(people|population)\b',
        r'\b\d+\s*%\s*(more than|over)\s+100%\b'
    ])
    
    # Reference patterns
    reference_patterns: List[str] = Field(default_factory=lambda: [
        r'\(.*\d{4}.*\)',  # Citations with years
        r'\[.*\]',         # Bracketed references
        r'according to.*',  # Attribution phrases
        r'studies show.*',  # Research claims
        r'research indicates.*'
    ])


class FactChecker:
    """Checks factual accuracy of claims."""
    
    def __init__(self, config: HallucinationConfig):
        self.config = config
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in config.fact_patterns.items()
        }
    
    async def check_facts(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[HallucinationDetection]:
        """Check factual accuracy of text."""
        detections = []
        
        # Check for impossible claims
        detections.extend(await self._check_impossible_claims(text))
        
        # Check temporal consistency
        detections.extend(await self._check_temporal_consistency(text))
        
        # Check numerical claims
        detections.extend(await self._check_numerical_claims(text))
        
        return detections
    
    async def _check_impossible_claims(self, text: str) -> List[HallucinationDetection]:
        """Check for impossible or illogical claims."""
        detections = []
        
        for pattern in self.config.impossible_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detection = HallucinationDetection(
                    hallucination_type=HallucinationType.IMPOSSIBLE_CLAIM,
                    confidence=0.9,
                    description=f"Impossible claim detected: {match.group()}",
                    location=f"Position {match.start()}-{match.end()}",
                    evidence=[f"Pattern: {pattern}"],
                    severity="high",
                    suggested_correction="Review and correct the impossible claim"
                )
                detections.append(detection)
        
        return detections
    
    async def _check_temporal_consistency(self, text: str) -> List[HallucinationDetection]:
        """Check for temporal inconsistencies."""
        detections = []
        
        # Extract years from text
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        years = [int(match.group()) for match in year_pattern.finditer(text)]
        
        current_year = datetime.now().year
        
        for year in years:
            if year > current_year:
                detection = HallucinationDetection(
                    hallucination_type=HallucinationType.TEMPORAL_ERROR,
                    confidence=0.95,
                    description=f"Future year mentioned: {year}",
                    location=f"Year: {year}",
                    evidence=[f"Current year: {current_year}"],
                    severity="medium",
                    suggested_correction=f"Verify if {year} is correct or should be {current_year}"
                )
                detections.append(detection)
        
        return detections
    
    async def _check_numerical_claims(self, text: str) -> List[HallucinationDetection]:
        """Check for suspicious numerical claims."""
        detections = []
        
        # Check for percentages over 100%
        percentage_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*%\b')
        for match in percentage_pattern.finditer(text):
            percentage = float(match.group(1))
            if percentage > 100:
                detection = HallucinationDetection(
                    hallucination_type=HallucinationType.IMPOSSIBLE_CLAIM,
                    confidence=0.8,
                    description=f"Percentage over 100%: {percentage}%",
                    location=f"Position {match.start()}-{match.end()}",
                    evidence=[f"Percentage: {percentage}%"],
                    severity="medium",
                    suggested_correction="Verify the percentage calculation"
                )
                detections.append(detection)
        
        return detections


class ConsistencyChecker:
    """Checks for internal consistency in text."""
    
    def __init__(self, config: HallucinationConfig):
        self.config = config
    
    async def check_consistency(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[HallucinationDetection]:
        """Check for internal consistency."""
        detections = []
        
        # Check for contradictory statements
        detections.extend(await self._check_contradictions(text))
        
        # Check for inconsistent facts
        detections.extend(await self._check_fact_consistency(text))
        
        return detections
    
    async def _check_contradictions(self, text: str) -> List[HallucinationDetection]:
        """Check for logical contradictions."""
        detections = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Look for contradictory patterns
        contradiction_patterns = [
            (r'\b(always|never|all|none)\b', r'\b(sometimes|some|few|many)\b'),
            (r'\b(increase|rise|grow)\b', r'\b(decrease|fall|shrink)\b'),
            (r'\b(before|earlier)\b', r'\b(after|later)\b')
        ]
        
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                for pattern1, pattern2 in contradiction_patterns:
                    if (re.search(pattern1, sentence1, re.IGNORECASE) and 
                        re.search(pattern2, sentence2, re.IGNORECASE)):
                        
                        detection = HallucinationDetection(
                            hallucination_type=HallucinationType.LOGICAL_CONTRADICTION,
                            confidence=0.7,
                            description=f"Potential contradiction between sentences {i+1} and {j+1}",
                            location=f"Sentences {i+1}-{j+1}",
                            evidence=[sentence1.strip(), sentence2.strip()],
                            severity="medium",
                            suggested_correction="Review for logical consistency"
                        )
                        detections.append(detection)
        
        return detections
    
    async def _check_fact_consistency(self, text: str) -> List[HallucinationDetection]:
        """Check for consistent facts throughout text."""
        detections = []
        
        # Extract and compare dates
        date_pattern = re.compile(r'\b(19|20)\d{2}\b')
        dates = [(match.group(), match.start()) for match in date_pattern.finditer(text)]
        
        # Check for inconsistent date references to the same entity
        entities = self._extract_entities(text)
        for entity in entities:
            entity_dates = []
            for date, pos in dates:
                # Simple proximity check (within 50 characters)
                entity_pos = text.lower().find(entity.lower())
                if entity_pos != -1 and abs(pos - entity_pos) < 50:
                    entity_dates.append(date)
            
            if len(set(entity_dates)) > 1:
                detection = HallucinationDetection(
                    hallucination_type=HallucinationType.INCONSISTENCY,
                    confidence=0.6,
                    description=f"Inconsistent dates for {entity}: {entity_dates}",
                    location=f"Entity: {entity}",
                    evidence=entity_dates,
                    severity="medium",
                    suggested_correction="Verify correct date for the entity"
                )
                detections.append(detection)
        
        return detections
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text (simplified)."""
        # Simple capitalized word extraction
        entity_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        entities = [match.group() for match in entity_pattern.finditer(text)]
        
        # Filter out common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
        return [entity for entity in entities if entity not in common_words]


class ReferenceValidator:
    """Validates references and citations."""
    
    def __init__(self, config: HallucinationConfig):
        self.config = config
        self.reference_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in config.reference_patterns
        ]
    
    async def validate_references(self, text: str) -> List[HallucinationDetection]:
        """Validate references and citations in text."""
        detections = []
        
        # Check for fabricated references
        detections.extend(await self._check_fabricated_references(text))
        
        # Check for unsupported claims
        detections.extend(await self._check_unsupported_claims(text))
        
        return detections
    
    async def _check_fabricated_references(self, text: str) -> List[HallucinationDetection]:
        """Check for potentially fabricated references."""
        detections = []
        
        # Look for citation patterns
        for pattern in self.reference_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                reference = match.group()
                
                # Simple heuristics for fabricated references
                suspicion_score = 0
                
                # Check for overly specific but unverifiable details
                if re.search(r'\d{4}[a-z]', reference):  # Year with letter suffix
                    suspicion_score += 0.3
                
                # Check for unusual formatting
                if len(reference) > 100:  # Very long references
                    suspicion_score += 0.2
                
                if suspicion_score > 0.4:
                    detection = HallucinationDetection(
                        hallucination_type=HallucinationType.FABRICATED_REFERENCE,
                        confidence=suspicion_score,
                        description=f"Potentially fabricated reference: {reference[:50]}...",
                        location=f"Position {match.start()}-{match.end()}",
                        evidence=[reference],
                        severity="medium",
                        suggested_correction="Verify the reference exists and is accurate"
                    )
                    detections.append(detection)
        
        return detections
    
    async def _check_unsupported_claims(self, text: str) -> List[HallucinationDetection]:
        """Check for claims that lack proper support."""
        detections = []
        
        # Look for strong claims without references
        strong_claim_patterns = [
            r'\b(studies show|research proves|scientists discovered)\b',
            r'\b(according to|based on|as reported by)\b',
            r'\b(it is proven|evidence shows|data indicates)\b'
        ]
        
        for pattern in strong_claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Check if there's a reference nearby (within 100 characters)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                has_reference = any(
                    ref_pattern.search(context) 
                    for ref_pattern in self.reference_patterns
                )
                
                if not has_reference:
                    detection = HallucinationDetection(
                        hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                        confidence=0.7,
                        description=f"Unsupported claim: {match.group()}",
                        location=f"Position {match.start()}-{match.end()}",
                        evidence=[context],
                        severity="medium",
                        suggested_correction="Add proper citation or reference"
                    )
                    detections.append(detection)
        
        return detections


class HallucinationDetector:
    """Main hallucination detection system."""
    
    def __init__(self, config: Optional[HallucinationConfig] = None):
        self.config = config or HallucinationConfig()
        self.fact_checker = FactChecker(self.config)
        self.consistency_checker = ConsistencyChecker(self.config)
        self.reference_validator = ReferenceValidator(self.config)
        
        # Detection metrics
        self.total_checks = 0
        self.hallucinations_detected = 0
        self.detections_by_type: Dict[HallucinationType, int] = {}
    
    async def detect_hallucinations(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect hallucinations in the given text."""
        self.total_checks += 1
        start_time = datetime.utcnow()
        
        all_detections = []
        
        # Run all detection methods concurrently
        tasks = []
        
        if self.config.enable_fact_checking:
            tasks.append(self.fact_checker.check_facts(text, context))
        
        if self.config.enable_consistency_checking:
            tasks.append(self.consistency_checker.check_consistency(text, context))
        
        if self.config.enable_reference_validation:
            tasks.append(self.reference_validator.validate_references(text))
        
        # Execute all checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect detections
        for result in results:
            if isinstance(result, list):
                all_detections.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Hallucination detection error: {result}")
        
        # Update metrics
        if all_detections:
            self.hallucinations_detected += 1
            for detection in all_detections:
                self.detections_by_type[detection.hallucination_type] = (
                    self.detections_by_type.get(detection.hallucination_type, 0) + 1
                )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(all_detections)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "has_hallucinations": len(all_detections) > 0,
            "overall_confidence": overall_confidence,
            "confidence_level": self._get_confidence_level(overall_confidence),
            "detections": [
                {
                    "type": d.hallucination_type.value,
                    "confidence": d.confidence,
                    "description": d.description,
                    "location": d.location,
                    "evidence": d.evidence,
                    "severity": d.severity,
                    "suggested_correction": d.suggested_correction
                }
                for d in all_detections
            ],
            "processing_time_ms": processing_time,
            "context": context or {}
        }
    
    def _calculate_overall_confidence(self, detections: List[HallucinationDetection]) -> float:
        """Calculate overall confidence in the text's accuracy."""
        if not detections:
            return 1.0
        
        # Weight by severity and confidence
        total_weight = 0
        weighted_confidence = 0
        
        severity_weights = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        
        for detection in detections:
            weight = severity_weights.get(detection.severity, 0.5)
            total_weight += weight
            weighted_confidence += (1 - detection.confidence) * weight
        
        if total_weight == 0:
            return 1.0
        
        return max(0.0, weighted_confidence / total_weight)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get hallucination detection metrics."""
        return {
            "total_checks": self.total_checks,
            "hallucinations_detected": self.hallucinations_detected,
            "detection_rate": self.hallucinations_detected / max(1, self.total_checks),
            "detections_by_type": {
                detection_type.value: count
                for detection_type, count in self.detections_by_type.items()
            },
            "config": {
                "fact_checking_enabled": self.config.enable_fact_checking,
                "consistency_checking_enabled": self.config.enable_consistency_checking,
                "reference_validation_enabled": self.config.enable_reference_validation,
                "confidence_threshold": self.config.min_confidence_threshold
            }
        }
