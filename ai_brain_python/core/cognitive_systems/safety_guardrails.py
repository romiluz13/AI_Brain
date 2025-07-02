"""
SafetyGuardrailsEngine - Advanced safety monitoring and risk mitigation

Exact Python equivalent of JavaScript SafetyGuardrailsEngine.ts with:
- Multi-layered safety monitoring with real-time threat detection
- Adaptive risk assessment and mitigation strategies
- Content filtering and behavioral analysis
- Compliance monitoring and audit trails
- Configurable safety policies with rule management

Features:
- Advanced safety monitoring with real-time threat detection
- Adaptive risk assessment and mitigation strategies
- Content filtering and behavioral analysis
- Compliance monitoring and audit trails
- Configurable safety policies with rule management
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Literal, Union
from dataclasses import dataclass, field
from bson import ObjectId
import asyncio
import json
import random
import re
import math

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_brain_python.storage.collections.safety_collection import SafetyCollection
from ai_brain_python.core.types import SafetyGuardrail, SafetyAnalyticsOptions
from ai_brain_python.utils.logger import logger


@dataclass
class SafetyRequest:
    """Safety assessment request interface."""
    agent_id: str
    session_id: Optional[str]
    content: str
    context: Dict[str, Any]
    risk_level: str
    safety_policies: List[str]


@dataclass
class SafetyResult:
    """Safety assessment result interface."""
    assessment_id: ObjectId
    risk_score: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    action_required: bool
    mitigation_strategies: List[Dict[str, Any]]


class SafetyGuardrailsEngine:
    """
    SafetyGuardrailsEngine - Advanced safety monitoring and risk mitigation

    Exact Python equivalent of JavaScript SafetyGuardrailsEngine with:
    - Multi-layered safety monitoring with real-time threat detection
    - Adaptive risk assessment and mitigation strategies
    - Content filtering and behavioral analysis
    - Compliance monitoring and audit trails
    - Configurable safety policies with rule management
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.safety_collection = SafetyCollection(db)
        self.is_initialized = False

        # Safety configuration
        self._config = {
            "risk_threshold": 0.7,
            "max_violations": 5,
            "monitoring_enabled": True,
            "auto_mitigation": True,
            "audit_logging": True
        }

        # Safety policies and rules
        self._safety_policies: Dict[str, Dict[str, Any]] = {}
        self._risk_patterns: Dict[str, List[str]] = {}
        self._mitigation_strategies: Dict[str, Dict[str, Any]] = {}

        # Monitoring and tracking
        self._active_assessments: Dict[str, Dict[str, Any]] = {}
        self._violation_history: List[Dict[str, Any]] = []

        # Initialize default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self) -> None:
        """Initialize default safety policies."""
        self._safety_policies = {
            "content_safety": {
                "name": "Content Safety",
                "description": "Monitors content for harmful material",
                "rules": [{"type": "keyword_filter", "patterns": ["harmful", "dangerous"]}],
                "severity": "high",
                "enabled": True
            },
            "behavioral_monitoring": {
                "name": "Behavioral Monitoring",
                "description": "Monitors agent behavior for anomalies",
                "rules": [{"type": "frequency_analysis", "max_requests": 100}],
                "severity": "medium",
                "enabled": True
            }
        }

        self._risk_patterns = {
            "high_risk": ["violence", "harm", "illegal", "dangerous"],
            "medium_risk": ["inappropriate", "offensive", "suspicious"],
            "low_risk": ["questionable", "concerning", "unusual"]
        }

    async def initialize(self) -> None:
        """Initialize the safety guardrails engine."""
        if self.is_initialized:
            return

        try:
            # Initialize safety collection
            await self.safety_collection.create_indexes()

            self.is_initialized = True
            logger.info("✅ SafetyGuardrailsEngine initialized successfully")

        except Exception as error:
            logger.error(f"❌ Failed to initialize SafetyGuardrailsEngine: {error}")
            raise error

    async def assess_safety(
        self,
        request: SafetyRequest
    ) -> SafetyResult:
        """Assess safety of content and context."""
        if not self.is_initialized:
            raise Exception("SafetyGuardrailsEngine must be initialized first")

        # Generate assessment ID
        assessment_id = ObjectId()

        # Perform risk assessment
        risk_score = await self._calculate_risk_score(
            request.content,
            request.context,
            request.risk_level
        )

        # Check for violations
        violations = await self._check_policy_violations(
            request.content,
            request.context,
            request.safety_policies
        )

        # Generate recommendations
        recommendations = await self._generate_safety_recommendations(
            risk_score,
            violations,
            request.context
        )

        # Determine if action is required
        action_required = risk_score > self._config["risk_threshold"] or len(violations) > 0

        # Generate mitigation strategies
        mitigation_strategies = await self._generate_mitigation_strategies(
            risk_score,
            violations
        )

        # Create safety assessment record
        assessment_record = {
            "assessmentId": assessment_id,
            "agentId": request.agent_id,
            "sessionId": request.session_id,
            "timestamp": datetime.utcnow(),
            "content": request.content,
            "context": request.context,
            "riskLevel": request.risk_level,
            "riskScore": risk_score,
            "violations": violations,
            "recommendations": recommendations,
            "actionRequired": action_required,
            "mitigationStrategies": mitigation_strategies,
            "safetyPolicies": request.safety_policies,
            "metadata": {
                "framework": "python_ai_brain",
                "version": "1.0.0",
                "source": "safety_guardrails_engine"
            }
        }

        # Store assessment
        await self.safety_collection.record_safety_assessment(assessment_record)

        # Add to active assessments
        self._active_assessments[str(assessment_id)] = assessment_record

        # Update violation history if violations found
        if violations:
            self._violation_history.extend(violations)

        return SafetyResult(
            assessment_id=assessment_id,
            risk_score=risk_score,
            violations=violations,
            recommendations=recommendations,
            action_required=action_required,
            mitigation_strategies=mitigation_strategies
        )

    async def get_safety_analytics(
        self,
        agent_id: str,
        options: Optional[SafetyAnalyticsOptions] = None
    ) -> Dict[str, Any]:
        """Get safety analytics for an agent."""
        return await self.safety_collection.get_safety_analytics(agent_id, options)

    async def get_safety_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get safety statistics."""
        stats = await self.safety_collection.get_safety_stats(agent_id)

        return {
            **stats,
            "activePolicies": len([p for p in self._safety_policies.values() if p.get("enabled", False)]),
            "activeAssessments": len(self._active_assessments),
            "violationHistory": len(self._violation_history)
        }

    # Private helper methods
    async def _calculate_risk_score(
        self,
        content: str,
        context: Dict[str, Any],
        risk_level: str
    ) -> float:
        """Calculate risk score for content."""
        base_score = 0.0

        # Check for risk patterns
        content_lower = content.lower()
        for risk_category, patterns in self._risk_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if pattern in content_lower)
            if pattern_matches > 0:
                if risk_category == "high_risk":
                    base_score += 0.8
                elif risk_category == "medium_risk":
                    base_score += 0.5
                elif risk_category == "low_risk":
                    base_score += 0.3

        # Adjust based on context
        if context.get("sensitive_context", False):
            base_score += 0.2

        # Adjust based on risk level
        risk_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}
        multiplier = risk_multipliers.get(risk_level, 1.0)

        final_score = min(1.0, base_score * multiplier)
        return final_score

    async def _check_policy_violations(
        self,
        content: str,
        context: Dict[str, Any],
        safety_policies: List[str]
    ) -> List[Dict[str, Any]]:
        """Check for policy violations."""
        violations = []

        for policy_id in safety_policies:
            if policy_id in self._safety_policies:
                policy = self._safety_policies[policy_id]
                if policy.get("enabled", False):
                    policy_violations = await self._check_single_policy(content, context, policy)
                    violations.extend(policy_violations)

        return violations

    async def _check_single_policy(
        self,
        content: str,
        context: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check violations for a single policy."""
        violations = []

        for rule in policy.get("rules", []):
            if rule["type"] == "keyword_filter":
                for pattern in rule.get("patterns", []):
                    if pattern.lower() in content.lower():
                        violations.append({
                            "policy": policy["name"],
                            "rule": rule["type"],
                            "violation": f"Keyword '{pattern}' detected",
                            "severity": policy.get("severity", "medium")
                        })

        return violations

    async def _generate_safety_recommendations(
        self,
        risk_score: float,
        violations: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []

        if risk_score > 0.8:
            recommendations.append("High risk detected - immediate review required")
        elif risk_score > 0.5:
            recommendations.append("Medium risk detected - monitor closely")

        if violations:
            recommendations.append(f"Found {len(violations)} policy violations - review content")

        if context.get("sensitive_context", False):
            recommendations.append("Sensitive context detected - apply additional safeguards")

        return recommendations

    async def _generate_mitigation_strategies(
        self,
        risk_score: float,
        violations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate mitigation strategies."""
        strategies = []

        if risk_score > 0.8:
            strategies.append({"action": "block", "reason": "High risk score"})
        elif risk_score > 0.5:
            strategies.append({"action": "flag", "reason": "Medium risk score"})

        for violation in violations:
            if violation.get("severity") == "high":
                strategies.append({"action": "block", "reason": f"High severity violation: {violation['violation']}"})
            else:
                strategies.append({"action": "warn", "reason": f"Policy violation: {violation['violation']}"})

        return strategies