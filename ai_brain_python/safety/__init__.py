"""
Safety and Monitoring Systems for AI Brain.

Comprehensive safety framework including guardrails, compliance logging,
hallucination detection, and real-time monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .safety_guardrails import SafetyGuardrails, SafetyConfig, SafetyViolation
from .compliance_logger import ComplianceLogger, ComplianceConfig, EventType, SeverityLevel
from .hallucination_detector import HallucinationDetector, HallucinationConfig
from .real_time_monitor import RealTimeMonitor, MonitoringConfig, Metric, MetricType

logger = logging.getLogger(__name__)


class SafetySystemConfig:
    """Configuration for the integrated safety system."""

    def __init__(
        self,
        safety_config: Optional[SafetyConfig] = None,
        compliance_config: Optional[ComplianceConfig] = None,
        hallucination_config: Optional[HallucinationConfig] = None,
        monitoring_config: Optional[MonitoringConfig] = None
    ):
        self.safety_config = safety_config or SafetyConfig()
        self.compliance_config = compliance_config or ComplianceConfig()
        self.hallucination_config = hallucination_config or HallucinationConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()


class IntegratedSafetySystem:
    """
    Integrated Safety System for AI Brain.

    Orchestrates all safety components including guardrails, compliance,
    hallucination detection, and real-time monitoring.
    """

    def __init__(self, config: Optional[SafetySystemConfig] = None):
        self.config = config or SafetySystemConfig()

        # Initialize safety components
        self.safety_guardrails = SafetyGuardrails(self.config.safety_config)
        self.compliance_logger = ComplianceLogger(self.config.compliance_config)
        self.hallucination_detector = HallucinationDetector(self.config.hallucination_config)
        self.real_time_monitor = RealTimeMonitor(self.config.monitoring_config)

        # System state
        self.initialized = False
        self.total_checks = 0
        self.safety_violations = 0
        self.compliance_events = 0
        self.hallucinations_detected = 0

    async def initialize(self) -> None:
        """Initialize the integrated safety system."""
        if self.initialized:
            return

        try:
            # Initialize compliance logger
            await self.compliance_logger.initialize()

            # Start real-time monitoring
            await self.real_time_monitor.start()

            # Log system initialization
            await self.compliance_logger.log_event(
                event_type=EventType.SYSTEM_ACCESS,
                description="Safety system initialized",
                severity=SeverityLevel.INFO,
                metadata={"component": "integrated_safety_system"}
            )

            self.initialized = True
            logger.info("Integrated safety system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize safety system: {e}")
            raise

    async def comprehensive_safety_check(
        self,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive safety check on input text.

        Runs all safety checks concurrently and returns aggregated results.
        """
        if not self.initialized:
            await self.initialize()

        start_time = datetime.utcnow()
        self.total_checks += 1

        # Record request start
        self.real_time_monitor.record_request(0, True)  # Will update with actual time

        try:
            # Run all safety checks concurrently
            safety_task = self.safety_guardrails.check_safety(text, context)
            hallucination_task = self.hallucination_detector.detect_hallucinations(text, context)

            safety_result, hallucination_result = await asyncio.gather(
                safety_task, hallucination_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(safety_result, Exception):
                logger.error(f"Safety guardrails error: {safety_result}")
                safety_result = {"is_safe": False, "error": str(safety_result)}

            if isinstance(hallucination_result, Exception):
                logger.error(f"Hallucination detection error: {hallucination_result}")
                hallucination_result = {"has_hallucinations": True, "error": str(hallucination_result)}

            # Aggregate results
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Determine overall safety status
            overall_safe = (
                safety_result.get("is_safe", False) and
                not hallucination_result.get("has_hallucinations", True)
            )

            # Update metrics
            if not overall_safe:
                self.safety_violations += 1

            if safety_result.get("violations"):
                self.safety_violations += len(safety_result["violations"])

            if hallucination_result.get("has_hallucinations"):
                self.hallucinations_detected += 1

            # Log compliance event
            await self._log_safety_check_event(
                text, user_id, session_id, overall_safe, safety_result, hallucination_result
            )

            # Update monitoring metrics
            self.real_time_monitor.record_request(processing_time, overall_safe)
            await self._update_safety_metrics(safety_result, hallucination_result)

            # Prepare response
            response = {
                "overall_safe": overall_safe,
                "processing_time_ms": processing_time,
                "timestamp": start_time.isoformat(),
                "safety_guardrails": safety_result,
                "hallucination_detection": hallucination_result,
                "recommendations": self._generate_recommendations(safety_result, hallucination_result),
                "context": context or {}
            }

            return response

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.real_time_monitor.record_request(processing_time, False, "safety_check_error")

            logger.error(f"Comprehensive safety check failed: {e}")

            # Log error event
            await self.compliance_logger.log_event(
                event_type=EventType.SECURITY_INCIDENT,
                description=f"Safety check error: {str(e)}",
                user_id=user_id,
                session_id=session_id,
                severity=SeverityLevel.ERROR,
                metadata={"error": str(e), "text_length": len(text)}
            )

            return {
                "overall_safe": False,
                "processing_time_ms": processing_time,
                "timestamp": start_time.isoformat(),
                "error": str(e),
                "context": context or {}
            }

    async def _log_safety_check_event(
        self,
        text: str,
        user_id: Optional[str],
        session_id: Optional[str],
        overall_safe: bool,
        safety_result: Dict[str, Any],
        hallucination_result: Dict[str, Any]
    ) -> None:
        """Log safety check event for compliance."""
        self.compliance_events += 1

        # Determine severity
        if not overall_safe:
            if (safety_result.get("violations") and
                any(v.get("severity") == "critical" for v in safety_result["violations"])):
                severity = SeverityLevel.CRITICAL
            elif hallucination_result.get("has_hallucinations"):
                severity = SeverityLevel.WARNING
            else:
                severity = SeverityLevel.WARNING
        else:
            severity = SeverityLevel.INFO

        # Log event
        await self.compliance_logger.log_event(
            event_type=EventType.DATA_PROCESSING,
            description=f"Safety check completed - Safe: {overall_safe}",
            user_id=user_id,
            session_id=session_id,
            severity=severity,
            metadata={
                "text_length": len(text),
                "safety_violations": len(safety_result.get("violations", [])),
                "hallucinations_detected": hallucination_result.get("has_hallucinations", False),
                "overall_safe": overall_safe
            },
            data_categories=["user_input", "safety_analysis"]
        )

    async def _update_safety_metrics(
        self,
        safety_result: Dict[str, Any],
        hallucination_result: Dict[str, Any]
    ) -> None:
        """Update safety metrics for monitoring."""
        timestamp = datetime.utcnow()

        # Safety violations metric
        violation_count = len(safety_result.get("violations", []))
        self.real_time_monitor.add_custom_metric(Metric(
            name="safety_violations_count",
            value=violation_count,
            timestamp=timestamp,
            metric_type=MetricType.SAFETY,
            unit="count"
        ))

        # Hallucination detection metric
        has_hallucinations = hallucination_result.get("has_hallucinations", False)
        self.real_time_monitor.add_custom_metric(Metric(
            name="hallucination_detected",
            value=1 if has_hallucinations else 0,
            timestamp=timestamp,
            metric_type=MetricType.SAFETY,
            unit="boolean"
        ))

        # Overall safety rate
        safety_rate = (self.total_checks - self.safety_violations) / max(1, self.total_checks) * 100
        self.real_time_monitor.add_custom_metric(Metric(
            name="safety_rate_percent",
            value=safety_rate,
            timestamp=timestamp,
            metric_type=MetricType.SAFETY,
            unit="percent"
        ))

    def _generate_recommendations(
        self,
        safety_result: Dict[str, Any],
        hallucination_result: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on safety check results."""
        recommendations = []

        # Safety guardrails recommendations
        if safety_result.get("violations"):
            for violation in safety_result["violations"]:
                if violation.get("suggested_action"):
                    recommendations.append(violation["suggested_action"])

        # Hallucination detection recommendations
        if hallucination_result.get("has_hallucinations"):
            for detection in hallucination_result.get("detections", []):
                if detection.get("suggested_correction"):
                    recommendations.append(detection["suggested_correction"])

        # General recommendations
        if not safety_result.get("is_safe"):
            recommendations.append("Review content for safety compliance before use")

        if hallucination_result.get("has_hallucinations"):
            recommendations.append("Verify factual accuracy and add proper citations")

        return list(set(recommendations))  # Remove duplicates

    async def get_safety_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive safety dashboard data."""
        return {
            "system_status": {
                "initialized": self.initialized,
                "total_checks": self.total_checks,
                "safety_violations": self.safety_violations,
                "compliance_events": self.compliance_events,
                "hallucinations_detected": self.hallucinations_detected
            },
            "safety_metrics": self.safety_guardrails.get_safety_metrics(),
            "hallucination_metrics": self.hallucination_detector.get_detection_metrics(),
            "compliance_metrics": self.compliance_logger.get_metrics(),
            "monitoring_data": self.real_time_monitor.get_dashboard_data(),
            "health_status": self.real_time_monitor.get_health_status()
        }

    async def shutdown(self) -> None:
        """Shutdown the safety system gracefully."""
        try:
            # Stop monitoring
            await self.real_time_monitor.stop()

            # Log shutdown event
            await self.compliance_logger.log_event(
                event_type=EventType.SYSTEM_ACCESS,
                description="Safety system shutdown",
                severity=SeverityLevel.INFO,
                metadata={"component": "integrated_safety_system"}
            )

            logger.info("Safety system shutdown completed")

        except Exception as e:
            logger.error(f"Error during safety system shutdown: {e}")


# Export main classes
__all__ = [
    "IntegratedSafetySystem",
    "SafetySystemConfig",
    "SafetyGuardrails",
    "SafetyConfig",
    "ComplianceLogger",
    "ComplianceConfig",
    "HallucinationDetector",
    "HallucinationConfig",
    "RealTimeMonitor",
    "MonitoringConfig"
]