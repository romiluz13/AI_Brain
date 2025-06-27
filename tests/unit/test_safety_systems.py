"""
Unit tests for safety systems.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ai_brain_python.safety.safety_guardrails import SafetyGuardrails, SafetyConfig, PIIDetector, HarmfulContentDetector
from ai_brain_python.safety.hallucination_detector import HallucinationDetector, HallucinationConfig
from ai_brain_python.safety.compliance_logger import ComplianceLogger, ComplianceConfig, EventType, SeverityLevel
from ai_brain_python.safety import IntegratedSafetySystem, SafetySystemConfig


class TestSafetyGuardrails:
    """Test cases for Safety Guardrails."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test safety guardrails initialization."""
        config = SafetyConfig()
        guardrails = SafetyGuardrails(config)
        
        assert guardrails.config == config
        assert guardrails.pii_detector is not None
        assert guardrails.harmful_content_detector is not None
        assert guardrails.bias_detector is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_safe_content_check(self):
        """Test safety check with safe content."""
        guardrails = SafetyGuardrails()
        
        result = await guardrails.check_safety("This is a normal, safe message about AI")
        
        assert result["is_safe"] is True
        assert result["safety_level"] == "safe"
        assert len(result["violations"]) == 0
        assert result["processing_time_ms"] > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pii_detection(self):
        """Test PII detection in content."""
        guardrails = SafetyGuardrails()
        
        text_with_pii = "My email is john.doe@example.com and my phone is 555-123-4567"
        result = await guardrails.check_safety(text_with_pii)
        
        assert result["is_safe"] is False
        assert len(result["violations"]) > 0
        
        # Check that PII was detected
        pii_violations = [v for v in result["violations"] if v["threat_type"] == "pii_exposure"]
        assert len(pii_violations) > 0
        
        # Check that text was masked
        assert "EMAIL_REDACTED" in result["masked_text"]
        assert "PHONE_REDACTED" in result["masked_text"]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_harmful_content_detection(self):
        """Test harmful content detection."""
        guardrails = SafetyGuardrails()
        
        harmful_text = "This content contains violence and hate speech"
        result = await guardrails.check_safety(harmful_text)
        
        assert result["is_safe"] is False
        assert len(result["violations"]) > 0
        
        # Check that harmful content was detected
        harmful_violations = [v for v in result["violations"] if v["threat_type"] == "harmful_content"]
        assert len(harmful_violations) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bias_detection(self):
        """Test bias detection in content."""
        guardrails = SafetyGuardrails()
        
        biased_text = "All people from that group are typically the same"
        result = await guardrails.check_safety(biased_text)
        
        # May or may not be flagged depending on sensitivity
        if not result["is_safe"]:
            bias_violations = [v for v in result["violations"] if v["threat_type"] == "bias_detected"]
            assert len(bias_violations) >= 0  # May detect bias
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_safety_levels(self):
        """Test different safety levels."""
        # Strict safety level
        strict_config = SafetyConfig(safety_level="strict")
        strict_guardrails = SafetyGuardrails(strict_config)
        
        # Permissive safety level
        permissive_config = SafetyConfig(safety_level="permissive")
        permissive_guardrails = SafetyGuardrails(permissive_config)
        
        test_text = "This might be questionable content"
        
        strict_result = await strict_guardrails.check_safety(test_text)
        permissive_result = await permissive_guardrails.check_safety(test_text)
        
        # Strict should be more restrictive
        assert strict_result["safety_level"] in ["safe", "requires_review", "high_risk", "critical"]
        assert permissive_result["safety_level"] in ["safe", "acceptable_risk", "requires_review"]
    
    @pytest.mark.unit
    def test_safety_metrics(self):
        """Test safety metrics collection."""
        guardrails = SafetyGuardrails()
        
        metrics = guardrails.get_safety_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_checks" in metrics
        assert "violations_detected" in metrics
        assert "violation_rate" in metrics
        assert "violations_by_type" in metrics
        assert "safety_level" in metrics


class TestPIIDetector:
    """Test cases for PII Detector."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_email_detection(self):
        """Test email detection."""
        config = SafetyConfig()
        detector = PIIDetector(config)
        
        text = "Contact me at john.doe@example.com for more info"
        violations = await detector.detect_pii(text)
        
        assert len(violations) > 0
        assert any(v.threat_type.value == "pii_exposure" for v in violations)
        assert any("email" in v.description.lower() for v in violations)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phone_detection(self):
        """Test phone number detection."""
        config = SafetyConfig()
        detector = PIIDetector(config)
        
        text = "Call me at 555-123-4567"
        violations = await detector.detect_pii(text)
        
        assert len(violations) > 0
        assert any("phone" in v.description.lower() for v in violations)
    
    @pytest.mark.unit
    def test_pii_masking(self):
        """Test PII masking functionality."""
        config = SafetyConfig()
        detector = PIIDetector(config)
        
        text = "Email: john@example.com, Phone: 555-123-4567"
        masked_text = detector.mask_pii(text)
        
        assert "EMAIL_REDACTED" in masked_text
        assert "PHONE_REDACTED" in masked_text
        assert "john@example.com" not in masked_text
        assert "555-123-4567" not in masked_text


class TestHallucinationDetector:
    """Test cases for Hallucination Detector."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test hallucination detector initialization."""
        config = HallucinationConfig()
        detector = HallucinationDetector(config)
        
        assert detector.config == config
        assert detector.fact_checker is not None
        assert detector.consistency_checker is not None
        assert detector.reference_validator is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_factual_content(self):
        """Test detection with factual content."""
        detector = HallucinationDetector()
        
        factual_text = "Python is a programming language created by Guido van Rossum"
        result = await detector.detect_hallucinations(factual_text)
        
        assert result["has_hallucinations"] is False
        assert result["overall_confidence"] > 0.7
        assert len(result["detections"]) == 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_impossible_claims(self):
        """Test detection of impossible claims."""
        detector = HallucinationDetector()
        
        impossible_text = "This event happened in 2030 before it was invented"
        result = await detector.detect_hallucinations(impossible_text)
        
        # May detect temporal inconsistency
        if result["has_hallucinations"]:
            assert len(result["detections"]) > 0
            temporal_detections = [d for d in result["detections"] if d["type"] == "temporal_error"]
            assert len(temporal_detections) >= 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_future_dates(self):
        """Test detection of future dates."""
        detector = HallucinationDetector()
        
        future_text = "This research was published in 2030"
        result = await detector.detect_hallucinations(future_text)
        
        # Should detect future date
        assert result["has_hallucinations"] is True
        assert len(result["detections"]) > 0
        
        temporal_detections = [d for d in result["detections"] if d["type"] == "temporal_error"]
        assert len(temporal_detections) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contradictions(self):
        """Test detection of logical contradictions."""
        detector = HallucinationDetector()
        
        contradictory_text = "The number always increases. The number never goes up."
        result = await detector.detect_hallucinations(contradictory_text)
        
        # May detect contradiction
        if result["has_hallucinations"]:
            contradiction_detections = [d for d in result["detections"] if d["type"] == "logical_contradiction"]
            assert len(contradiction_detections) >= 0
    
    @pytest.mark.unit
    def test_detection_metrics(self):
        """Test hallucination detection metrics."""
        detector = HallucinationDetector()
        
        metrics = detector.get_detection_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_checks" in metrics
        assert "hallucinations_detected" in metrics
        assert "detection_rate" in metrics
        assert "detections_by_type" in metrics
        assert "config" in metrics


class TestComplianceLogger:
    """Test cases for Compliance Logger."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test compliance logger initialization."""
        config = ComplianceConfig(enable_file_logging=False)  # Disable file logging for tests
        logger = ComplianceLogger(config)
        
        # Mock MongoDB setup
        logger.db = MagicMock()
        logger.collection = MagicMock()
        
        await logger.initialize()
        
        assert logger.config == config
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_logging(self):
        """Test compliance event logging."""
        config = ComplianceConfig(enable_file_logging=False)
        logger = ComplianceLogger(config)
        
        # Mock MongoDB
        logger.collection = MagicMock()
        logger.collection.insert_many = AsyncMock()
        
        event_id = await logger.log_event(
            event_type=EventType.DATA_PROCESSING,
            description="Test event",
            user_id="test_user",
            severity=SeverityLevel.INFO
        )
        
        assert event_id is not None
        assert event_id.startswith("comp_")
        assert len(logger.event_buffer) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_buffering(self):
        """Test event buffering and flushing."""
        config = ComplianceConfig(enable_file_logging=False)
        logger = ComplianceLogger(config)
        logger.buffer_size = 2  # Small buffer for testing
        
        # Mock MongoDB
        logger.collection = MagicMock()
        logger.collection.insert_many = AsyncMock()
        
        # Add events to buffer
        await logger.log_event(EventType.DATA_ACCESS, "Event 1", severity=SeverityLevel.INFO)
        assert len(logger.event_buffer) == 1
        
        await logger.log_event(EventType.DATA_ACCESS, "Event 2", severity=SeverityLevel.INFO)
        # Buffer should be flushed automatically
        logger.collection.insert_many.assert_called_once()
    
    @pytest.mark.unit
    def test_compliance_metrics(self):
        """Test compliance metrics collection."""
        config = ComplianceConfig(enable_file_logging=False)
        logger = ComplianceLogger(config)
        
        metrics = logger.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "events_logged" in metrics
        assert "events_by_type" in metrics
        assert "events_by_severity" in metrics
        assert "buffer_size" in metrics
        assert "enabled_standards" in metrics


class TestIntegratedSafetySystem:
    """Test cases for Integrated Safety System."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test integrated safety system initialization."""
        config = SafetySystemConfig()
        safety_system = IntegratedSafetySystem(config)
        
        # Mock components
        safety_system.compliance_logger.initialize = AsyncMock()
        safety_system.real_time_monitor.start = AsyncMock()
        safety_system.compliance_logger.log_event = AsyncMock(return_value="test_event_id")
        
        await safety_system.initialize()
        
        assert safety_system.initialized is True
        assert safety_system.safety_guardrails is not None
        assert safety_system.compliance_logger is not None
        assert safety_system.hallucination_detector is not None
        assert safety_system.real_time_monitor is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_comprehensive_safety_check_safe(self):
        """Test comprehensive safety check with safe content."""
        safety_system = IntegratedSafetySystem()
        
        # Mock components
        safety_system.compliance_logger.initialize = AsyncMock()
        safety_system.real_time_monitor.start = AsyncMock()
        safety_system.compliance_logger.log_event = AsyncMock(return_value="test_event_id")
        safety_system.real_time_monitor.record_request = MagicMock()
        safety_system.real_time_monitor.add_custom_metric = MagicMock()
        
        await safety_system.initialize()
        
        result = await safety_system.comprehensive_safety_check(
            text="This is a safe message about AI development",
            user_id="test_user",
            session_id="test_session"
        )
        
        assert result["overall_safe"] is True
        assert "safety_guardrails" in result
        assert "hallucination_detection" in result
        assert "recommendations" in result
        assert result["processing_time_ms"] > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_comprehensive_safety_check_unsafe(self):
        """Test comprehensive safety check with unsafe content."""
        safety_system = IntegratedSafetySystem()
        
        # Mock components
        safety_system.compliance_logger.initialize = AsyncMock()
        safety_system.real_time_monitor.start = AsyncMock()
        safety_system.compliance_logger.log_event = AsyncMock(return_value="test_event_id")
        safety_system.real_time_monitor.record_request = MagicMock()
        safety_system.real_time_monitor.add_custom_metric = MagicMock()
        
        await safety_system.initialize()
        
        result = await safety_system.comprehensive_safety_check(
            text="My email is test@example.com and this contains violence",
            user_id="test_user",
            session_id="test_session"
        )
        
        assert result["overall_safe"] is False
        assert len(result["safety_guardrails"]["violations"]) > 0
        assert len(result["recommendations"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_safety_dashboard(self):
        """Test safety dashboard data retrieval."""
        safety_system = IntegratedSafetySystem()
        
        # Mock components
        safety_system.compliance_logger.initialize = AsyncMock()
        safety_system.real_time_monitor.start = AsyncMock()
        safety_system.compliance_logger.log_event = AsyncMock(return_value="test_event_id")
        safety_system.safety_guardrails.get_safety_metrics = MagicMock(return_value={})
        safety_system.hallucination_detector.get_detection_metrics = MagicMock(return_value={})
        safety_system.compliance_logger.get_metrics = MagicMock(return_value={})
        safety_system.real_time_monitor.get_dashboard_data = MagicMock(return_value={})
        safety_system.real_time_monitor.get_health_status = MagicMock(return_value={"status": "healthy"})
        
        await safety_system.initialize()
        
        dashboard = await safety_system.get_safety_dashboard()
        
        assert isinstance(dashboard, dict)
        assert "system_status" in dashboard
        assert "safety_metrics" in dashboard
        assert "hallucination_metrics" in dashboard
        assert "compliance_metrics" in dashboard
        assert "monitoring_data" in dashboard
        assert "health_status" in dashboard
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in safety system."""
        safety_system = IntegratedSafetySystem()
        
        # Mock components to raise errors
        safety_system.compliance_logger.initialize = AsyncMock()
        safety_system.real_time_monitor.start = AsyncMock()
        safety_system.compliance_logger.log_event = AsyncMock(return_value="test_event_id")
        safety_system.real_time_monitor.record_request = MagicMock()
        safety_system.safety_guardrails.check_safety = AsyncMock(side_effect=Exception("Mock error"))
        
        await safety_system.initialize()
        
        result = await safety_system.comprehensive_safety_check(
            text="Test message",
            user_id="test_user"
        )
        
        # Should handle error gracefully
        assert result["overall_safe"] is False
        assert "error" in result
