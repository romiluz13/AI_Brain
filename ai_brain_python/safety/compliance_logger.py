"""
Compliance Logging System for AI Brain.

Provides comprehensive audit trails, compliance monitoring,
and regulatory reporting capabilities.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"


class EventType(str, Enum):
    """Types of compliance events."""
    DATA_ACCESS = "data_access"
    DATA_PROCESSING = "data_processing"
    DATA_STORAGE = "data_storage"
    DATA_DELETION = "data_deletion"
    USER_CONSENT = "user_consent"
    PRIVACY_REQUEST = "privacy_request"
    SECURITY_INCIDENT = "security_incident"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    AUDIT_EVENT = "audit_event"


class SeverityLevel(str, Enum):
    """Severity levels for compliance events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ComplianceEvent:
    """Represents a compliance event."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    description: str
    severity: SeverityLevel
    compliance_standards: List[ComplianceStandard]
    metadata: Dict[str, Any]
    data_categories: List[str]
    retention_period: Optional[int] = None  # days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "description": self.description,
            "severity": self.severity.value,
            "compliance_standards": [std.value for std in self.compliance_standards],
            "metadata": self.metadata,
            "data_categories": self.data_categories,
            "retention_period": self.retention_period
        }


class ComplianceConfig(BaseModel):
    """Configuration for compliance logging."""
    
    enabled_standards: List[ComplianceStandard] = Field(
        default=[ComplianceStandard.GDPR, ComplianceStandard.CCPA]
    )
    
    # Storage settings
    mongodb_uri: Optional[str] = Field(default=None)
    database_name: str = Field(default="ai_brain_compliance")
    collection_name: str = Field(default="compliance_events")
    
    # File logging settings
    enable_file_logging: bool = Field(default=True)
    log_directory: str = Field(default="./logs/compliance")
    log_rotation_days: int = Field(default=30)
    
    # Retention policies
    default_retention_days: int = Field(default=2555)  # 7 years for GDPR
    retention_by_standard: Dict[str, int] = Field(default_factory=lambda: {
        "gdpr": 2555,  # 7 years
        "ccpa": 1095,  # 3 years
        "hipaa": 2190, # 6 years
        "sox": 2555,   # 7 years
        "pci_dss": 365 # 1 year
    })
    
    # Privacy settings
    anonymize_user_data: bool = Field(default=True)
    encrypt_sensitive_data: bool = Field(default=True)
    
    # Alerting
    enable_real_time_alerts: bool = Field(default=True)
    alert_on_severity: List[SeverityLevel] = Field(
        default=[SeverityLevel.ERROR, SeverityLevel.CRITICAL]
    )


class ComplianceLogger:
    """Main compliance logging system."""
    
    def __init__(self, config: Optional[ComplianceConfig] = None):
        self.config = config or ComplianceConfig()
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.collection = None
        
        # Initialize file logging
        if self.config.enable_file_logging:
            self._setup_file_logging()
        
        # Event buffer for batch processing
        self.event_buffer: List[ComplianceEvent] = []
        self.buffer_size = 100
        self.last_flush = datetime.utcnow()
        
        # Metrics
        self.events_logged = 0
        self.events_by_type: Dict[EventType, int] = {}
        self.events_by_severity: Dict[SeverityLevel, int] = {}
    
    async def initialize(self) -> None:
        """Initialize the compliance logger."""
        if self.config.mongodb_uri:
            await self._setup_mongodb()
        
        # Start background tasks
        asyncio.create_task(self._periodic_flush())
        asyncio.create_task(self._cleanup_old_events())
        
        logger.info("Compliance logger initialized")
    
    async def _setup_mongodb(self) -> None:
        """Setup MongoDB connection for compliance logging."""
        try:
            client = AsyncIOMotorClient(self.config.mongodb_uri)
            self.db = client[self.config.database_name]
            self.collection = self.db[self.config.collection_name]
            
            # Create indexes for efficient querying
            await self.collection.create_index([
                ("timestamp", -1),
                ("event_type", 1),
                ("severity", 1)
            ])
            
            await self.collection.create_index([
                ("user_id", 1),
                ("timestamp", -1)
            ])
            
            await self.collection.create_index([
                ("compliance_standards", 1),
                ("timestamp", -1)
            ])
            
            logger.info("MongoDB compliance logging setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup MongoDB compliance logging: {e}")
    
    def _setup_file_logging(self) -> None:
        """Setup file-based compliance logging."""
        log_dir = Path(self.config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup rotating file handler
        from logging.handlers import RotatingFileHandler
        
        compliance_logger = logging.getLogger("compliance")
        compliance_logger.setLevel(logging.INFO)
        
        handler = RotatingFileHandler(
            log_dir / "compliance.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=self.config.log_rotation_days
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        compliance_logger.addHandler(handler)
        
        self.file_logger = compliance_logger
    
    async def log_event(
        self,
        event_type: EventType,
        description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        data_categories: Optional[List[str]] = None,
        compliance_standards: Optional[List[ComplianceStandard]] = None
    ) -> str:
        """Log a compliance event."""
        
        # Generate unique event ID
        event_id = f"comp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
        
        # Determine applicable compliance standards
        if compliance_standards is None:
            compliance_standards = self.config.enabled_standards
        
        # Create compliance event
        event = ComplianceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=self._anonymize_user_id(user_id) if self.config.anonymize_user_data else user_id,
            session_id=session_id,
            description=description,
            severity=severity,
            compliance_standards=compliance_standards,
            metadata=metadata or {},
            data_categories=data_categories or [],
            retention_period=self._get_retention_period(compliance_standards)
        )
        
        # Add to buffer
        self.event_buffer.append(event)
        
        # Update metrics
        self.events_logged += 1
        self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1
        self.events_by_severity[severity] = self.events_by_severity.get(severity, 0) + 1
        
        # File logging
        if self.config.enable_file_logging:
            self.file_logger.info(json.dumps(event.to_dict()))
        
        # Real-time alerts
        if (self.config.enable_real_time_alerts and 
            severity in self.config.alert_on_severity):
            await self._send_alert(event)
        
        # Flush if buffer is full
        if len(self.event_buffer) >= self.buffer_size:
            await self._flush_events()
        
        return event_id
    
    def _anonymize_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Anonymize user ID for privacy compliance."""
        if not user_id:
            return None
        
        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def _get_retention_period(self, standards: List[ComplianceStandard]) -> int:
        """Get retention period based on compliance standards."""
        max_retention = self.config.default_retention_days
        
        for standard in standards:
            standard_retention = self.config.retention_by_standard.get(
                standard.value, self.config.default_retention_days
            )
            max_retention = max(max_retention, standard_retention)
        
        return max_retention
    
    async def _flush_events(self) -> None:
        """Flush events from buffer to storage."""
        if not self.event_buffer:
            return
        
        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()
        
        # Store in MongoDB
        if self.collection:
            try:
                documents = [event.to_dict() for event in events_to_flush]
                await self.collection.insert_many(documents)
                logger.debug(f"Flushed {len(events_to_flush)} compliance events to MongoDB")
            except Exception as e:
                logger.error(f"Failed to flush events to MongoDB: {e}")
                # Re-add events to buffer for retry
                self.event_buffer.extend(events_to_flush)
        
        self.last_flush = datetime.utcnow()
    
    async def _periodic_flush(self) -> None:
        """Periodically flush events from buffer."""
        while True:
            await asyncio.sleep(60)  # Flush every minute
            
            # Flush if buffer has events or it's been too long
            if (self.event_buffer or 
                (datetime.utcnow() - self.last_flush).seconds > 300):
                await self._flush_events()
    
    async def _cleanup_old_events(self) -> None:
        """Clean up old events based on retention policies."""
        while True:
            await asyncio.sleep(24 * 60 * 60)  # Run daily
            
            if not self.collection:
                continue
            
            try:
                # Calculate cutoff dates for each standard
                for standard, retention_days in self.config.retention_by_standard.items():
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    
                    result = await self.collection.delete_many({
                        "compliance_standards": standard,
                        "timestamp": {"$lt": cutoff_date.isoformat()}
                    })
                    
                    if result.deleted_count > 0:
                        logger.info(f"Cleaned up {result.deleted_count} old {standard} events")
                        
            except Exception as e:
                logger.error(f"Failed to cleanup old events: {e}")
    
    async def _send_alert(self, event: ComplianceEvent) -> None:
        """Send real-time alert for critical events."""
        alert_message = {
            "alert_type": "compliance_violation",
            "event_id": event.event_id,
            "severity": event.severity.value,
            "description": event.description,
            "timestamp": event.timestamp.isoformat(),
            "compliance_standards": [std.value for std in event.compliance_standards]
        }
        
        # Log alert (in production, this would send to monitoring system)
        logger.warning(f"COMPLIANCE ALERT: {json.dumps(alert_message)}")
    
    async def get_compliance_report(
        self,
        standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[EventType]] = None
    ) -> Dict[str, Any]:
        """Generate compliance report for a specific standard."""
        if not self.collection:
            return {"error": "MongoDB not configured"}
        
        # Build query
        query = {
            "compliance_standards": standard.value,
            "timestamp": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }
        }
        
        if event_types:
            query["event_type"] = {"$in": [et.value for et in event_types]}
        
        # Get events
        cursor = self.collection.find(query).sort("timestamp", -1)
        events = await cursor.to_list(length=None)
        
        # Generate report
        report = {
            "standard": standard.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(events),
            "events_by_type": {},
            "events_by_severity": {},
            "events": events
        }
        
        # Aggregate statistics
        for event in events:
            event_type = event.get("event_type", "unknown")
            severity = event.get("severity", "unknown")
            
            report["events_by_type"][event_type] = (
                report["events_by_type"].get(event_type, 0) + 1
            )
            report["events_by_severity"][severity] = (
                report["events_by_severity"].get(severity, 0) + 1
            )
        
        return report
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get compliance logging metrics."""
        return {
            "events_logged": self.events_logged,
            "events_by_type": {
                event_type.value: count
                for event_type, count in self.events_by_type.items()
            },
            "events_by_severity": {
                severity.value: count
                for severity, count in self.events_by_severity.items()
            },
            "buffer_size": len(self.event_buffer),
            "last_flush": self.last_flush.isoformat(),
            "enabled_standards": [std.value for std in self.config.enabled_standards]
        }
