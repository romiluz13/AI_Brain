"""
Real-time Monitoring System for AI Brain.

Provides comprehensive monitoring of system performance, safety metrics,
and operational health with real-time alerts and dashboards.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    SAFETY = "safety"
    USAGE = "usage"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class Metric:
    """Represents a system metric."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MonitoringConfig(BaseModel):
    """Configuration for real-time monitoring."""
    
    # Metric collection settings
    collection_interval: int = Field(default=10)  # seconds
    metric_retention_hours: int = Field(default=24)
    
    # Alert settings
    enable_alerts: bool = Field(default=True)
    alert_cooldown_minutes: int = Field(default=5)
    
    # Performance thresholds
    max_response_time_ms: float = Field(default=5000)
    max_memory_usage_mb: float = Field(default=1024)
    max_cpu_usage_percent: float = Field(default=80)
    min_success_rate_percent: float = Field(default=95)
    
    # Safety thresholds
    max_safety_violations_per_hour: int = Field(default=10)
    max_hallucination_rate_percent: float = Field(default=5)
    max_pii_exposure_rate_percent: float = Field(default=1)
    
    # System thresholds
    max_error_rate_percent: float = Field(default=5)
    max_queue_size: int = Field(default=1000)
    min_available_connections: int = Field(default=5)
    
    # Notification settings
    webhook_urls: List[str] = Field(default_factory=list)
    email_recipients: List[str] = Field(default_factory=list)


class MetricCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_collection = datetime.utcnow()
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_count = 0
        self.total_requests = 0
    
    async def collect_metrics(self) -> None:
        """Collect system metrics."""
        timestamp = datetime.utcnow()
        
        # Performance metrics
        await self._collect_performance_metrics(timestamp)
        
        # System metrics
        await self._collect_system_metrics(timestamp)
        
        # Safety metrics
        await self._collect_safety_metrics(timestamp)
        
        # Usage metrics
        await self._collect_usage_metrics(timestamp)
        
        self.last_collection = timestamp
    
    async def _collect_performance_metrics(self, timestamp: datetime) -> None:
        """Collect performance-related metrics."""
        # Average response time
        if self.request_times:
            avg_response_time = sum(self.request_times) / len(self.request_times)
            self.add_metric(Metric(
                name="avg_response_time_ms",
                value=avg_response_time,
                timestamp=timestamp,
                metric_type=MetricType.PERFORMANCE,
                unit="milliseconds"
            ))
        
        # Success rate
        if self.total_requests > 0:
            success_rate = (self.success_count / self.total_requests) * 100
            self.add_metric(Metric(
                name="success_rate_percent",
                value=success_rate,
                timestamp=timestamp,
                metric_type=MetricType.PERFORMANCE,
                unit="percent"
            ))
        
        # Error rate
        total_errors = sum(self.error_counts.values())
        if self.total_requests > 0:
            error_rate = (total_errors / self.total_requests) * 100
            self.add_metric(Metric(
                name="error_rate_percent",
                value=error_rate,
                timestamp=timestamp,
                metric_type=MetricType.ERROR,
                unit="percent"
            ))
    
    async def _collect_system_metrics(self, timestamp: datetime) -> None:
        """Collect system-related metrics."""
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.add_metric(Metric(
            name="cpu_usage_percent",
            value=cpu_percent,
            timestamp=timestamp,
            metric_type=MetricType.SYSTEM,
            unit="percent"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        self.add_metric(Metric(
            name="memory_usage_mb",
            value=memory_mb,
            timestamp=timestamp,
            metric_type=MetricType.SYSTEM,
            unit="megabytes"
        ))
        
        # Memory percentage
        self.add_metric(Metric(
            name="memory_usage_percent",
            value=memory.percent,
            timestamp=timestamp,
            metric_type=MetricType.SYSTEM,
            unit="percent"
        ))
    
    async def _collect_safety_metrics(self, timestamp: datetime) -> None:
        """Collect safety-related metrics."""
        # These would be populated by the safety systems
        # For now, we'll create placeholder metrics
        
        self.add_metric(Metric(
            name="safety_violations_count",
            value=0,  # Would be updated by safety guardrails
            timestamp=timestamp,
            metric_type=MetricType.SAFETY,
            unit="count"
        ))
        
        self.add_metric(Metric(
            name="hallucination_rate_percent",
            value=0,  # Would be updated by hallucination detector
            timestamp=timestamp,
            metric_type=MetricType.SAFETY,
            unit="percent"
        ))
    
    async def _collect_usage_metrics(self, timestamp: datetime) -> None:
        """Collect usage-related metrics."""
        self.add_metric(Metric(
            name="total_requests",
            value=self.total_requests,
            timestamp=timestamp,
            metric_type=MetricType.USAGE,
            unit="count"
        ))
        
        self.add_metric(Metric(
            name="requests_per_minute",
            value=len([t for t in self.request_times if timestamp - timedelta(minutes=1) <= datetime.fromtimestamp(t)]),
            timestamp=timestamp,
            metric_type=MetricType.USAGE,
            unit="count"
        ))
    
    def add_metric(self, metric: Metric) -> None:
        """Add a metric to the collection."""
        self.metrics[metric.name].append(metric)
        
        # Clean up old metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.metric_retention_hours)
        while (self.metrics[metric.name] and 
               self.metrics[metric.name][0].timestamp < cutoff_time):
            self.metrics[metric.name].popleft()
    
    def record_request(self, response_time_ms: float, success: bool, error_type: Optional[str] = None) -> None:
        """Record a request for metrics."""
        self.request_times.append(response_time_ms)
        self.total_requests += 1
        
        if success:
            self.success_count += 1
        elif error_type:
            self.error_counts[error_type] += 1
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Metric]:
        """Get metric history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            metric for metric in self.metrics[metric_name]
            if metric.timestamp >= cutoff_time
        ]
    
    def get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Get the latest value for a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
    
    async def check_thresholds(self, metrics: Dict[str, Metric]) -> List[Alert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        if not self.config.enable_alerts:
            return alerts
        
        # Performance alerts
        alerts.extend(await self._check_performance_thresholds(metrics))
        
        # Safety alerts
        alerts.extend(await self._check_safety_thresholds(metrics))
        
        # System alerts
        alerts.extend(await self._check_system_thresholds(metrics))
        
        # Process new alerts
        for alert in alerts:
            await self._process_alert(alert)
        
        return alerts
    
    async def _check_performance_thresholds(self, metrics: Dict[str, Metric]) -> List[Alert]:
        """Check performance thresholds."""
        alerts = []
        
        # Response time
        if "avg_response_time_ms" in metrics:
            metric = metrics["avg_response_time_ms"]
            if metric.value > self.config.max_response_time_ms:
                alert = Alert(
                    alert_id=f"perf_response_time_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    title="High Response Time",
                    description=f"Average response time ({metric.value:.2f}ms) exceeds threshold ({self.config.max_response_time_ms}ms)",
                    timestamp=metric.timestamp,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=self.config.max_response_time_ms
                )
                alerts.append(alert)
        
        # Success rate
        if "success_rate_percent" in metrics:
            metric = metrics["success_rate_percent"]
            if metric.value < self.config.min_success_rate_percent:
                alert = Alert(
                    alert_id=f"perf_success_rate_{int(time.time())}",
                    level=AlertLevel.ERROR,
                    title="Low Success Rate",
                    description=f"Success rate ({metric.value:.2f}%) below threshold ({self.config.min_success_rate_percent}%)",
                    timestamp=metric.timestamp,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=self.config.min_success_rate_percent
                )
                alerts.append(alert)
        
        return alerts
    
    async def _check_safety_thresholds(self, metrics: Dict[str, Metric]) -> List[Alert]:
        """Check safety thresholds."""
        alerts = []
        
        # Hallucination rate
        if "hallucination_rate_percent" in metrics:
            metric = metrics["hallucination_rate_percent"]
            if metric.value > self.config.max_hallucination_rate_percent:
                alert = Alert(
                    alert_id=f"safety_hallucination_{int(time.time())}",
                    level=AlertLevel.CRITICAL,
                    title="High Hallucination Rate",
                    description=f"Hallucination rate ({metric.value:.2f}%) exceeds threshold ({self.config.max_hallucination_rate_percent}%)",
                    timestamp=metric.timestamp,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=self.config.max_hallucination_rate_percent
                )
                alerts.append(alert)
        
        return alerts
    
    async def _check_system_thresholds(self, metrics: Dict[str, Metric]) -> List[Alert]:
        """Check system thresholds."""
        alerts = []
        
        # CPU usage
        if "cpu_usage_percent" in metrics:
            metric = metrics["cpu_usage_percent"]
            if metric.value > self.config.max_cpu_usage_percent:
                alert = Alert(
                    alert_id=f"system_cpu_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    title="High CPU Usage",
                    description=f"CPU usage ({metric.value:.2f}%) exceeds threshold ({self.config.max_cpu_usage_percent}%)",
                    timestamp=metric.timestamp,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=self.config.max_cpu_usage_percent
                )
                alerts.append(alert)
        
        # Memory usage
        if "memory_usage_mb" in metrics:
            metric = metrics["memory_usage_mb"]
            if metric.value > self.config.max_memory_usage_mb:
                alert = Alert(
                    alert_id=f"system_memory_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    title="High Memory Usage",
                    description=f"Memory usage ({metric.value:.2f}MB) exceeds threshold ({self.config.max_memory_usage_mb}MB)",
                    timestamp=metric.timestamp,
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=self.config.max_memory_usage_mb
                )
                alerts.append(alert)
        
        return alerts
    
    async def _process_alert(self, alert: Alert) -> None:
        """Process a new alert."""
        # Check cooldown
        if self._is_in_cooldown(alert):
            return
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert.metric_name or "unknown"] = alert.timestamp
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"ALERT: {alert.title} - {alert.description}")
    
    def _is_in_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period."""
        if not alert.metric_name:
            return False
        
        last_alert_time = self.last_alert_times.get(alert.metric_name)
        if not last_alert_time:
            return False
        
        cooldown_period = timedelta(minutes=self.config.alert_cooldown_minutes)
        return (alert.timestamp - last_alert_time) < cooldown_period
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications."""
        # In a real implementation, this would send to webhooks, email, etc.
        notification_data = {
            "alert_id": alert.alert_id,
            "level": alert.level.value,
            "title": alert.title,
            "description": alert.description,
            "timestamp": alert.timestamp.isoformat(),
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold": alert.threshold
        }
        
        logger.info(f"Sending alert notification: {notification_data}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]


class RealTimeMonitor:
    """Main real-time monitoring system."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.metric_collector = MetricCollector(self.config)
        self.alert_manager = AlertManager(self.config)
        
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Real-time monitoring started")
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Real-time monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                await self.metric_collector.collect_metrics()
                
                # Get latest metrics
                latest_metrics = {}
                for metric_name in self.metric_collector.metrics:
                    latest_metric = self.metric_collector.get_latest_metric(metric_name)
                    if latest_metric:
                        latest_metrics[metric_name] = latest_metric
                
                # Check thresholds and generate alerts
                await self.alert_manager.check_thresholds(latest_metrics)
                
                # Wait for next collection interval
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.collection_interval)
    
    def record_request(self, response_time_ms: float, success: bool, error_type: Optional[str] = None) -> None:
        """Record a request for monitoring."""
        self.metric_collector.record_request(response_time_ms, success, error_type)
    
    def add_custom_metric(self, metric: Metric) -> None:
        """Add a custom metric."""
        self.metric_collector.add_metric(metric)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "system_status": "healthy" if not self.alert_manager.active_alerts else "degraded",
            "active_alerts": len(self.alert_manager.active_alerts),
            "metrics": {
                name: {
                    "current": self.metric_collector.get_latest_metric(name).value
                    if self.metric_collector.get_latest_metric(name) else None,
                    "history": [
                        {"timestamp": m.timestamp.isoformat(), "value": m.value}
                        for m in self.metric_collector.get_metric_history(name, hours=1)
                    ]
                }
                for name in self.metric_collector.metrics
            },
            "alerts": [
                {
                    "id": alert.alert_id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alert_manager.get_active_alerts()
            ]
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        error_alerts = [a for a in active_alerts if a.level == AlertLevel.ERROR]
        
        if critical_alerts:
            status = "critical"
        elif error_alerts:
            status = "degraded"
        elif active_alerts:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "error_alerts": len(error_alerts),
            "uptime_seconds": 0,  # Would track actual uptime in real implementation
            "monitoring_enabled": self.running
        }
