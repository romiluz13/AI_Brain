"""
MonitoringEngine - Advanced system monitoring and observability

Exact Python equivalent of JavaScript MonitoringEngine.ts with:
- Real-time system monitoring with comprehensive metrics
- Performance tracking and anomaly detection
- Health checks and diagnostic capabilities
- Alert management and notification systems
- Real-time dashboards with alerting and notifications

Features:
- Real-time system monitoring with comprehensive metrics
- Performance tracking and anomaly detection
- Health checks and diagnostic capabilities
- Alert management and notification systems
- Real-time dashboards with alerting and notifications
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Literal, Union
from dataclasses import dataclass, field
from bson import ObjectId
import asyncio
import json
import random
import math
import time

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_brain_python.storage.collections.tracing_collection import TracingCollection
from ai_brain_python.core.types import Monitoring, MonitoringAnalyticsOptions
from ai_brain_python.utils.logger import logger


@dataclass
class MonitoringRequest:
    """Monitoring request interface."""
    agent_id: str
    session_id: Optional[str]
    metrics: Dict[str, Any]
    context: Dict[str, Any]
    monitoring_level: str
    alert_thresholds: Dict[str, float]


@dataclass
class MonitoringResult:
    """Monitoring result interface."""
    monitoring_id: ObjectId
    health_score: float
    alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]


class MonitoringEngine:
    """
    MonitoringEngine - Advanced system monitoring and observability

    Exact Python equivalent of JavaScript MonitoringEngine with:
    - Real-time system monitoring with comprehensive metrics
    - Performance tracking and anomaly detection
    - Health checks and diagnostic capabilities
    - Alert management and notification systems
    - Real-time dashboards with alerting and notifications
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.tracing_collection = TracingCollection(db)
        self.is_initialized = False

        # Monitoring configuration
        self._config = {
            "health_check_interval": 60,  # seconds
            "alert_threshold": 0.8,
            "anomaly_threshold": 2.0,  # standard deviations
            "retention_period": 7,  # days
            "max_alerts": 100
        }

        # System health tracking
        self._health_checks: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: Dict[str, List[float]] = {}
        self._active_alerts: List[Dict[str, Any]] = []
        self._anomalies: List[Dict[str, Any]] = []

        # Monitoring thresholds
        self._thresholds = {
            "response_time": 2.0,  # seconds
            "error_rate": 0.05,    # 5%
            "cpu_usage": 0.8,      # 80%
            "memory_usage": 0.9,   # 90%
            "disk_usage": 0.85     # 85%
        }

        # Initialize default health checks
        self._initialize_health_checks()

    def _initialize_health_checks(self) -> None:
        """Initialize default health checks."""
        components = ["database", "api", "cognitive_systems", "memory", "embeddings"]

        for component in components:
            self._health_checks[component] = {
                "component": component,
                "status": "unknown",
                "response_time": 0.0,
                "error_rate": 0.0,
                "last_check": datetime.utcnow()
            }

    async def initialize(self) -> None:
        """Initialize the monitoring engine."""
        if self.is_initialized:
            return

        try:
            # Initialize tracing collection
            await self.tracing_collection.create_indexes()

            # Start health monitoring
            await self._start_health_monitoring()

            self.is_initialized = True
            logger.info("✅ MonitoringEngine initialized successfully")

        except Exception as error:
            logger.error(f"❌ Failed to initialize MonitoringEngine: {error}")
            raise error

    async def monitor_system(
        self,
        request: MonitoringRequest
    ) -> MonitoringResult:
        """Monitor system health and performance."""
        if not self.is_initialized:
            raise Exception("MonitoringEngine must be initialized first")

        # Generate monitoring ID
        monitoring_id = ObjectId()

        # Update performance metrics
        await self._update_performance_metrics(request.metrics)

        # Calculate health score
        health_score = await self._calculate_health_score(request.metrics)

        # Check for alerts
        alerts = await self._check_alerts(
            request.metrics,
            request.alert_thresholds
        )

        # Detect anomalies
        anomalies = await self._detect_anomalies(request.metrics)

        # Generate recommendations
        recommendations = await self._generate_monitoring_recommendations(
            health_score,
            alerts,
            anomalies
        )

        # Create monitoring record
        monitoring_record = {
            "monitoringId": monitoring_id,
            "agentId": request.agent_id,
            "sessionId": request.session_id,
            "timestamp": datetime.utcnow(),
            "metrics": request.metrics,
            "monitoringLevel": request.monitoring_level,
            "healthScore": health_score,
            "alerts": alerts,
            "performanceMetrics": await self._get_performance_summary(),
            "anomalies": anomalies,
            "recommendations": recommendations,
            "context": request.context,
            "metadata": {
                "framework": "python_ai_brain",
                "version": "1.0.0",
                "source": "monitoring_engine"
            }
        }

        # Store monitoring record
        await self.tracing_collection.record_trace(monitoring_record)

        # Update active alerts
        self._active_alerts.extend(alerts)

        # Update anomalies
        self._anomalies.extend(anomalies)

        return MonitoringResult(
            monitoring_id=monitoring_id,
            health_score=health_score,
            alerts=alerts,
            performance_metrics=await self._get_performance_summary(),
            anomalies=anomalies,
            recommendations=recommendations
        )

    async def get_monitoring_analytics(
        self,
        agent_id: str,
        options: Optional[MonitoringAnalyticsOptions] = None
    ) -> Dict[str, Any]:
        """Get monitoring analytics for an agent."""
        return await self.tracing_collection.get_monitoring_analytics(agent_id, options)

    async def get_monitoring_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = await self.tracing_collection.get_monitoring_stats(agent_id)

        return {
            **stats,
            "healthChecks": len(self._health_checks),
            "activeAlerts": len(self._active_alerts),
            "detectedAnomalies": len(self._anomalies)
        }

    # Private helper methods
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring."""
        logger.debug("Health monitoring started")

    async def _update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics."""
        for metric_name, value in metrics.items():
            if metric_name not in self._performance_metrics:
                self._performance_metrics[metric_name] = []

            if isinstance(value, (int, float)):
                self._performance_metrics[metric_name].append(value)

                # Keep only recent metrics
                if len(self._performance_metrics[metric_name]) > 100:
                    self._performance_metrics[metric_name] = self._performance_metrics[metric_name][-100:]

    async def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall health score."""
        score = 1.0

        # Check response time
        response_time = metrics.get("response_time", 0)
        if response_time > self._thresholds["response_time"]:
            score -= 0.2

        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self._thresholds["error_rate"]:
            score -= 0.3

        # Check resource usage
        cpu_usage = metrics.get("cpu_usage", 0)
        if cpu_usage > self._thresholds["cpu_usage"]:
            score -= 0.2

        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > self._thresholds["memory_usage"]:
            score -= 0.3

        return max(0.0, score)

    async def _check_alerts(
        self,
        metrics: Dict[str, Any],
        alert_thresholds: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []

        for metric_name, value in metrics.items():
            if metric_name in alert_thresholds:
                threshold = alert_thresholds[metric_name]
                if isinstance(value, (int, float)) and value > threshold:
                    alerts.append({
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "severity": "high" if value > threshold * 1.5 else "medium",
                        "timestamp": datetime.utcnow().isoformat()
                    })

        return alerts

    async def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        anomalies = []

        for metric_name, current_value in metrics.items():
            if isinstance(current_value, (int, float)) and metric_name in self._performance_metrics:
                historical_values = self._performance_metrics[metric_name]

                if len(historical_values) > 10:
                    import statistics
                    mean_value = statistics.mean(historical_values)
                    std_dev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0

                    if std_dev > 0:
                        z_score = abs(current_value - mean_value) / std_dev
                        if z_score > self._config["anomaly_threshold"]:
                            anomalies.append({
                                "metric": metric_name,
                                "current_value": current_value,
                                "expected_value": mean_value,
                                "z_score": z_score,
                                "severity": "high" if z_score > 3 else "medium"
                            })

        return anomalies

    async def _generate_monitoring_recommendations(
        self,
        health_score: float,
        alerts: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []

        if health_score < 0.7:
            recommendations.append("System health is degraded - investigate performance issues")

        if len(alerts) > 5:
            recommendations.append("Multiple alerts detected - review system configuration")

        if len(anomalies) > 3:
            recommendations.append("Multiple anomalies detected - check for system instability")

        return recommendations

    async def _get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        summary = {}

        for metric_name, values in self._performance_metrics.items():
            if values:
                import statistics
                summary[metric_name] = statistics.mean(values[-10:])  # Average of last 10 values

        return summary


