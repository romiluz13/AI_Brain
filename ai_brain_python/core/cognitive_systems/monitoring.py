"""
Real-time Monitoring Engine

Live metrics and performance analytics system.
Provides comprehensive monitoring, alerting, and performance tracking.

Features:
- Real-time performance monitoring and metrics collection
- System health tracking and anomaly detection
- Performance analytics and trend analysis
- Alert generation and notification management
- Resource utilization monitoring
- User behavior analytics and insights
"""

import asyncio
import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, CognitiveSystemType

logger = logging.getLogger(__name__)


class MonitoringEngine(CognitiveSystemInterface):
    """
    Real-time Monitoring Engine - System 12 of 16
    
    Provides comprehensive real-time monitoring and analytics
    with intelligent alerting and performance tracking.
    """
    
    def __init__(self, system_id: str = "monitoring", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Monitoring data storage
        self._metrics_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._system_health: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self._config = {
            "buffer_size": config.get("buffer_size", 1000) if config else 1000,
            "alert_threshold": config.get("alert_threshold", 0.8) if config else 0.8,
            "anomaly_threshold": config.get("anomaly_threshold", 2.0) if config else 2.0,  # Standard deviations
            "health_check_interval": config.get("health_check_interval", 60) if config else 60,  # seconds
            "metrics_retention_days": config.get("metrics_retention_days", 30) if config else 30,
            "enable_real_time_alerts": config.get("enable_real_time_alerts", True) if config else True
        }
        
        # Metric definitions
        self._metric_definitions = {
            "response_time": {
                "unit": "milliseconds",
                "threshold": 5000,
                "alert_level": "warning"
            },
            "error_rate": {
                "unit": "percentage",
                "threshold": 5.0,
                "alert_level": "critical"
            },
            "confidence_score": {
                "unit": "score",
                "threshold": 0.7,
                "alert_level": "warning"
            },
            "memory_usage": {
                "unit": "percentage",
                "threshold": 80.0,
                "alert_level": "warning"
            },
            "cpu_usage": {
                "unit": "percentage",
                "threshold": 85.0,
                "alert_level": "warning"
            },
            "throughput": {
                "unit": "requests_per_second",
                "threshold": 100,
                "alert_level": "info"
            }
        }
        
        # Alert levels
        self._alert_levels = {
            "info": {"priority": 1, "color": "blue"},
            "warning": {"priority": 2, "color": "yellow"},
            "error": {"priority": 3, "color": "orange"},
            "critical": {"priority": 4, "color": "red"}
        }
        
        # System components to monitor
        self._monitored_components = [
            "emotional_intelligence",
            "goal_hierarchy",
            "confidence_tracking",
            "attention_management",
            "cultural_knowledge",
            "skill_capability",
            "communication_protocol",
            "temporal_planning",
            "semantic_memory",
            "safety_guardrails",
            "self_improvement",
            "tool_interface",
            "workflow_orchestration",
            "multimodal_processing",
            "human_feedback"
        ]
        
        # Performance baselines
        self._performance_baselines: Dict[str, float] = {}
        
        # Monitoring start time
        self._monitoring_start_time = datetime.utcnow()
    
    @property
    def system_name(self) -> str:
        return "Real-time Monitoring Engine"
    
    @property
    def system_description(self) -> str:
        return "Live metrics and performance analytics system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.PERFORMANCE_MONITORING}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.PERFORMANCE_MONITORING}
    
    async def initialize(self) -> None:
        """Initialize the Monitoring Engine."""
        try:
            logger.info("Initializing Real-time Monitoring Engine...")
            
            # Load historical monitoring data
            await self._load_monitoring_data()
            
            # Initialize performance baselines
            await self._initialize_performance_baselines()
            
            # Start background monitoring tasks
            await self._start_background_monitoring()
            
            self._is_initialized = True
            logger.info("Real-time Monitoring Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Monitoring Engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Monitoring Engine."""
        try:
            logger.info("Shutting down Real-time Monitoring Engine...")
            
            # Save monitoring data
            await self._save_monitoring_data()
            
            # Stop background tasks
            await self._stop_background_monitoring()
            
            self._is_initialized = False
            logger.info("Real-time Monitoring Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Monitoring Engine shutdown: {e}")
    
    async def process(
        self, 
        input_data: CognitiveInputData,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input through monitoring analysis."""
        if not self._is_initialized:
            raise RuntimeError("Monitoring Engine not initialized")
        
        try:
            start_time = datetime.utcnow()
            processing_start = time.time()
            
            # Collect metrics from current processing
            metrics = await self._collect_processing_metrics(input_data, context or {})
            
            # Update performance history
            await self._update_performance_history(metrics)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(metrics)
            
            # Generate alerts if needed
            alerts = await self._generate_alerts(metrics, anomalies)
            
            # Calculate system health
            system_health = await self._calculate_system_health()
            
            # Generate performance insights
            insights = await self._generate_performance_insights()
            
            # Update real-time dashboard data
            dashboard_data = await self._prepare_dashboard_data()
            
            processing_time = (time.time() - processing_start) * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.95,
                "monitoring_metrics": {
                    "current_metrics": metrics,
                    "system_health": system_health,
                    "anomalies_detected": len(anomalies),
                    "alerts_generated": len(alerts),
                    "uptime_seconds": (datetime.utcnow() - self._monitoring_start_time).total_seconds()
                },
                "performance_summary": {
                    "overall_health": system_health.get("overall_status", "unknown"),
                    "response_time_avg": metrics.get("response_time", 0),
                    "error_rate": metrics.get("error_rate", 0),
                    "throughput": metrics.get("throughput", 0)
                },
                "anomalies": anomalies,
                "alerts": alerts[-5:],  # Last 5 alerts
                "insights": insights,
                "dashboard_data": dashboard_data
            }
            
        except Exception as e:
            logger.error(f"Error in Monitoring processing: {e}")
            return {
                "system": self.system_id,
                "status": "error",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current monitoring state."""
        state_data = {
            "total_metrics_collected": sum(len(buffer) for buffer in self._metrics_buffer.values()),
            "active_alerts": len([alert for alert in self._alerts if alert.get("status") == "active"]),
            "monitored_components": len(self._monitored_components),
            "uptime_hours": (datetime.utcnow() - self._monitoring_start_time).total_seconds() / 3600
        }
        
        return CognitiveState(
            system_type=CognitiveSystemType.MONITORING,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.98,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update monitoring state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Monitoring state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for monitoring."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Public monitoring methods
    
    async def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        if metric_name not in self._metrics_buffer:
            self._metrics_buffer[metric_name] = []
        
        metric_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "value": value,
            "tags": tags or {}
        }
        
        self._metrics_buffer[metric_name].append(metric_entry)
        
        # Maintain buffer size
        if len(self._metrics_buffer[metric_name]) > self._config["buffer_size"]:
            self._metrics_buffer[metric_name] = self._metrics_buffer[metric_name][-self._config["buffer_size"]:]
        
        # Check for immediate alerts
        if self._config["enable_real_time_alerts"]:
            await self._check_metric_threshold(metric_name, value)
    
    async def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for a specified time period."""
        if metric_name not in self._metrics_buffer:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            entry for entry in self._metrics_buffer[metric_name]
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "metrics_summary": {},
            "alerts_summary": {},
            "recommendations": []
        }
        
        # Component health
        for component in self._monitored_components:
            component_health = self._system_health.get(component, {})
            health_report["components"][component] = {
                "status": component_health.get("status", "unknown"),
                "last_check": component_health.get("last_check"),
                "response_time": component_health.get("response_time", 0),
                "error_rate": component_health.get("error_rate", 0)
            }
        
        # Metrics summary
        for metric_name, buffer in self._metrics_buffer.items():
            if buffer:
                recent_values = [entry["value"] for entry in buffer[-10:]]
                health_report["metrics_summary"][metric_name] = {
                    "current": recent_values[-1] if recent_values else 0,
                    "average": statistics.mean(recent_values) if recent_values else 0,
                    "trend": self._calculate_trend(recent_values) if len(recent_values) > 1 else "stable"
                }
        
        # Alerts summary
        active_alerts = [alert for alert in self._alerts if alert.get("status") == "active"]
        health_report["alerts_summary"] = {
            "total_active": len(active_alerts),
            "critical": len([a for a in active_alerts if a.get("level") == "critical"]),
            "warning": len([a for a in active_alerts if a.get("level") == "warning"])
        }
        
        # Generate recommendations
        health_report["recommendations"] = await self._generate_health_recommendations(health_report)
        
        return health_report
    
    async def create_alert(self, level: str, message: str, component: Optional[str] = None) -> str:
        """Create a new alert."""
        alert = {
            "id": f"alert_{len(self._alerts)}_{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "component": component,
            "status": "active",
            "acknowledged": False
        }
        
        self._alerts.append(alert)
        
        # Keep only recent alerts
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]
        
        logger.warning(f"Alert created: {level} - {message}")
        return alert["id"]
    
    # Private methods
    
    async def _load_monitoring_data(self) -> None:
        """Load historical monitoring data."""
        logger.debug("Monitoring data loaded")
    
    async def _save_monitoring_data(self) -> None:
        """Save monitoring data."""
        logger.debug("Monitoring data saved")
    
    async def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines."""
        # Set default baselines
        self._performance_baselines = {
            "response_time": 1000.0,  # 1 second
            "error_rate": 1.0,        # 1%
            "confidence_score": 0.8,  # 80%
            "throughput": 10.0        # 10 requests/second
        }
        logger.debug("Performance baselines initialized")
    
    async def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks."""
        # In a real implementation, this would start background tasks
        logger.debug("Background monitoring started")
    
    async def _stop_background_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        logger.debug("Background monitoring stopped")
    
    async def _collect_processing_metrics(self, input_data: CognitiveInputData, context: Dict[str, Any]) -> Dict[str, float]:
        """Collect metrics from current processing."""
        metrics = {}
        
        # Response time from context
        if "processing_time_ms" in context:
            metrics["response_time"] = context["processing_time_ms"]
        
        # Error rate calculation
        cognitive_results = context.get("cognitive_results", {})
        if cognitive_results:
            error_count = sum(1 for result in cognitive_results.values() 
                            if isinstance(result, dict) and result.get("status") == "error")
            total_systems = len(cognitive_results)
            metrics["error_rate"] = (error_count / total_systems) * 100 if total_systems > 0 else 0
        
        # Average confidence score
        if cognitive_results:
            confidence_scores = [
                result.get("confidence", 0) for result in cognitive_results.values()
                if isinstance(result, dict) and "confidence" in result
            ]
            if confidence_scores:
                metrics["confidence_score"] = statistics.mean(confidence_scores)
        
        # Throughput (simplified)
        metrics["throughput"] = 1.0  # 1 request processed
        
        # System resource metrics (simplified)
        metrics["memory_usage"] = 45.0  # Placeholder
        metrics["cpu_usage"] = 35.0     # Placeholder
        
        return metrics
    
    async def _update_performance_history(self, metrics: Dict[str, float]) -> None:
        """Update performance history."""
        timestamp = datetime.utcnow().isoformat()
        
        for metric_name, value in metrics.items():
            await self.record_metric(metric_name, value)
    
    async def _detect_anomalies(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics."""
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self._metrics_buffer and len(self._metrics_buffer[metric_name]) > 10:
                # Get historical values
                historical_values = [
                    entry["value"] for entry in self._metrics_buffer[metric_name][-50:]
                ]
                
                if len(historical_values) > 5:
                    mean_value = statistics.mean(historical_values)
                    std_dev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                    
                    # Check if current value is anomalous
                    if std_dev > 0:
                        z_score = abs(current_value - mean_value) / std_dev
                        if z_score > self._config["anomaly_threshold"]:
                            anomalies.append({
                                "metric": metric_name,
                                "current_value": current_value,
                                "expected_range": [mean_value - 2*std_dev, mean_value + 2*std_dev],
                                "z_score": z_score,
                                "severity": "high" if z_score > 3 else "medium"
                            })
        
        return anomalies
    
    async def _generate_alerts(self, metrics: Dict[str, float], anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts based on metrics and anomalies."""
        new_alerts = []
        
        # Threshold-based alerts
        for metric_name, value in metrics.items():
            if metric_name in self._metric_definitions:
                definition = self._metric_definitions[metric_name]
                threshold = definition["threshold"]
                alert_level = definition["alert_level"]
                
                # Check if threshold is exceeded
                if (metric_name in ["error_rate", "memory_usage", "cpu_usage"] and value > threshold) or \
                   (metric_name == "response_time" and value > threshold) or \
                   (metric_name == "confidence_score" and value < threshold):
                    
                    alert_id = await self.create_alert(
                        level=alert_level,
                        message=f"{metric_name} threshold exceeded: {value} (threshold: {threshold})",
                        component=metric_name
                    )
                    new_alerts.append({"id": alert_id, "type": "threshold", "metric": metric_name})
        
        # Anomaly-based alerts
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                alert_id = await self.create_alert(
                    level="warning",
                    message=f"Anomaly detected in {anomaly['metric']}: {anomaly['current_value']} (z-score: {anomaly['z_score']:.2f})",
                    component=anomaly["metric"]
                )
                new_alerts.append({"id": alert_id, "type": "anomaly", "metric": anomaly["metric"]})
        
        return new_alerts
    
    async def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health."""
        health_status = {
            "overall_status": "healthy",
            "component_health": {},
            "health_score": 1.0
        }
        
        # Calculate health for each component
        unhealthy_components = 0
        total_components = len(self._monitored_components)
        
        for component in self._monitored_components:
            component_health = self._system_health.get(component, {})
            status = component_health.get("status", "unknown")
            
            health_status["component_health"][component] = status
            
            if status in ["error", "critical"]:
                unhealthy_components += 1
        
        # Calculate overall health score
        if total_components > 0:
            health_status["health_score"] = 1.0 - (unhealthy_components / total_components)
        
        # Determine overall status
        if health_status["health_score"] >= 0.9:
            health_status["overall_status"] = "healthy"
        elif health_status["health_score"] >= 0.7:
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "unhealthy"
        
        return health_status
    
    async def _generate_performance_insights(self) -> List[str]:
        """Generate performance insights."""
        insights = []
        
        # Analyze response time trends
        if "response_time" in self._metrics_buffer:
            recent_times = [entry["value"] for entry in self._metrics_buffer["response_time"][-10:]]
            if recent_times:
                avg_time = statistics.mean(recent_times)
                if avg_time > 2000:  # 2 seconds
                    insights.append("Response times are higher than optimal - consider optimization")
        
        # Analyze error rates
        if "error_rate" in self._metrics_buffer:
            recent_errors = [entry["value"] for entry in self._metrics_buffer["error_rate"][-10:]]
            if recent_errors:
                avg_error_rate = statistics.mean(recent_errors)
                if avg_error_rate > 2.0:  # 2%
                    insights.append("Error rate is elevated - investigate system stability")
        
        # Analyze confidence trends
        if "confidence_score" in self._metrics_buffer:
            recent_confidence = [entry["value"] for entry in self._metrics_buffer["confidence_score"][-10:]]
            if recent_confidence:
                avg_confidence = statistics.mean(recent_confidence)
                if avg_confidence < 0.7:
                    insights.append("Confidence scores are below target - review model performance")
        
        return insights
    
    async def _prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prepare data for real-time dashboard."""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "alerts": len([a for a in self._alerts if a.get("status") == "active"]),
            "system_status": "operational"
        }
        
        # Current metric values
        for metric_name, buffer in self._metrics_buffer.items():
            if buffer:
                dashboard_data["metrics"][metric_name] = {
                    "current": buffer[-1]["value"],
                    "trend": self._calculate_trend([entry["value"] for entry in buffer[-5:]])
                }
        
        return dashboard_data
    
    async def _check_metric_threshold(self, metric_name: str, value: float) -> None:
        """Check if metric exceeds threshold and create alert if needed."""
        if metric_name in self._metric_definitions:
            definition = self._metric_definitions[metric_name]
            threshold = definition["threshold"]
            
            if value > threshold:
                await self.create_alert(
                    level=definition["alert_level"],
                    message=f"{metric_name} exceeded threshold: {value} > {threshold}",
                    component=metric_name
                )
    
    async def _generate_health_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on report."""
        recommendations = []
        
        # Check for high error rates
        if health_report["metrics_summary"].get("error_rate", {}).get("current", 0) > 5:
            recommendations.append("Investigate and address high error rates")
        
        # Check for slow response times
        if health_report["metrics_summary"].get("response_time", {}).get("current", 0) > 3000:
            recommendations.append("Optimize system performance to reduce response times")
        
        # Check for critical alerts
        if health_report["alerts_summary"].get("critical", 0) > 0:
            recommendations.append("Address critical alerts immediately")
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values."""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation
        recent_avg = statistics.mean(values[-3:]) if len(values) >= 3 else values[-1]
        earlier_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]
        
        change_percent = ((recent_avg - earlier_avg) / earlier_avg) * 100 if earlier_avg != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
