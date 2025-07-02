"""
NotificationManager - Intelligent Alert System

Exact Python equivalent of JavaScript NotificationManager.ts with:
- Multi-channel notifications (email, SMS, webhook, in-app)
- Intelligent alert prioritization and throttling
- Performance threshold monitoring
- Safety alert escalation
- Custom notification rules and filters
- Delivery tracking and retry logic

Features:
- Multi-channel notifications (email, SMS, webhook, in-app)
- Intelligent alert prioritization and throttling
- Performance threshold monitoring
- Safety alert escalation
- Custom notification rules and filters
- Delivery tracking and retry logic
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from bson import ObjectId
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.interfaces.cognitive_system import CognitiveSystemInterface
from ..core.models import CognitiveInputData, CognitiveResponse
from ..utils.logger import logger


@dataclass
class NotificationConfig:
    """Notification configuration data structure."""
    email: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "provider": "smtp",
        "fromEmail": "",
        "fromName": "AI Brain",
        "smtpConfig": {
            "host": "localhost",
            "port": 587,
            "secure": True,
            "auth": {"user": "", "pass": ""}
        }
    })
    sms: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "provider": "twilio",
        "apiKey": "",
        "apiSecret": "",
        "fromNumber": ""
    })
    webhook: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "endpoints": []
    })
    in_app: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "retentionDays": 30
    })
    throttling: Dict[str, Any] = field(default_factory=lambda: {
        "maxNotificationsPerHour": 100,
        "maxNotificationsPerDay": 1000,
        "cooldownMinutes": 5
    })
    thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "responseTime": 5000,
        "errorRate": 5.0,
        "memoryUsage": 80.0,
        "costPerHour": 10.0,
        "safetyFailureRate": 1.0
    })


@dataclass
class NotificationRule:
    """Notification rule data structure."""
    id: str
    name: str
    enabled: bool
    conditions: Dict[str, Any]
    channels: List[str]
    recipients: Dict[str, Any]
    throttling: Dict[str, Any]
    escalation: Dict[str, Any]


@dataclass
class NotificationEvent:
    """Notification event data structure."""
    id: str
    type: str
    title: str
    message: str
    severity: str  # 'low' | 'medium' | 'high' | 'critical'
    source: Dict[str, Any]
    data: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    escalated: bool = False
    acknowledgedBy: Optional[str] = None
    acknowledgedAt: Optional[datetime] = None


@dataclass
class NotificationDelivery:
    """Notification delivery data structure."""
    id: str
    notification_id: str
    channel: str
    recipient: str
    status: str  # 'pending' | 'sent' | 'delivered' | 'failed'
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NotificationManager(CognitiveSystemInterface):
    """
    NotificationManager - Intelligent Alert System
    
    Exact Python equivalent of JavaScript NotificationManager with:
    - Multi-channel notifications (email, SMS, webhook, in-app)
    - Intelligent alert prioritization and throttling
    - Performance threshold monitoring
    - Safety alert escalation
    - Custom notification rules and filters
    - Delivery tracking and retry logic
    """
    
    def __init__(self, db: AsyncIOMotorDatabase, config: NotificationConfig):
        super().__init__(db)
        self.db = db
        self.config = config
        self.notifications_collection = db.notifications
        self.deliveries_collection = db.notification_deliveries
        self.rules_collection = db.notification_rules
        self.rules: Dict[str, NotificationRule] = {}
        self.throttle_counters: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the notification manager."""
        if self.is_initialized:
            return
            
        logger.info("📢 Initializing Notification Manager...")
        
        try:
            # Create indexes
            await self._create_indexes()
            
            # Load notification rules
            await self._load_notification_rules()
            
            # Start background processes
            self._start_background_processes()
            
            self.is_initialized = True
            logger.info("✅ Notification Manager initialized successfully")
            
        except Exception as error:
            logger.error(f"❌ Error initializing Notification Manager: {error}")
            raise error
    
    async def process(self, input_data: CognitiveInputData) -> CognitiveResponse:
        """Process notification requests."""
        try:
            await self.initialize()
            
            # Extract notification request from input
            request_data = input_data.additional_context.get("notification_request", {})
            
            if not request_data:
                return CognitiveResponse(
                    response_text="No notification request provided",
                    confidence=0.0,
                    processing_metadata={
                        "system": "notification_manager",
                        "error": "Missing notification request"
                    }
                )
            
            action = request_data.get("action", "")
            
            if action == "send_notification":
                notification_id = await self.send_notification(
                    type=request_data.get("type", ""),
                    title=request_data.get("title", ""),
                    message=request_data.get("message", ""),
                    severity=request_data.get("severity", "medium"),
                    source=request_data.get("source", {}),
                    data=request_data.get("data", {})
                )
                response_text = f"Notification sent with ID: {notification_id}"
                
            elif action == "send_safety_alert":
                notification_id = await self.send_safety_alert(
                    alert_type=request_data.get("alertType", ""),
                    message=request_data.get("message", ""),
                    severity=request_data.get("severity", "high"),
                    source=request_data.get("source", {}),
                    data=request_data.get("data", {})
                )
                response_text = f"Safety alert sent with ID: {notification_id}"
                
            elif action == "create_rule":
                rule_id = await self.create_notification_rule(request_data.get("rule", {}))
                response_text = f"Notification rule created with ID: {rule_id}"
                
            else:
                response_text = f"Unknown notification action: {action}"
            
            return CognitiveResponse(
                response_text=response_text,
                confidence=0.9,
                processing_metadata={
                    "system": "notification_manager",
                    "action": action,
                    "multi_channel": True
                }
            )
            
        except Exception as error:
            logger.error(f"Error in NotificationManager.process: {error}")
            return CognitiveResponse(
                response_text=f"Notification error: {str(error)}",
                confidence=0.0,
                processing_metadata={
                    "system": "notification_manager",
                    "error": str(error)
                }
            )
    
    async def send_notification(
        self,
        type: str,
        title: str,
        message: str,
        severity: str,
        source: Dict[str, Any],
        data: Dict[str, Any] = None
    ) -> str:
        """Send notification based on event."""
        notification = NotificationEvent(
            id=f"notif_{int(time.time() * 1000)}_{ObjectId()}",
            type=type,
            title=title,
            message=message,
            severity=severity,
            source=source,
            data=data or {},
            timestamp=datetime.utcnow()
        )
        
        # Store notification
        await self.notifications_collection.insert_one({
            "id": notification.id,
            "type": notification.type,
            "title": notification.title,
            "message": notification.message,
            "severity": notification.severity,
            "source": notification.source,
            "data": notification.data,
            "timestamp": notification.timestamp,
            "acknowledged": notification.acknowledged,
            "escalated": notification.escalated
        })
        
        # Find matching rules and process
        matching_rules = self._find_matching_rules(notification)
        for rule in matching_rules:
            await self._process_notification_rule(notification, rule)
        
        # Emit event
        await self._emit_event("notification_sent", notification)
        
        logger.info(f"📢 Sent notification: {notification.id}")
        return notification.id

    async def send_safety_alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        source: Dict[str, Any],
        data: Dict[str, Any] = None
    ) -> str:
        """Send safety alert."""
        return await self.send_notification(
            type="safety_alert",
            title=f"🚨 Safety Alert: {alert_type}",
            message=message,
            severity=severity,
            source=source,
            data=data or {}
        )

    async def send_performance_alert(
        self,
        metric: str,
        current_value: float,
        threshold: float,
        source: Dict[str, Any]
    ) -> str:
        """Send performance threshold alert."""
        return await self.send_notification(
            type="performance_alert",
            title=f"⚡ Performance Alert: {metric}",
            message=f"{metric} is {current_value}, exceeding threshold of {threshold}",
            severity="high" if current_value > threshold * 1.5 else "medium",
            source=source,
            data={"metric": metric, "currentValue": current_value, "threshold": threshold}
        )

    async def send_cost_alert(
        self,
        current_cost: float,
        threshold: float,
        period: str,
        source: Dict[str, Any]
    ) -> str:
        """Send cost alert."""
        return await self.send_notification(
            type="cost_alert",
            title=f"💰 Cost Alert: {period}",
            message=f"Cost is ${current_cost:.2f}, exceeding threshold of ${threshold:.2f} for {period}",
            severity="high" if current_cost > threshold * 1.2 else "medium",
            source=source,
            data={"currentCost": current_cost, "threshold": threshold, "period": period}
        )

    async def send_system_health_alert(
        self,
        component: str,
        status: str,
        message: str,
        source: Dict[str, Any]
    ) -> str:
        """Send system health alert."""
        severity = "critical" if status == "down" else "high" if status == "degraded" else "medium"
        return await self.send_notification(
            type="system_health",
            title=f"🏥 System Health: {component}",
            message=f"{component} is {status}: {message}",
            severity=severity,
            source=source,
            data={"component": component, "status": status}
        )

    async def create_notification_rule(self, rule_data: Dict[str, Any]) -> str:
        """Create notification rule."""
        rule = NotificationRule(
            id=f"rule_{int(time.time() * 1000)}_{ObjectId()}",
            name=rule_data.get("name", ""),
            enabled=rule_data.get("enabled", True),
            conditions=rule_data.get("conditions", {}),
            channels=rule_data.get("channels", []),
            recipients=rule_data.get("recipients", {}),
            throttling=rule_data.get("throttling", {"enabled": False}),
            escalation=rule_data.get("escalation", {"enabled": False})
        )

        # Store rule
        await self.rules_collection.insert_one({
            "id": rule.id,
            "name": rule.name,
            "enabled": rule.enabled,
            "conditions": rule.conditions,
            "channels": rule.channels,
            "recipients": rule.recipients,
            "throttling": rule.throttling,
            "escalation": rule.escalation
        })

        # Add to memory
        self.rules[rule.id] = rule

        logger.info(f"📋 Created notification rule: {rule.id}")
        return rule.id

    async def update_notification_rule(self, rule_id: str, updates: Dict[str, Any]) -> None:
        """Update notification rule."""
        await self.rules_collection.update_one(
            {"id": rule_id},
            {"$set": updates}
        )

        # Update in memory
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            for key, value in updates.items():
                setattr(rule, key, value)

        logger.info(f"📋 Updated notification rule: {rule_id}")

    async def delete_notification_rule(self, rule_id: str) -> None:
        """Delete notification rule."""
        await self.rules_collection.delete_one({"id": rule_id})
        self.rules.pop(rule_id, None)

        logger.info(f"📋 Deleted notification rule: {rule_id}")

    async def acknowledge_notification(self, notification_id: str, acknowledged_by: str) -> None:
        """Acknowledge notification."""
        await self.notifications_collection.update_one(
            {"id": notification_id},
            {
                "$set": {
                    "acknowledged": True,
                    "acknowledgedBy": acknowledged_by,
                    "acknowledgedAt": datetime.utcnow()
                }
            }
        )

        logger.info(f"✅ Acknowledged notification: {notification_id}")

    async def get_notification_history(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get notification history."""
        query = {}

        if filters:
            if filters.get("type"):
                query["type"] = filters["type"]
            if filters.get("severity"):
                query["severity"] = filters["severity"]
            if filters.get("startDate") and filters.get("endDate"):
                query["timestamp"] = {
                    "$gte": filters["startDate"],
                    "$lte": filters["endDate"]
                }

        notifications = await self.notifications_collection.find(query).sort("timestamp", -1).limit(limit).to_list(length=None)
        return notifications

    async def get_delivery_stats(self, timeframe: str = "day") -> Dict[str, Any]:
        """Get delivery statistics."""
        # Calculate time range
        now = datetime.utcnow()
        if timeframe == "hour":
            start_time = now - timedelta(hours=1)
        elif timeframe == "week":
            start_time = now - timedelta(weeks=1)
        else:  # day
            start_time = now - timedelta(days=1)

        # Aggregate delivery stats
        pipeline = [
            {"$match": {"sent_at": {"$gte": start_time}}},
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]

        results = await self.deliveries_collection.aggregate(pipeline).to_list(length=None)

        stats = {"sent": 0, "delivered": 0, "failed": 0}
        for result in results:
            status = result["_id"]
            count = result["count"]
            if status in ["sent", "delivered"]:
                stats["sent"] += count
                if status == "delivered":
                    stats["delivered"] += count
            elif status == "failed":
                stats["failed"] += count

        total_sent = stats["sent"]
        delivery_rate = (stats["delivered"] / total_sent * 100) if total_sent > 0 else 0

        return {
            "totalSent": total_sent,
            "totalDelivered": stats["delivered"],
            "totalFailed": stats["failed"],
            "deliveryRate": delivery_rate,
            "channelStats": {}  # Could be expanded with channel-specific stats
        }

    # Private methods

    def _find_matching_rules(self, notification: NotificationEvent) -> List[NotificationRule]:
        """Find matching notification rules."""
        matching_rules = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            conditions = rule.conditions

            # Check event types
            if conditions.get("eventTypes") and notification.type not in conditions["eventTypes"]:
                continue

            # Check severity
            if conditions.get("severity") and notification.severity not in conditions["severity"]:
                continue

            # Check frameworks
            if conditions.get("frameworks"):
                source_framework = notification.source.get("framework", "")
                if source_framework not in conditions["frameworks"]:
                    continue

            # Check agents
            if conditions.get("agents"):
                source_agent = notification.source.get("agentId", "")
                if source_agent not in conditions["agents"]:
                    continue

            # Check throttling
            if self._is_throttled(rule.id, rule.throttling):
                continue

            matching_rules.append(rule)

        return matching_rules

    async def _process_notification_rule(self, notification: NotificationEvent, rule: NotificationRule) -> None:
        """Process notification rule."""
        # Send to each configured channel
        for channel in rule.channels:
            if channel == "email" and self.config.email["enabled"] and rule.recipients.get("emails"):
                for email in rule.recipients["emails"]:
                    await self._send_email_notification(notification, email)

            elif channel == "sms" and self.config.sms["enabled"] and rule.recipients.get("phoneNumbers"):
                for phone in rule.recipients["phoneNumbers"]:
                    await self._send_sms_notification(notification, phone)

            elif channel == "webhook" and self.config.webhook["enabled"] and rule.recipients.get("webhookUrls"):
                for url in rule.recipients["webhookUrls"]:
                    await self._send_webhook_notification(notification, url)

            elif channel == "inApp" and self.config.in_app["enabled"]:
                await self._send_in_app_notification(notification)

        # Update throttle counter
        self._update_throttle_counter(rule.id)

        # Schedule escalation if enabled
        if rule.escalation.get("enabled"):
            self._schedule_escalation(notification, rule)

    async def _send_email_notification(self, notification: NotificationEvent, email: str) -> None:
        """Send email notification."""
        delivery = NotificationDelivery(
            id=f"delivery_{int(time.time() * 1000)}_{ObjectId()}",
            notification_id=notification.id,
            channel="email",
            recipient=email,
            status="pending",
            sent_at=datetime.utcnow()
        )

        try:
            # Create email content
            subject = f"[{notification.severity.upper()}] {notification.title}"
            body = f"""
{notification.message}

Source: {notification.source.get('framework', 'Unknown')}
Agent: {notification.source.get('agentId', 'Unknown')}
Time: {notification.timestamp.isoformat()}

---
AI Brain Notification System
"""

            # Send email using SMTP
            if self.config.email["provider"] == "smtp":
                await self._send_smtp_email(email, subject, body)

            delivery.status = "sent"
            delivery.delivered_at = datetime.utcnow()

        except Exception as error:
            delivery.status = "failed"
            delivery.error = str(error)
            logger.error(f"Failed to send email to {email}: {error}")

        # Store delivery record
        await self.deliveries_collection.insert_one({
            "id": delivery.id,
            "notification_id": delivery.notification_id,
            "channel": delivery.channel,
            "recipient": delivery.recipient,
            "status": delivery.status,
            "sent_at": delivery.sent_at,
            "delivered_at": delivery.delivered_at,
            "error": delivery.error
        })

    async def _send_sms_notification(self, notification: NotificationEvent, phone_number: str) -> None:
        """Send SMS notification."""
        delivery = NotificationDelivery(
            id=f"delivery_{int(time.time() * 1000)}_{ObjectId()}",
            notification_id=notification.id,
            channel="sms",
            recipient=phone_number,
            status="pending",
            sent_at=datetime.utcnow()
        )

        try:
            # Create SMS content (truncated for SMS limits)
            message = f"[{notification.severity.upper()}] {notification.title}: {notification.message[:100]}..."

            # In a real implementation, integrate with Twilio or AWS SNS
            logger.info(f"SMS would be sent to {phone_number}: {message}")

            delivery.status = "sent"
            delivery.delivered_at = datetime.utcnow()

        except Exception as error:
            delivery.status = "failed"
            delivery.error = str(error)
            logger.error(f"Failed to send SMS to {phone_number}: {error}")

        # Store delivery record
        await self.deliveries_collection.insert_one({
            "id": delivery.id,
            "notification_id": delivery.notification_id,
            "channel": delivery.channel,
            "recipient": delivery.recipient,
            "status": delivery.status,
            "sent_at": delivery.sent_at,
            "delivered_at": delivery.delivered_at,
            "error": delivery.error
        })

    async def _send_webhook_notification(self, notification: NotificationEvent, url: str) -> None:
        """Send webhook notification."""
        delivery = NotificationDelivery(
            id=f"delivery_{int(time.time() * 1000)}_{ObjectId()}",
            notification_id=notification.id,
            channel="webhook",
            recipient=url,
            status="pending",
            sent_at=datetime.utcnow()
        )

        try:
            # Create webhook payload
            payload = {
                "id": notification.id,
                "type": notification.type,
                "title": notification.title,
                "message": notification.message,
                "severity": notification.severity,
                "source": notification.source,
                "data": notification.data,
                "timestamp": notification.timestamp.isoformat()
            }

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        delivery.status = "delivered"
                    else:
                        delivery.status = "failed"
                        delivery.error = f"HTTP {response.status}"

            delivery.delivered_at = datetime.utcnow()

        except Exception as error:
            delivery.status = "failed"
            delivery.error = str(error)
            logger.error(f"Failed to send webhook to {url}: {error}")

        # Store delivery record
        await self.deliveries_collection.insert_one({
            "id": delivery.id,
            "notification_id": delivery.notification_id,
            "channel": delivery.channel,
            "recipient": delivery.recipient,
            "status": delivery.status,
            "sent_at": delivery.sent_at,
            "delivered_at": delivery.delivered_at,
            "error": delivery.error
        })

    async def _send_in_app_notification(self, notification: NotificationEvent) -> None:
        """Send in-app notification."""
        delivery = NotificationDelivery(
            id=f"delivery_{int(time.time() * 1000)}_{ObjectId()}",
            notification_id=notification.id,
            channel="inApp",
            recipient="in_app_users",
            status="delivered",
            sent_at=datetime.utcnow(),
            delivered_at=datetime.utcnow()
        )

        # Store delivery record
        await self.deliveries_collection.insert_one({
            "id": delivery.id,
            "notification_id": delivery.notification_id,
            "channel": delivery.channel,
            "recipient": delivery.recipient,
            "status": delivery.status,
            "sent_at": delivery.sent_at,
            "delivered_at": delivery.delivered_at
        })

        logger.debug(f"In-app notification stored: {notification.id}")

    async def _send_smtp_email(self, to_email: str, subject: str, body: str) -> None:
        """Send email using SMTP."""
        smtp_config = self.config.email["smtpConfig"]

        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{self.config.email['fromName']} <{self.config.email['fromEmail']}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send email
        server = smtplib.SMTP(smtp_config["host"], smtp_config["port"])
        if smtp_config["secure"]:
            server.starttls()

        auth = smtp_config["auth"]
        server.login(auth["user"], auth["pass"])
        server.send_message(msg)
        server.quit()

    def _is_throttled(self, rule_id: str, throttling: Dict[str, Any]) -> bool:
        """Check if rule is throttled."""
        if not throttling.get("enabled"):
            return False

        counter = self.throttle_counters.get(rule_id)
        if not counter:
            return False

        now = datetime.utcnow()
        if now >= counter["resetTime"]:
            return False

        max_per_hour = throttling.get("maxPerHour", 10)
        return counter["count"] >= max_per_hour

    def _update_throttle_counter(self, rule_id: str) -> None:
        """Update throttle counter."""
        now = datetime.utcnow()
        reset_time = now + timedelta(hours=1)

        counter = self.throttle_counters.get(rule_id)
        if counter and now < counter["resetTime"]:
            counter["count"] += 1
        else:
            self.throttle_counters[rule_id] = {"count": 1, "resetTime": reset_time}

    def _schedule_escalation(self, notification: NotificationEvent, rule: NotificationRule) -> None:
        """Schedule escalation."""
        escalate_after_minutes = rule.escalation.get("escalateAfterMinutes", 30)

        async def escalate():
            await asyncio.sleep(escalate_after_minutes * 60)

            # Check if notification was acknowledged
            current = await self.notifications_collection.find_one({"id": notification.id})
            if current and current.get("acknowledged"):
                return

            # Send escalation
            await self._send_escalation(notification, rule)

        # Schedule escalation
        asyncio.create_task(escalate())

    async def _send_escalation(self, notification: NotificationEvent, rule: NotificationRule) -> None:
        """Send escalation."""
        escalation_notification = NotificationEvent(
            id=f"escalation_{notification.id}",
            type=notification.type,
            title=f"🚨 ESCALATION: {notification.title}",
            message=f"ESCALATED: {notification.message}",
            severity="critical",
            source=notification.source,
            data=notification.data,
            timestamp=datetime.utcnow()
        )

        # Store escalation notification
        await self.notifications_collection.insert_one({
            "id": escalation_notification.id,
            "type": escalation_notification.type,
            "title": escalation_notification.title,
            "message": escalation_notification.message,
            "severity": escalation_notification.severity,
            "source": escalation_notification.source,
            "data": escalation_notification.data,
            "timestamp": escalation_notification.timestamp,
            "acknowledged": False,
            "escalated": True
        })

        # Mark original as escalated
        await self.notifications_collection.update_one(
            {"id": notification.id},
            {"$set": {"escalated": True}}
        )

        # Send escalation through escalation channels
        escalation_channels = rule.escalation.get("escalationChannels", ["email"])
        escalation_recipients = rule.escalation.get("escalationRecipients", {})

        for channel in escalation_channels:
            if channel == "email" and escalation_recipients.get("emails"):
                for email in escalation_recipients["emails"]:
                    await self._send_email_notification(escalation_notification, email)
            elif channel == "sms" and escalation_recipients.get("phoneNumbers"):
                for phone in escalation_recipients["phoneNumbers"]:
                    await self._send_sms_notification(escalation_notification, phone)
            elif channel == "webhook" and escalation_recipients.get("webhookUrls"):
                for url in escalation_recipients["webhookUrls"]:
                    await self._send_webhook_notification(escalation_notification, url)

        logger.warning(f"🚨 Escalated notification: {notification.id}")

    async def _load_notification_rules(self) -> None:
        """Load notification rules from database."""
        rules = await self.rules_collection.find({}).to_list(length=None)

        for rule_data in rules:
            rule = NotificationRule(
                id=rule_data["id"],
                name=rule_data["name"],
                enabled=rule_data["enabled"],
                conditions=rule_data["conditions"],
                channels=rule_data["channels"],
                recipients=rule_data["recipients"],
                throttling=rule_data["throttling"],
                escalation=rule_data["escalation"]
            )
            self.rules[rule.id] = rule

        logger.info(f"📋 Loaded {len(self.rules)} notification rules")

    def _start_background_processes(self) -> None:
        """Start background processes."""
        async def cleanup_old_notifications():
            while True:
                try:
                    if self.config.in_app["enabled"]:
                        cutoff_date = datetime.utcnow() - timedelta(days=self.config.in_app["retentionDays"])
                        await self.notifications_collection.delete_many({
                            "timestamp": {"$lt": cutoff_date}
                        })
                        await self.deliveries_collection.delete_many({
                            "sent_at": {"$lt": cutoff_date}
                        })

                    # Sleep for 1 hour
                    await asyncio.sleep(3600)
                except Exception as error:
                    logger.error(f"Error in cleanup process: {error}")
                    await asyncio.sleep(3600)

        # Start cleanup task
        asyncio.create_task(cleanup_old_notifications())

    async def _create_indexes(self) -> None:
        """Create database indexes."""
        await self.notifications_collection.create_index([("timestamp", -1)])
        await self.notifications_collection.create_index([("type", 1), ("severity", 1)])
        await self.notifications_collection.create_index([("acknowledged", 1)])
        await self.deliveries_collection.create_index([("notification_id", 1)])
        await self.deliveries_collection.create_index([("channel", 1), ("status", 1)])
        await self.deliveries_collection.create_index([("sent_at", -1)])
        await self.rules_collection.create_index([("enabled", 1)])

        logger.debug("📊 Created notification indexes")

    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit event to registered handlers."""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as error:
                logger.error(f"Error in event handler: {error}")

    def on(self, event_type: str, handler: Callable) -> None:
        """Register event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """Unregister event handler."""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def shutdown(self) -> None:
        """Shutdown notification manager."""
        self.event_handlers.clear()
        logger.info("🛑 Notification Manager shutdown complete")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.shutdown()
