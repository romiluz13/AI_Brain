"""
AI Brain Python Cognitive Systems

This module contains implementations of all 16 cognitive systems:

Core Systems (T006):
1. Emotional Intelligence Engine - Emotion detection, empathy modeling, mood tracking
2. Goal Hierarchy Manager - Hierarchical goal planning and achievement tracking
3. Confidence Tracking Engine - Real-time uncertainty assessment and reliability tracking
4. Attention Management System - Dynamic attention allocation and focus control
5. Cultural Knowledge Engine - Cross-cultural intelligence and adaptation
6. Skill Capability Manager - Dynamic skill acquisition and proficiency tracking
7. Communication Protocol Manager - Multi-protocol communication management
8. Temporal Planning Engine - Time-aware task management and scheduling

Enhanced Systems (T007):
9. Semantic Memory Engine - Perfect recall with MongoDB vector search
10. Safety Guardrails Engine - Multi-layer safety and compliance systems
11. Self-Improvement Engine - Continuous learning and optimization
12. Real-time Monitoring Engine - Live metrics and performance analytics
13. Advanced Tool Interface - Tool recovery and validation systems
14. Workflow Orchestration Engine - Intelligent routing and parallel processing
15. Multi-Modal Processing Engine - Image/audio/video processing
16. Human Feedback Integration Engine - Approval workflows and learning

Each system implements the CognitiveSystemInterface for consistency.
"""

# Core Systems (T006)
from ai_brain_python.core.cognitive_systems.emotional_intelligence import EmotionalIntelligenceEngine
from ai_brain_python.core.cognitive_systems.goal_hierarchy import GoalHierarchyManager
from ai_brain_python.core.cognitive_systems.confidence_tracking import ConfidenceTrackingEngine
from ai_brain_python.core.cognitive_systems.attention_management import AttentionManagementSystem
from ai_brain_python.core.cognitive_systems.cultural_knowledge import CulturalKnowledgeEngine
from ai_brain_python.core.cognitive_systems.skill_capability import SkillCapabilityManager
from ai_brain_python.core.cognitive_systems.communication_protocol import CommunicationProtocolManager
from ai_brain_python.core.cognitive_systems.temporal_planning import TemporalPlanningEngine

# Enhanced Systems (T007) - COMPLETED
from ai_brain_python.core.cognitive_systems.semantic_memory import SemanticMemoryEngine
from ai_brain_python.core.cognitive_systems.safety_guardrails import SafetyGuardrailsEngine
from ai_brain_python.core.cognitive_systems.self_improvement import SelfImprovementEngine
from ai_brain_python.core.cognitive_systems.monitoring import MonitoringEngine
from ai_brain_python.core.cognitive_systems.tool_interface import AdvancedToolInterface
from ai_brain_python.core.cognitive_systems.workflow_orchestration import WorkflowOrchestrationEngine
from ai_brain_python.core.cognitive_systems.multimodal_processing import MultiModalProcessingEngine
from ai_brain_python.core.cognitive_systems.human_feedback import HumanFeedbackIntegrationEngine

__all__ = [
    # Core Systems (T006)
    "EmotionalIntelligenceEngine",
    "GoalHierarchyManager",
    "ConfidenceTrackingEngine",
    "AttentionManagementSystem",
    "CulturalKnowledgeEngine",
    "SkillCapabilityManager",
    "CommunicationProtocolManager",
    "TemporalPlanningEngine",

    # Enhanced Systems (T007) - COMPLETED
    "SemanticMemoryEngine",
    "SafetyGuardrailsEngine",
    "SelfImprovementEngine",
    "MonitoringEngine",
    "AdvancedToolInterface",
    "WorkflowOrchestrationEngine",
    "MultiModalProcessingEngine",
    "HumanFeedbackIntegrationEngine",
]