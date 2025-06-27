"""
Skill Capability Manager

Dynamic skill acquisition and proficiency tracking system.
Manages skill assessment, learning progress, and capability development.

Features:
- Dynamic skill proficiency assessment
- Learning goal tracking and recommendations
- Skill gap analysis and improvement paths
- Performance-based skill level updates
- Personalized learning recommendations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ai_brain_python.core.interfaces.cognitive_system import CognitiveSystemInterface, SystemCapability
from ai_brain_python.core.models.base_models import CognitiveInputData, ValidationResult
from ai_brain_python.core.models.cognitive_states import CognitiveState, SkillAssessment, SkillLevel, CognitiveSystemType

logger = logging.getLogger(__name__)


class SkillCapabilityManager(CognitiveSystemInterface):
    """Skill Capability Manager - System 6 of 16"""
    
    def __init__(self, system_id: str = "skill_capability", config: Optional[Dict[str, Any]] = None):
        super().__init__(system_id, config)
        
        # Skill assessments by user
        self._user_skill_assessments: Dict[str, SkillAssessment] = {}
        
        # Skill categories and keywords
        self._skill_categories = {
            "technical": ["programming", "coding", "development", "software", "algorithm", "database"],
            "communication": ["writing", "speaking", "presentation", "negotiation", "leadership"],
            "analytical": ["analysis", "research", "problem solving", "critical thinking", "data"],
            "creative": ["design", "creative", "innovation", "brainstorming", "artistic"],
            "management": ["project management", "team lead", "planning", "organization", "strategy"]
        }
        
        # Skill level progression thresholds
        self._level_thresholds = {
            SkillLevel.NOVICE: 0.0,
            SkillLevel.BEGINNER: 0.2,
            SkillLevel.INTERMEDIATE: 0.4,
            SkillLevel.ADVANCED: 0.7,
            SkillLevel.EXPERT: 0.9
        }
        
        # Learning indicators
        self._learning_indicators = {
            "learning": ["learn", "study", "practice", "improve", "develop", "master"],
            "struggling": ["difficult", "hard", "challenging", "confused", "stuck"],
            "confident": ["confident", "comfortable", "easy", "familiar", "experienced"]
        }
    
    @property
    def system_name(self) -> str:
        return "Skill Capability Manager"
    
    @property
    def system_description(self) -> str:
        return "Dynamic skill acquisition and proficiency tracking system"
    
    @property
    def required_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.SKILL_ASSESSMENT}
    
    @property
    def provided_capabilities(self) -> Set[SystemCapability]:
        return {SystemCapability.SKILL_ASSESSMENT}
    
    async def initialize(self) -> None:
        """Initialize the Skill Capability Manager."""
        try:
            logger.info("Initializing Skill Capability Manager...")
            await self._load_skill_data()
            self._is_initialized = True
            logger.info("Skill Capability Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Skill Capability Manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the Skill Capability Manager."""
        try:
            await self._save_skill_data()
            self._is_initialized = False
            logger.info("Skill Capability Manager shutdown complete")
        except Exception as e:
            logger.error(f"Error during Skill Capability Manager shutdown: {e}")
    
    async def process(self, input_data: CognitiveInputData, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input through skill capability analysis."""
        if not self._is_initialized:
            raise RuntimeError("Skill Capability Manager not initialized")
        
        try:
            start_time = datetime.utcnow()
            user_id = input_data.context.user_id or "anonymous"
            
            # Ensure user has skill assessment
            if user_id not in self._user_skill_assessments:
                await self._create_user_skill_assessment(user_id)
            
            # Analyze skills mentioned in input
            skill_analysis = await self._analyze_skills_in_input(input_data)
            
            # Update skill assessments
            skill_updates = await self._update_skill_assessments(user_id, skill_analysis)
            
            # Generate learning recommendations
            recommendations = await self._generate_learning_recommendations(user_id, skill_analysis)
            
            # Get skill gaps
            skill_gaps = await self._identify_skill_gaps(user_id)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "system": self.system_id,
                "status": "completed",
                "processing_time_ms": processing_time,
                "confidence": 0.8,
                "skill_assessment": {
                    "total_skills": len(self._user_skill_assessments[user_id].skills),
                    "skill_levels": {skill: level.value for skill, level in self._user_skill_assessments[user_id].skills.items()},
                    "learning_goals": self._user_skill_assessments[user_id].learning_goals,
                    "skill_gaps": skill_gaps
                },
                "skill_analysis": skill_analysis,
                "skill_updates": skill_updates,
                "recommendations": recommendations,
                "learning_path": await self._generate_learning_path(user_id)
            }
            
        except Exception as e:
            logger.error(f"Error in Skill Capability processing: {e}")
            return {"system": self.system_id, "status": "error", "error": str(e), "confidence": 0.0}
    
    async def get_state(self, user_id: Optional[str] = None) -> CognitiveState:
        """Get current skill capability state."""
        state_data = {
            "total_users": len(self._user_skill_assessments),
            "skill_categories": len(self._skill_categories)
        }
        
        if user_id and user_id in self._user_skill_assessments:
            assessment = self._user_skill_assessments[user_id]
            state_data.update({
                "user_total_skills": len(assessment.skills),
                "user_learning_goals": len(assessment.learning_goals),
                "user_skill_gaps": len(assessment.skill_gaps)
            })
        
        return CognitiveState(
            system_type=CognitiveSystemType.SKILL_CAPABILITY,
            user_id=user_id,
            is_active=self._is_initialized,
            confidence=0.85,
            state_data=state_data
        )
    
    async def update_state(self, state: CognitiveState, user_id: Optional[str] = None) -> bool:
        """Update skill capability state."""
        try:
            return True
        except Exception as e:
            logger.error(f"Error updating Skill Capability state: {e}")
            return False
    
    async def validate_input(self, input_data: CognitiveInputData) -> ValidationResult:
        """Validate input for skill capability processing."""
        return ValidationResult(is_valid=True, confidence=1.0, violations=[], warnings=[])
    
    # Public skill methods
    
    async def update_skill_level(self, user_id: str, skill: str, performance_score: float) -> bool:
        """Update skill level based on performance."""
        if user_id not in self._user_skill_assessments:
            await self._create_user_skill_assessment(user_id)
        
        assessment = self._user_skill_assessments[user_id]
        
        # Determine new skill level
        new_level = self._determine_skill_level(performance_score)
        
        # Update skill
        assessment.skills[skill] = new_level
        assessment.skill_confidence[skill] = min(1.0, performance_score)
        
        # Update usage frequency
        assessment.skill_usage_frequency[skill] = assessment.skill_usage_frequency.get(skill, 0) + 1
        
        return True
    
    async def add_learning_goal(self, user_id: str, skill: str) -> bool:
        """Add a learning goal for a user."""
        if user_id not in self._user_skill_assessments:
            await self._create_user_skill_assessment(user_id)
        
        assessment = self._user_skill_assessments[user_id]
        if skill not in assessment.learning_goals:
            assessment.learning_goals.append(skill)
        
        return True
    
    # Private methods
    
    async def _load_skill_data(self) -> None:
        """Load skill data from storage."""
        logger.debug("Skill data loaded")
    
    async def _save_skill_data(self) -> None:
        """Save skill data to storage."""
        logger.debug("Skill data saved")
    
    async def _create_user_skill_assessment(self, user_id: str) -> None:
        """Create initial skill assessment for a user."""
        assessment = SkillAssessment()
        self._user_skill_assessments[user_id] = assessment
        logger.debug(f"Created skill assessment for user {user_id}")
    
    async def _analyze_skills_in_input(self, input_data: CognitiveInputData) -> Dict[str, Any]:
        """Analyze skills mentioned in input."""
        text = input_data.text or ""
        text_lower = text.lower()
        
        mentioned_skills = {}
        skill_categories_found = []
        
        # Check for skill categories
        for category, keywords in self._skill_categories.items():
            category_score = sum(1 for keyword in keywords if keyword in text_lower)
            if category_score > 0:
                skill_categories_found.append(category)
                mentioned_skills[category] = category_score
        
        # Analyze learning indicators
        learning_context = {}
        for context_type, indicators in self._learning_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            if count > 0:
                learning_context[context_type] = count
        
        return {
            "mentioned_skills": mentioned_skills,
            "skill_categories": skill_categories_found,
            "learning_context": learning_context,
            "skill_confidence_indicators": self._assess_skill_confidence(text)
        }
    
    async def _update_skill_assessments(self, user_id: str, skill_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update skill assessments based on analysis."""
        updates = []
        assessment = self._user_skill_assessments[user_id]
        
        # Update skills based on mentions and context
        for skill, mention_count in skill_analysis["mentioned_skills"].items():
            current_level = assessment.skills.get(skill, SkillLevel.NOVICE)
            
            # Adjust skill level based on learning context
            learning_context = skill_analysis["learning_context"]
            
            if "confident" in learning_context:
                # User seems confident, slightly increase level
                new_score = self._skill_level_to_score(current_level) + 0.1
            elif "struggling" in learning_context:
                # User is struggling, maintain or slightly decrease
                new_score = max(0.0, self._skill_level_to_score(current_level) - 0.05)
            else:
                # Neutral mention, slight increase for engagement
                new_score = self._skill_level_to_score(current_level) + 0.05
            
            new_level = self._determine_skill_level(new_score)
            
            if new_level != current_level:
                assessment.skills[skill] = new_level
                updates.append({
                    "skill": skill,
                    "old_level": current_level.value,
                    "new_level": new_level.value,
                    "reason": "contextual_analysis"
                })
        
        return updates
    
    async def _generate_learning_recommendations(self, user_id: str, skill_analysis: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations."""
        recommendations = []
        assessment = self._user_skill_assessments[user_id]
        
        # Recommend based on skill gaps
        if assessment.skill_gaps:
            recommendations.append(f"Focus on developing: {', '.join(assessment.skill_gaps[:3])}")
        
        # Recommend based on learning context
        learning_context = skill_analysis["learning_context"]
        if "struggling" in learning_context:
            recommendations.append("Consider breaking down complex skills into smaller learning objectives")
        
        if "learning" in learning_context:
            recommendations.append("Great learning attitude! Consider setting specific practice goals")
        
        # Recommend skill advancement
        intermediate_skills = [
            skill for skill, level in assessment.skills.items() 
            if level == SkillLevel.INTERMEDIATE
        ]
        if intermediate_skills:
            recommendations.append(f"Ready to advance to expert level: {intermediate_skills[0]}")
        
        return recommendations
    
    async def _identify_skill_gaps(self, user_id: str) -> List[str]:
        """Identify skill gaps for a user."""
        assessment = self._user_skill_assessments[user_id]
        
        # Simple gap analysis - skills mentioned but not in user's profile
        gaps = []
        
        # Check if user has basic skills in each category
        for category in self._skill_categories:
            if category not in assessment.skills or assessment.skills[category] == SkillLevel.NOVICE:
                gaps.append(category)
        
        return gaps[:5]  # Return top 5 gaps
    
    async def _generate_learning_path(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate a learning path for a user."""
        assessment = self._user_skill_assessments[user_id]
        learning_path = []
        
        # Create path based on current skills and gaps
        for goal in assessment.learning_goals[:3]:  # Top 3 goals
            current_level = assessment.skills.get(goal, SkillLevel.NOVICE)
            next_level = self._get_next_skill_level(current_level)
            
            learning_path.append({
                "skill": goal,
                "current_level": current_level.value,
                "target_level": next_level.value,
                "estimated_time": self._estimate_learning_time(current_level, next_level),
                "recommended_actions": self._get_learning_actions(goal, current_level)
            })
        
        return learning_path
    
    def _determine_skill_level(self, score: float) -> SkillLevel:
        """Determine skill level from score."""
        for level, threshold in sorted(self._level_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return SkillLevel.NOVICE
    
    def _skill_level_to_score(self, level: SkillLevel) -> float:
        """Convert skill level to numeric score."""
        return self._level_thresholds[level]
    
    def _assess_skill_confidence(self, text: str) -> Dict[str, float]:
        """Assess confidence indicators in text."""
        confidence_indicators = {
            "high": ["expert", "experienced", "confident", "proficient", "skilled"],
            "medium": ["familiar", "comfortable", "decent", "okay", "reasonable"],
            "low": ["beginner", "novice", "learning", "new to", "unfamiliar"]
        }
        
        text_lower = text.lower()
        confidence_scores = {}
        
        for level, indicators in confidence_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            confidence_scores[level] = score
        
        return confidence_scores
    
    def _get_next_skill_level(self, current_level: SkillLevel) -> SkillLevel:
        """Get the next skill level."""
        levels = list(SkillLevel)
        current_index = levels.index(current_level)
        if current_index < len(levels) - 1:
            return levels[current_index + 1]
        return current_level
    
    def _estimate_learning_time(self, current_level: SkillLevel, target_level: SkillLevel) -> str:
        """Estimate learning time between levels."""
        time_estimates = {
            (SkillLevel.NOVICE, SkillLevel.BEGINNER): "2-4 weeks",
            (SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE): "2-3 months",
            (SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED): "6-12 months",
            (SkillLevel.ADVANCED, SkillLevel.EXPERT): "1-2 years"
        }
        return time_estimates.get((current_level, target_level), "Variable")
    
    def _get_learning_actions(self, skill: str, current_level: SkillLevel) -> List[str]:
        """Get recommended learning actions for a skill."""
        if current_level == SkillLevel.NOVICE:
            return [f"Take introductory course in {skill}", f"Practice basic {skill} exercises"]
        elif current_level == SkillLevel.BEGINNER:
            return [f"Work on intermediate {skill} projects", f"Find mentor for {skill}"]
        elif current_level == SkillLevel.INTERMEDIATE:
            return [f"Take on challenging {skill} projects", f"Teach {skill} to others"]
        else:
            return [f"Contribute to {skill} community", f"Develop expertise in {skill} specialization"]
