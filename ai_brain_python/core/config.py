"""
Configuration Management System for Universal AI Brain Python

Exact Python equivalent of JavaScript configuration system with:
- Same configuration structure and field names
- Identical default values and validation
- Same environment variable handling
- Matching configuration merging logic
"""

import os
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field, validator


# ============================================================================
# CONFIGURATION MODELS - Exact TypeScript Equivalents
# ============================================================================

class MongoDBConfig(BaseModel):
    """Exact equivalent of JavaScript mongodb config."""
    connection_string: str = Field(..., alias='connectionString')
    database_name: Optional[str] = Field(None, alias='databaseName')
    collections: Optional[Dict[str, str]] = None
    
    class Config:
        allow_population_by_field_name = True


class IntelligenceConfig(BaseModel):
    """Exact equivalent of JavaScript intelligence config."""
    embedding_model: Optional[str] = Field(None, alias='embeddingModel')
    vector_dimensions: Optional[int] = Field(None, alias='vectorDimensions')
    similarity_threshold: Optional[float] = Field(None, alias='similarityThreshold')
    max_context_length: Optional[int] = Field(None, alias='maxContextLength')
    
    # Hybrid Search Configuration - exact same as JavaScript
    enable_hybrid_search: Optional[bool] = Field(None, alias='enableHybridSearch')
    hybrid_search_vector_weight: Optional[float] = Field(None, alias='hybridSearchVectorWeight')
    hybrid_search_text_weight: Optional[float] = Field(None, alias='hybridSearchTextWeight')
    hybrid_search_fallback_to_vector: Optional[bool] = Field(None, alias='hybridSearchFallbackToVector')
    
    class Config:
        allow_population_by_field_name = True


class SafetyConfig(BaseModel):
    """Exact equivalent of JavaScript safety config."""
    enable_content_filtering: Optional[bool] = Field(None, alias='enableContentFiltering')
    enable_pii_detection: Optional[bool] = Field(None, alias='enablePIIDetection')
    enable_hallucination_detection: Optional[bool] = Field(None, alias='enableHallucinationDetection')
    enable_compliance_logging: Optional[bool] = Field(None, alias='enableComplianceLogging')
    safety_level: Optional[Literal['strict', 'moderate', 'permissive']] = Field(None, alias='safetyLevel')
    
    class Config:
        allow_population_by_field_name = True


class MonitoringConfig(BaseModel):
    """Exact equivalent of JavaScript monitoring config."""
    enable_real_time_monitoring: Optional[bool] = Field(None, alias='enableRealTimeMonitoring')
    enable_performance_tracking: Optional[bool] = Field(None, alias='enablePerformanceTracking')
    enable_cost_tracking: Optional[bool] = Field(None, alias='enableCostTracking')
    enable_error_tracking: Optional[bool] = Field(None, alias='enableErrorTracking')
    metrics_retention_days: Optional[int] = Field(None, alias='metricsRetentionDays')
    alerting_enabled: Optional[bool] = Field(None, alias='alertingEnabled')
    dashboard_refresh_interval: Optional[int] = Field(None, alias='dashboardRefreshInterval')
    
    class Config:
        allow_population_by_field_name = True


class SelfImprovementConfig(BaseModel):
    """Exact equivalent of JavaScript selfImprovement config."""
    enable_automatic_optimization: Optional[bool] = Field(None, alias='enableAutomaticOptimization')
    learning_rate: Optional[float] = Field(None, alias='learningRate')
    optimization_interval: Optional[int] = Field(None, alias='optimizationInterval')
    feedback_loop_enabled: Optional[bool] = Field(None, alias='feedbackLoopEnabled')
    
    class Config:
        allow_population_by_field_name = True


class APIProviderConfig(BaseModel):
    """API provider configuration."""
    api_key: str = Field(..., alias='apiKey')
    base_url: Optional[str] = Field(None, alias='baseURL')
    
    class Config:
        allow_population_by_field_name = True


class APIsConfig(BaseModel):
    """Exact equivalent of JavaScript apis config."""
    openai: Optional[APIProviderConfig] = None
    voyage: Optional[APIProviderConfig] = None
    
    class Config:
        allow_population_by_field_name = True


class SimpleAIBrainConfig(BaseModel):
    """Exact equivalent of JavaScript SimpleAIBrainConfig interface."""
    mongo_uri: Optional[str] = Field(None, alias='mongoUri')
    database_name: Optional[str] = Field(None, alias='databaseName')
    api_key: Optional[str] = Field(None, alias='apiKey')
    provider: Optional[Literal['voyage', 'openai']] = None
    mode: Optional[Literal['demo', 'basic', 'production']] = None
    
    class Config:
        allow_population_by_field_name = True


class UniversalAIBrainConfig(BaseModel):
    """
    Exact equivalent of JavaScript UniversalAIBrainConfig interface.
    
    Supports both simple and advanced configuration patterns.
    """
    # Advanced configuration
    mongodb: Optional[MongoDBConfig] = None
    intelligence: Optional[IntelligenceConfig] = None
    safety: Optional[SafetyConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    self_improvement: Optional[SelfImprovementConfig] = Field(None, alias='selfImprovement')
    apis: Optional[APIsConfig] = None
    
    # Simple configuration (backward compatibility)
    mongo_uri: Optional[str] = Field(None, alias='mongoUri')
    database_name: Optional[str] = Field(None, alias='databaseName')
    api_key: Optional[str] = Field(None, alias='apiKey')
    provider: Optional[Literal['voyage', 'openai']] = None
    mode: Optional[Literal['demo', 'basic', 'production']] = None
    
    class Config:
        allow_population_by_field_name = True


# ============================================================================
# DEFAULT CONFIGURATION - Exact JavaScript Equivalent
# ============================================================================

DEFAULT_CONFIG = {
    "intelligence": {
        "embeddingModel": "voyage-large-2-instruct",
        "vectorDimensions": 1024,
        "similarityThreshold": 0.7,
        "maxContextLength": 4000,
        # Hybrid Search as DEFAULT - MongoDB's most powerful capability
        "enableHybridSearch": True,
        "hybridSearchVectorWeight": 0.7,
        "hybridSearchTextWeight": 0.3,
        "hybridSearchFallbackToVector": True
    },
    "safety": {
        "enableContentFiltering": True,
        "enablePIIDetection": True,
        "enableHallucinationDetection": True,
        "enableComplianceLogging": True,
        "safetyLevel": "moderate"
    },
    "monitoring": {
        "enableRealTimeMonitoring": True,
        "enablePerformanceTracking": True,
        "enableCostTracking": True,
        "enableErrorTracking": True,
        "metricsRetentionDays": 30,
        "alertingEnabled": False,
        "dashboardRefreshInterval": 5000
    },
    "collections": {
        "tracing": "agent_traces",
        "memory": "agent_memory",
        "context": "agent_context",
        "metrics": "agent_metrics",
        "audit": "agent_safety_logs"
    }
}


# ============================================================================
# CONFIGURATION UTILITIES - Exact JavaScript Equivalent
# ============================================================================

class ConfigManager:
    """
    Configuration manager - exact Python equivalent of JavaScript config handling.
    
    Provides same configuration merging and validation logic.
    """
    
    @staticmethod
    def create_simple_config(
        mongo_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[Literal['voyage', 'openai']] = None
    ) -> UniversalAIBrainConfig:
        """Create simple config - exact equivalent of JavaScript createSimpleConfig()."""
        # Use environment variables as fallback - same as JavaScript
        mongo_uri = mongo_uri or os.environ.get('MONGODB_CONNECTION_STRING') or os.environ.get('MONGODB_URI')
        api_key = api_key or os.environ.get('VOYAGE_API_KEY') or os.environ.get('OPENAI_API_KEY')
        
        # Auto-detect provider from API key - same logic as JavaScript
        if not provider and api_key:
            if os.environ.get('VOYAGE_API_KEY'):
                provider = 'voyage'
            elif os.environ.get('OPENAI_API_KEY'):
                provider = 'openai'
        
        # Generate database name with timestamp - same as JavaScript
        if not database_name:
            import time
            database_name = f"ai_brain_{int(time.time() * 1000)}"
        
        return UniversalAIBrainConfig(
            mongodb=MongoDBConfig(
                connectionString=mongo_uri,
                databaseName=database_name,
                collections=DEFAULT_CONFIG["collections"]
            ),
            intelligence=IntelligenceConfig(
                **DEFAULT_CONFIG["intelligence"],
                embeddingModel="voyage-large-2-instruct" if provider == 'voyage' else "text-embedding-3-small",
                vectorDimensions=1024 if provider == 'voyage' else 1536
            ),
            safety=SafetyConfig(**DEFAULT_CONFIG["safety"]),
            monitoring=MonitoringConfig(**DEFAULT_CONFIG["monitoring"]),
            apis=APIsConfig(**{
                provider: APIProviderConfig(
                    apiKey=api_key,
                    baseURL="https://api.voyageai.com/v1" if provider == 'voyage' else "https://api.openai.com/v1"
                )
            }) if api_key and provider else None
        )
    
    @staticmethod
    def merge_with_defaults(config: UniversalAIBrainConfig) -> UniversalAIBrainConfig:
        """Merge config with defaults - exact equivalent of JavaScript mergeWithDefaults()."""
        # Handle MongoDB configuration
        mongo_uri = (
            config.mongodb.connection_string if config.mongodb 
            else config.mongo_uri 
            or os.environ.get('MONGODB_CONNECTION_STRING')
        )
        
        database_name = (
            config.mongodb.database_name if config.mongodb 
            else config.database_name 
            or f"ai_brain_{int(__import__('time').time() * 1000)}"
        )
        
        # Merge collections with defaults
        collections = {**DEFAULT_CONFIG["collections"]}
        if config.mongodb and config.mongodb.collections:
            collections.update(config.mongodb.collections)
        
        # Merge intelligence config with defaults
        intelligence_config = {**DEFAULT_CONFIG["intelligence"]}
        if config.intelligence:
            intelligence_config.update(config.intelligence.dict(exclude_none=True, by_alias=True))
        
        # Merge safety config with defaults
        safety_config = {**DEFAULT_CONFIG["safety"]}
        if config.safety:
            safety_config.update(config.safety.dict(exclude_none=True, by_alias=True))
        
        # Merge monitoring config with defaults
        monitoring_config = {**DEFAULT_CONFIG["monitoring"]}
        if config.monitoring:
            monitoring_config.update(config.monitoring.dict(exclude_none=True, by_alias=True))
        
        return UniversalAIBrainConfig(
            mongodb=MongoDBConfig(
                connectionString=mongo_uri,
                databaseName=database_name,
                collections=collections
            ),
            intelligence=IntelligenceConfig(**intelligence_config),
            safety=SafetyConfig(**safety_config),
            monitoring=MonitoringConfig(**monitoring_config),
            selfImprovement=config.self_improvement,
            apis=config.apis
        )
