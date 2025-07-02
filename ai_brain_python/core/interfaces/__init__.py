"""
AI Brain Python Core Interfaces

This module contains abstract base classes and interfaces for:
- Cognitive system implementations
- Framework adapters
- Storage backends
- Processing pipelines

These interfaces ensure consistency and extensibility across
all cognitive systems and framework integrations.
"""

from ai_brain_python.core.interfaces.cognitive_system import (
    CognitiveSystemInterface,
    CognitiveProcessor,
    SystemCapability,
)

from ai_brain_python.core.interfaces.framework_adapter import (
    FrameworkAdapterInterface,
    AdapterCapability,
)

from ai_brain_python.core.interfaces.storage_backend import (
    StorageBackendInterface,
    CacheBackendInterface,
)

__all__ = [
    "CognitiveSystemInterface",
    "CognitiveProcessor",
    "SystemCapability",
    "FrameworkAdapterInterface",
    "AdapterCapability",
    "StorageBackendInterface",
    "CacheBackendInterface",
]