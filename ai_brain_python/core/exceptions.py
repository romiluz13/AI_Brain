"""
Error classes for Universal AI Brain Python

Exact Python equivalents of JavaScript error classes with:
- Same error hierarchy and inheritance
- Identical error codes and messages
- Matching constructor signatures
- Same error details structure
"""

from typing import Any, Dict, Optional


class BrainError(Exception):
    """
    Base error class - exact Python equivalent of JavaScript BrainError.
    
    Provides same constructor signature and error structure.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize BrainError with same parameters as JavaScript version."""
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.name = 'BrainError'
    
    def __str__(self) -> str:
        """String representation matching JavaScript error format."""
        return self.message
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"{self.name}(message='{self.message}', code='{self.code}', details={self.details})"


class FrameworkIntegrationError(BrainError):
    """
    Framework integration error - exact Python equivalent of JavaScript FrameworkIntegrationError.
    
    Provides same constructor pattern and error formatting.
    """
    
    def __init__(
        self,
        framework_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize FrameworkIntegrationError with same parameters as JavaScript version."""
        error_message = f"Framework integration error ({framework_name}): {message}"
        super().__init__(error_message, 'FRAMEWORK_INTEGRATION_ERROR', details)
        self.framework_name = framework_name
        self.name = 'FrameworkIntegrationError'


class VectorSearchError(BrainError):
    """
    Vector search error - exact Python equivalent of JavaScript VectorSearchError.
    
    Provides same constructor pattern and error formatting.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize VectorSearchError with same parameters as JavaScript version."""
        error_message = f"Vector search error: {message}"
        super().__init__(error_message, 'VECTOR_SEARCH_ERROR', details)
        self.name = 'VectorSearchError'


class CognitiveSystemError(BrainError):
    """
    Cognitive system error - Python-specific error for cognitive system failures.
    
    Follows same pattern as JavaScript error classes.
    """
    
    def __init__(
        self,
        system_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize CognitiveSystemError with system context."""
        error_message = f"Cognitive system error ({system_name}): {message}"
        super().__init__(error_message, 'COGNITIVE_SYSTEM_ERROR', details)
        self.system_name = system_name
        self.name = 'CognitiveSystemError'


class SafetyViolationError(BrainError):
    """
    Safety violation error - Python-specific error for safety violations.
    
    Follows same pattern as JavaScript error classes.
    """
    
    def __init__(
        self,
        violation_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize SafetyViolationError with violation context."""
        error_message = f"Safety violation ({violation_type}): {message}"
        super().__init__(error_message, 'SAFETY_VIOLATION_ERROR', details)
        self.violation_type = violation_type
        self.name = 'SafetyViolationError'


class DatabaseConnectionError(BrainError):
    """
    Database connection error - Python-specific error for database issues.
    
    Follows same pattern as JavaScript error classes.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize DatabaseConnectionError."""
        error_message = f"Database connection error: {message}"
        super().__init__(error_message, 'DATABASE_CONNECTION_ERROR', details)
        self.name = 'DatabaseConnectionError'


class ValidationError(BrainError):
    """
    Validation error - Python-specific error for validation failures.
    
    Follows same pattern as JavaScript error classes.
    """
    
    def __init__(
        self,
        field: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize ValidationError with field context."""
        error_message = f"Validation error ({field}): {message}"
        super().__init__(error_message, 'VALIDATION_ERROR', details)
        self.field = field
        self.name = 'ValidationError'


# Export all error classes for easy importing
__all__ = [
    'BrainError',
    'FrameworkIntegrationError',
    'VectorSearchError',
    'CognitiveSystemError',
    'SafetyViolationError',
    'DatabaseConnectionError',
    'ValidationError',
]
