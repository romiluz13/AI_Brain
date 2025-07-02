"""
Logger utility for Universal AI Brain Python

Exact Python equivalent of JavaScript logger.ts with:
- Same interface and method signatures
- Identical log levels and formatting
- Matching console output format
- Same debug behavior based on environment
"""

import os
from typing import Any, Protocol


class Logger(Protocol):
    """Logger interface - exact equivalent of JavaScript Logger interface."""
    
    def info(self, message: str, *args: Any) -> None:
        """Log info level message."""
        ...
    
    def warn(self, message: str, *args: Any) -> None:
        """Log warning level message."""
        ...
    
    def error(self, message: str, *args: Any) -> None:
        """Log error level message."""
        ...
    
    def debug(self, message: str, *args: Any) -> None:
        """Log debug level message."""
        ...


class SimpleLogger:
    """
    Simple logger implementation - exact Python equivalent of JavaScript SimpleLogger.
    
    Provides identical logging behavior:
    - Same log format with level prefixes
    - Same console output methods
    - Same debug environment checking
    """
    
    def info(self, message: str, *args: Any) -> None:
        """Log info level message - exact equivalent of JavaScript info()."""
        print(f"[INFO] {message}", *args)
    
    def warn(self, message: str, *args: Any) -> None:
        """Log warning level message - exact equivalent of JavaScript warn()."""
        print(f"[WARN] {message}", *args)
    
    def error(self, message: str, *args: Any) -> None:
        """Log error level message - exact equivalent of JavaScript error()."""
        print(f"[ERROR] {message}", *args)
    
    def debug(self, message: str, *args: Any) -> None:
        """Log debug level message - exact equivalent of JavaScript debug()."""
        if os.environ.get('NODE_ENV') == 'development':
            print(f"[DEBUG] {message}", *args)


# Global logger instance - exact equivalent of JavaScript export
logger = SimpleLogger()
