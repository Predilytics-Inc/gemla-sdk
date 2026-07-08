from __future__ import annotations


class GemlaError(Exception):
    """Base exception for all GEMLA SDK errors."""


class GemlaInputError(GemlaError, ValueError):
    """Raised when user input has the wrong shape, type, or value."""


class GemlaValidationError(GemlaError):
    """Raised when an internal validation or release-contract check fails."""


class GemlaConfigurationError(GemlaError):
    """Raised when a configuration object is invalid or inconsistent."""