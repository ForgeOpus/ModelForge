"""
Custom exception hierarchy for ModelForge.
Provides structured error handling across the application.
"""


class ModelForgeException(Exception):
    """Base exception for all ModelForge errors."""
    pass


class ModelAccessError(ModelForgeException):
    """Raised when user doesn't have access to a model."""
    pass


class DatasetValidationError(ModelForgeException):
    """Raised when dataset doesn't meet requirements."""
    pass


class TrainingError(ModelForgeException):
    """Raised when an error occurs during training."""
    pass


class ProviderError(ModelForgeException):
    """Raised when an error occurs with a model provider."""
    pass


class ConfigurationError(ModelForgeException):
    """Raised when there's an issue with configuration."""
    pass


class HardwareError(ModelForgeException):
    """Raised when there's an issue with hardware detection."""
    pass


class DatabaseError(ModelForgeException):
    """Raised when there's an issue with database operations."""
    pass
