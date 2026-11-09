"""Custom exceptions for the AIOD system."""


class AIODError(Exception):
    """Base exception for all AIOD errors."""
    pass


class InvalidImageError(AIODError):
    """Raised when an image is missing, corrupted, or in an unsupported format."""
    pass


class ModelError(AIODError):
    """Raised when a model fails to load or inference fails."""
    pass


class DatabaseError(AIODError):
    """Raised when database connection or execution fails."""
    pass

