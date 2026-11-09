"""Core package: configuration, data models, and YOLO detector engine."""

from aiod.core.config import settings, Settings
from aiod.core.models import DetectionBox, DetectionResult
from aiod.core.detector import YOLODetector, get_detector

__all__ = [
    "settings",
    "Settings",
    "DetectionBox",
    "DetectionResult",
    "YOLODetector",
    "get_detector",
]

