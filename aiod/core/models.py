"""Domain models and data structures for object detection and analytics."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable, Union
import cv2
import numpy as np

from aiod.utils.exceptions import InvalidImageError


@dataclass
class DetectionBox:
    """Represents a single detected object bounding box with spatial metrics."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    @property
    def x1(self) -> float:
        """Top-left X coordinate."""
        return self.bbox[0]

    @property
    def y1(self) -> float:
        """Top-left Y coordinate."""
        return self.bbox[1]

    @property
    def x2(self) -> float:
        """Bottom-right X coordinate."""
        return self.bbox[2]

    @property
    def y2(self) -> float:
        """Bottom-right Y coordinate."""
        return self.bbox[3]

    @property
    def width(self) -> float:
        """Bounding box width."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Bounding box height."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Bounding box area in pixels."""
        return self.width * self.height

    @property
    def label(self) -> str:
        """Formatted label string: e.g. 'person (0.85)'."""
        return f"{self.class_name} ({self.confidence:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection box to a JSON-serializable dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "area": round(self.area, 2),
        }


@dataclass
class DetectionResult:
    """Structured result of object detection on an image, frame, or stream."""

    image_path: Optional[str] = None
    boxes: List[DetectionBox] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    annotated_array: Optional[np.ndarray] = None
    counts: Dict[str, int] = field(default_factory=dict)
    image_dimensions: Optional[Tuple[int, int]] = None  # (width, height)

    def __post_init__(self):
        self._recalculate_counts()

    def _recalculate_counts(self):
        """Update class-wise object counts."""
        counts: Dict[str, int] = {}
        for box in self.boxes:
            counts[box.class_name] = counts.get(box.class_name, 0) + 1
        self.counts = counts

    @property
    def total_count(self) -> int:
        """Total number of objects detected."""
        return len(self.boxes)

    @property
    def total_time_ms(self) -> float:
        """Total processing time including preprocess, inference, and postprocess."""
        return self.preprocess_time_ms + self.inference_time_ms + self.postprocess_time_ms

    @property
    def formatted_summary(self) -> str:
        """
        Backward-compatible comma-separated label list.
        Matches legacy format: 'Detected Objects: person (0.85), car (0.72)'.
        """
        if not self.boxes:
            return "No objects detected."
        labels = [box.label for box in self.boxes]
        return "Detected Objects: " + ", ".join(labels)

    @property
    def legacy_detected_objects_string(self) -> str:
        """String formatted for legacy MySQL detection_history column."""
        return ", ".join(box.label for box in self.boxes)

    def filter_by_classes(self, classes: Iterable[str]) -> "DetectionResult":
        """
        Return a new DetectionResult containing only detections matching the specified class names.

        Args:
            classes: Iterable of class names to keep (case-insensitive).

        Returns:
            Filtered DetectionResult.
        """
        allowed = {c.strip().lower() for c in classes}
        filtered_boxes = [box for box in self.boxes if box.class_name.lower() in allowed]

        return DetectionResult(
            image_path=self.image_path,
            boxes=filtered_boxes,
            inference_time_ms=self.inference_time_ms,
            preprocess_time_ms=self.preprocess_time_ms,
            postprocess_time_ms=self.postprocess_time_ms,
            annotated_array=self.annotated_array,
            image_dimensions=self.image_dimensions,
        )

    def filter_by_min_confidence(self, min_confidence: float) -> "DetectionResult":
        """
        Return a new DetectionResult containing only detections with confidence >= min_confidence.

        Args:
            min_confidence: Minimum confidence threshold (0.0 to 1.0).

        Returns:
            Filtered DetectionResult.
        """
        filtered_boxes = [box for box in self.boxes if box.confidence >= min_confidence]

        return DetectionResult(
            image_path=self.image_path,
            boxes=filtered_boxes,
            inference_time_ms=self.inference_time_ms,
            preprocess_time_ms=self.preprocess_time_ms,
            postprocess_time_ms=self.postprocess_time_ms,
            annotated_array=self.annotated_array,
            image_dimensions=self.image_dimensions,
        )

    def save_annotated_image(self, output_path: Union[str, Path]) -> Path:
        """
        Save the annotated detection image with drawn bounding boxes to disk.

        Args:
            output_path: Target path for the output image.

        Returns:
            Resolved Path of the saved image.

        Raises:
            InvalidImageError: If no annotated image array is available or saving fails.
        """
        if self.annotated_array is None:
            raise InvalidImageError("No annotated image array available to save.")

        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(dest), self.annotated_array)
        if not success or not dest.exists() or dest.stat().st_size == 0:
            raise InvalidImageError(f"Failed to write annotated image to: {dest}")

        return dest

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete result to a serializable dictionary."""
        return {
            "image_path": self.image_path,
            "total_count": self.total_count,
            "counts": self.counts,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "preprocess_time_ms": round(self.preprocess_time_ms, 2),
            "postprocess_time_ms": round(self.postprocess_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "image_dimensions": self.image_dimensions,
            "detections": [box.to_dict() for box in self.boxes],
        }
