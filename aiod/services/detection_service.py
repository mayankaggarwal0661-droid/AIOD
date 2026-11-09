"""High-level orchestration service for image detection and persistence."""

from pathlib import Path
from typing import Optional, Union, Tuple, Iterable, Any
from aiod.core.detector import YOLODetector, get_detector
from aiod.core.models import DetectionResult
from aiod.database.repository import DetectionRepository, detection_repository
from aiod.utils.exceptions import AIODError


class DetectionService:
    """
    Coordinates object detection and optional database persistence.
    Decouples UI and presentation logic from core detection and database operations.
    """

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        repository: Optional[DetectionRepository] = None,
    ):
        self.detector = detector if detector is not None else get_detector()
        self.repository = repository if repository is not None else detection_repository

    def detect_only(
        self,
        image_path: Union[str, Path, Any],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Iterable[Union[str, int]]] = None,
    ) -> DetectionResult:
        """
        Execute detection on an image without saving to database.

        Args:
            image_path: Path to the image file, numpy array, or PIL Image.
            conf: Optional confidence threshold.
            iou: Optional IoU threshold.
            classes: Optional list of class names or IDs to filter by.

        Returns:
            DetectionResult domain object.
        """
        return self.detector.detect(image_path, conf=conf, iou=iou, classes=classes)

    def detect_and_save(
        self,
        image_path: Union[str, Path, Any],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Iterable[Union[str, int]]] = None,
        annotated_image_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[DetectionResult, bool, str]:
        """
        Execute detection and persist the result into the database.

        Args:
            image_path: Path to the image file, numpy array, or PIL Image.
            conf: Optional confidence threshold.
            iou: Optional IoU threshold.
            classes: Optional list of class names or IDs to filter by.
            annotated_image_path: Optional path to the annotated output image.
            model_name: Optional model identifier string.

        Returns:
            Tuple of (DetectionResult, saved_successfully_bool, backend_used_string).
        """
        result = self.detector.detect(image_path, conf=conf, iou=iou, classes=classes)

        try:
            resolved_model = model_name or Path(self.detector.model_path).name
            resolved_conf = conf if conf is not None else 0.25
            image_path_str = str(image_path) if isinstance(image_path, (str, Path)) else "memory_frame"

            detection_id, backend = self.repository.save_detection(
                image_path=image_path_str,
                annotated_image_path=annotated_image_path,
                model_name=resolved_model,
                confidence_threshold=resolved_conf,
                inference_time_ms=result.inference_time_ms,
                boxes=result.boxes,
                result=result,
            )
            return result, bool(detection_id), backend
        except Exception as e:
            # Detection succeeded, but saving failed
            return result, False, str(e)


# Default singleton instance
detection_service = DetectionService()
