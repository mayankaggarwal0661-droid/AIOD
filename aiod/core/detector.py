"""Shared YOLO Object Detection Engine for AIOD."""

import time
import threading
from pathlib import Path
from typing import Optional, Union, List, Iterable, Dict, Tuple, Any
import numpy as np
from PIL import Image
from ultralytics import YOLO

from aiod.core.config import settings
from aiod.core.models import DetectionBox, DetectionResult
from aiod.utils.exceptions import ModelError, InvalidImageError
from aiod.utils.image_utils import validate_image_file


class YOLODetector:
    """
    Core YOLO object detection engine.
    Supports single images, video frames, PIL images, and OpenCV arrays.
    Provides configurable confidence/IoU thresholds, class filtering, and latency profiling.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        model_instance: Optional[Any] = None,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to YOLO weights (defaults to settings.model_path / 'yolo26n.pt').
            device: Target device string ('cpu', 'cuda', or empty for auto-detection).
            model_instance: Optional pre-loaded model or mock for testing.
        """
        self.model_path = Path(model_path) if model_path else settings.model_path
        self.device = device if device is not None else settings.device

        if model_instance is not None:
            self._model = model_instance
        else:
            self._model = self._load_model()

    def _load_model(self) -> YOLO:
        """Load and initialize the YOLO model weights with error handling."""
        if not self.model_path.exists():
            raise ModelError(
                f"Model weight file not found at: {self.model_path}. "
                f"Please ensure '{self.model_path.name}' is present."
            )

        try:
            return YOLO(str(self.model_path))
        except Exception as e:
            raise ModelError(f"Failed to load YOLO model from {self.model_path}: {e}") from e

    @property
    def class_names(self) -> Dict[int, str]:
        """Map of class indices to class names."""
        if hasattr(self._model, "names") and isinstance(self._model.names, dict):
            return self._model.names
        return {}

    def _validate_thresholds(self, conf: Optional[float], iou: Optional[float]) -> Tuple[float, float]:
        """Validate and return confidence and IoU thresholds."""
        confidence = conf if conf is not None else settings.default_confidence
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got: {confidence}")

        iou_threshold = iou if iou is not None else settings.default_iou
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got: {iou_threshold}")

        return confidence, iou_threshold

    def _resolve_class_filter(self, classes: Optional[Iterable[Union[str, int]]]) -> Tuple[Optional[List[int]], Optional[set]]:
        """
        Map a list of class names or IDs to class IDs recognized by YOLO.
        An empty filter or filter containing only whitespace allows all classes.

        Returns:
            Tuple of (list_of_class_ids_for_yolo, set_of_allowed_lowercase_names)
        """
        if not classes:
            return None, None

        name_to_id = {v.lower(): k for k, v in self.class_names.items()}
        class_ids: List[int] = []
        allowed_names = set()

        for c in classes:
            if isinstance(c, int):
                class_ids.append(c)
                if c in self.class_names:
                    allowed_names.add(self.class_names[c].lower())
            elif isinstance(c, str):
                normalized = c.strip().lower()
                if normalized:
                    allowed_names.add(normalized)
                    if normalized in name_to_id:
                        class_ids.append(name_to_id[normalized])

        if not allowed_names and not class_ids:
            return None, None

        return class_ids if class_ids else None, allowed_names

    def detect(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Iterable[Union[str, int]]] = None,
    ) -> DetectionResult:
        """
        Execute object detection on an image path, NumPy array, or PIL Image.

        Args:
            source: Image file path, NumPy BGR/RGB array, or PIL Image.
            conf: Confidence threshold (0.0 to 1.0).
            iou: Intersection-over-Union threshold for NMS (0.0 to 1.0).
            classes: Optional list of class names or IDs to filter detections by.

        Returns:
            Structured DetectionResult with bounding boxes, labels, counts, and timing.

        Raises:
            InvalidImageError: If the image source is invalid, empty, or corrupt.
            ModelError: If model prediction fails.
            ValueError: If confidence or IoU thresholds are outside [0.0, 1.0].
        """
        confidence_thresh, iou_thresh = self._validate_thresholds(conf, iou)
        class_id_filter, allowed_class_names = self._resolve_class_filter(classes)

        image_path_str: Optional[str] = None
        dimensions: Optional[Tuple[int, int]] = None
        predict_input: Any = source

        if isinstance(source, (str, Path)):
            validated_path = validate_image_file(source)
            image_path_str = str(validated_path)
            predict_input = image_path_str
        elif isinstance(source, np.ndarray):
            if source.size == 0 or source.ndim < 2:
                raise InvalidImageError("Invalid NumPy image array: array is empty or lacks spatial dimensions.")
            dimensions = (source.shape[1], source.shape[0])
            predict_input = source
        elif isinstance(source, Image.Image):
            if source.width == 0 or source.height == 0:
                raise InvalidImageError("Invalid PIL Image: width or height is 0.")
            dimensions = source.size
            predict_input = source
        else:
            raise InvalidImageError(f"Unsupported image input type: {type(source).__name__}")

        predict_kwargs: Dict[str, Any] = {
            "source": predict_input,
            "conf": confidence_thresh,
            "iou": iou_thresh,
            "verbose": False,
        }
        if class_id_filter:
            predict_kwargs["classes"] = class_id_filter
        if self.device:
            predict_kwargs["device"] = self.device

        start_time = time.perf_counter()

        try:
            results = self._model.predict(**predict_kwargs)
        except Exception as e:
            source_desc = image_path_str if image_path_str else f"frame ({type(source).__name__})"
            raise ModelError(f"YOLO prediction failed for {source_desc}: {e}") from e

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not results:
            return DetectionResult(
                image_path=image_path_str,
                boxes=[],
                inference_time_ms=total_elapsed_ms,
                annotated_array=None,
                image_dimensions=dimensions,
            )

        result = results[0]

        # Extract timing metrics
        preprocess_ms = 0.0
        inference_ms = total_elapsed_ms
        postprocess_ms = 0.0
        if hasattr(result, "speed") and isinstance(result.speed, dict):
            preprocess_ms = result.speed.get("preprocess", 0.0)
            inference_ms = result.speed.get("inference", total_elapsed_ms)
            postprocess_ms = result.speed.get("postprocess", 0.0)

        # Extract image dimensions if available from result
        if dimensions is None and hasattr(result, "orig_shape") and result.orig_shape:
            # orig_shape is (height, width)
            dimensions = (result.orig_shape[1], result.orig_shape[0])

        detected_boxes: List[DetectionBox] = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                box_conf = float(box.conf[0].item())
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # Secondary post-filter check if class filter provided
                if allowed_class_names and class_name.lower() not in allowed_class_names:
                    continue

                coords = tuple(box.xyxy[0].tolist())
                detected_boxes.append(
                    DetectionBox(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=box_conf,
                        bbox=coords,  # (x1, y1, x2, y2)
                    )
                )

        annotated_array = result.plot() if hasattr(result, "plot") else None

        return DetectionResult(
            image_path=image_path_str,
            boxes=detected_boxes,
            inference_time_ms=inference_ms,
            preprocess_time_ms=preprocess_ms,
            postprocess_time_ms=postprocess_ms,
            annotated_array=annotated_array,
            image_dimensions=dimensions,
        )

    def detect_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Iterable[Union[str, int]]] = None,
    ) -> DetectionResult:
        """
        Specialized detection method for streaming video and webcam frames.

        Args:
            frame: OpenCV BGR image array.
            conf: Confidence threshold.
            iou: IoU threshold.
            classes: Optional class filter.

        Returns:
            DetectionResult.
        """
        return self.detect(source=frame, conf=conf, iou=iou, classes=classes)

    def detect_and_save(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        output_path: Union[str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Iterable[Union[str, int]]] = None,
    ) -> Tuple[DetectionResult, Path]:
        """
        Execute detection and automatically save the annotated image to output_path.

        Args:
            source: Image path or image array.
            output_path: File path where annotated output should be saved.
            conf: Confidence threshold.
            iou: IoU threshold.
            classes: Class filter.

        Returns:
            Tuple of (DetectionResult, saved_output_path).
        """
        result = self.detect(source=source, conf=conf, iou=iou, classes=classes)
        saved_path = result.save_annotated_image(output_path)
        return result, saved_path


# Global singleton detector instance with thread safety lock
_detector_lock = threading.Lock()
_shared_detector: Optional[YOLODetector] = None


def get_detector(model_path: Optional[Union[str, Path]] = None) -> YOLODetector:
    """
    Retrieve or create the shared YOLO detector singleton in a thread-safe manner.
    Prevents loading multiple copies of the model weights into memory.
    """
    global _shared_detector
    if _shared_detector is None:
        with _detector_lock:
            if _shared_detector is None:
                _shared_detector = YOLODetector(model_path=model_path)
    return _shared_detector
