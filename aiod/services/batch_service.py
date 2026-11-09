"""Batch image processing service for AIOD."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable, Iterable

from aiod.core.detector import YOLODetector, get_detector
from aiod.core.models import DetectionResult
from aiod.utils.exceptions import InvalidImageError
from aiod.utils.image_utils import discover_images, validate_image_file


@dataclass
class BatchItemResult:
    """Result of processing an individual image in a batch."""

    image_path: Path
    status: str  # "success" | "failed"
    result: Optional[DetectionResult] = None
    error_message: Optional[str] = None
    annotated_output_path: Optional[Path] = None
    processing_time_ms: float = 0.0
    total_objects: int = 0
    counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert item result to dictionary."""
        return {
            "image_path": str(self.image_path),
            "status": self.status,
            "error_message": self.error_message,
            "annotated_output_path": str(self.annotated_output_path) if self.annotated_output_path else None,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "total_objects": self.total_objects,
            "counts": self.counts,
        }


@dataclass
class BatchSummary:
    """Comprehensive summary of a completed batch processing run."""

    input_directory: Path
    output_directory: Optional[Path] = None
    total_discovered: int = 0
    total_successful: int = 0
    total_failed: int = 0
    total_objects_detected: int = 0
    aggregated_class_counts: Dict[str, int] = field(default_factory=dict)
    total_processing_time_ms: float = 0.0
    average_time_per_image_ms: float = 0.0
    items: List[BatchItemResult] = field(default_factory=list)

    @property
    def successful_items(self) -> List[BatchItemResult]:
        """List of successfully processed items."""
        return [item for item in self.items if item.status == "success"]

    @property
    def failed_items(self) -> List[BatchItemResult]:
        """List of failed items with their error messages."""
        return [item for item in self.items if item.status == "failed"]

    def summary_report(self) -> str:
        """Generate a clean, human-readable summary report string."""
        lines = [
            "=" * 60,
            "BATCH PROCESSING SUMMARY REPORT",
            "=" * 60,
            f"Input Directory       : {self.input_directory}",
            f"Output Directory      : {self.output_directory or 'None'}",
            f"Total Discovered      : {self.total_discovered}",
            f"Successfully Processed: {self.total_successful}",
            f"Failed                : {self.total_failed}",
            f"Total Objects Found   : {self.total_objects_detected}",
            f"Total Processing Time : {self.total_processing_time_ms:.2f} ms ({self.total_processing_time_ms / 1000.0:.2f} s)",
            f"Average Time / Image  : {self.average_time_per_image_ms:.2f} ms",
        ]

        if self.aggregated_class_counts:
            lines.append("\nClass-Wise Object Breakdown:")
            sorted_counts = sorted(self.aggregated_class_counts.items(), key=lambda x: x[1], reverse=True)
            for cls_name, count in sorted_counts:
                lines.append(f"  - {cls_name:<16}: {count}")

        if self.failed_items:
            lines.append("\nFailed Images:")
            for item in self.failed_items:
                lines.append(f"  - {item.image_path.name}: {item.error_message}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire batch summary to a serializable dictionary."""
        return {
            "input_directory": str(self.input_directory),
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "total_discovered": self.total_discovered,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "total_objects_detected": self.total_objects_detected,
            "aggregated_class_counts": self.aggregated_class_counts,
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "average_time_per_image_ms": round(self.average_time_per_image_ms, 2),
            "items": [item.to_dict() for item in self.items],
        }


class BatchProcessor:
    """
    Batch image processor executing sequential object detection on entire directories.
    Prevents memory/GPU exhaustion while providing resilient error recovery.
    """

    def __init__(self, detector: Optional[YOLODetector] = None):
        """
        Initialize BatchProcessor.

        Args:
            detector: Optional shared YOLODetector instance.
        """
        self.detector = detector if detector is not None else get_detector()

    def process_directory(
        self,
        input_directory: Union[str, Path],
        output_directory: Optional[Union[str, Path]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Iterable[Union[str, int]]] = None,
        save_annotated: bool = True,
        recursive: bool = False,
        progress_callback: Optional[Callable[[int, int, BatchItemResult], None]] = None,
    ) -> BatchSummary:
        """
        Process all supported images in a directory.

        Args:
            input_directory: Target folder to discover images in.
            output_directory: Destination folder to write annotated images.
            conf: Confidence threshold (0.0 to 1.0).
            iou: IoU threshold for NMS (0.0 to 1.0).
            classes: Optional class filter.
            save_annotated: Whether to generate and save annotated output images.
            recursive: Whether to scan subdirectories recursively.
            progress_callback: Optional callable `callback(current_idx, total, item_result)`.

        Returns:
            BatchSummary detailing overall and per-image detection statistics.

        Raises:
            InvalidImageError: If the input directory does not exist or is invalid.
        """
        input_path = Path(input_directory).resolve()
        discovered_images = discover_images(input_path, recursive=recursive)

        output_path: Optional[Path] = None
        if output_directory is not None:
            output_path = Path(output_directory).resolve()
            output_path.mkdir(parents=True, exist_ok=True)

        batch_start_time = time.perf_counter()
        items: List[BatchItemResult] = []
        aggregated_counts: Dict[str, int] = {}
        total_objects = 0
        total_images = len(discovered_images)

        for index, img_file in enumerate(discovered_images, start=1):
            item_start = time.perf_counter()

            try:
                # 1. Pre-validate image header and format before passing to model
                validate_image_file(img_file)

                # 2. Run detection with shared YOLO engine
                result = self.detector.detect(
                    source=img_file,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                )

                # 3. Save annotated output image if requested
                annotated_path: Optional[Path] = None
                if save_annotated and output_path is not None and result.annotated_array is not None:
                    out_name = f"annotated_{img_file.name}"
                    annotated_path = result.save_annotated_image(output_path / out_name)

                item_time = (time.perf_counter() - item_start) * 1000.0

                item_result = BatchItemResult(
                    image_path=img_file,
                    status="success",
                    result=result,
                    annotated_output_path=annotated_path,
                    processing_time_ms=item_time,
                    total_objects=result.total_count,
                    counts=dict(result.counts),
                )

                # Aggregate statistics
                total_objects += result.total_count
                for cls_name, count in result.counts.items():
                    aggregated_counts[cls_name] = aggregated_counts.get(cls_name, 0) + count

            except Exception as err:
                # Resilient error recovery: continue processing remaining images
                item_time = (time.perf_counter() - item_start) * 1000.0
                item_result = BatchItemResult(
                    image_path=img_file,
                    status="failed",
                    error_message=str(err),
                    processing_time_ms=item_time,
                    total_objects=0,
                    counts={},
                )

            items.append(item_result)

            if progress_callback:
                progress_callback(index, total_images, item_result)

        total_elapsed_ms = (time.perf_counter() - batch_start_time) * 1000.0
        successful_count = sum(1 for item in items if item.status == "success")
        failed_count = len(items) - successful_count
        avg_time = (total_elapsed_ms / total_images) if total_images > 0 else 0.0

        return BatchSummary(
            input_directory=input_path,
            output_directory=output_path,
            total_discovered=total_images,
            total_successful=successful_count,
            total_failed=failed_count,
            total_objects_detected=total_objects,
            aggregated_class_counts=aggregated_counts,
            total_processing_time_ms=total_elapsed_ms,
            average_time_per_image_ms=avg_time,
            items=items,
        )


# Global default batch processor instance
batch_processor = BatchProcessor()

