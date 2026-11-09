"""Vision analytics and aggregation metrics for AIOD."""

import re
from typing import List, Dict, Any, Optional
from aiod.core.models import DetectionResult


class VisionAnalytics:
    """Provides analytical processing and metrics on detection results."""

    @staticmethod
    def analyze_result(result: DetectionResult) -> Dict[str, Any]:
        """
        Compute analytics summary for a single detection result.

        Args:
            result: The DetectionResult to analyze.

        Returns:
            Dictionary with counts, confidence stats, and latency metrics.
        """
        if not result.boxes:
            return {
                "total_objects": 0,
                "unique_classes": 0,
                "counts_by_class": {},
                "average_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
                "inference_time_ms": round(result.inference_time_ms, 2),
            }

        confidences = [box.confidence for box in result.boxes]
        return {
            "total_objects": len(result.boxes),
            "unique_classes": len(result.counts),
            "counts_by_class": dict(result.counts),
            "average_confidence": round(sum(confidences) / len(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "min_confidence": round(min(confidences), 4),
            "inference_time_ms": round(result.inference_time_ms, 2),
        }

    @staticmethod
    def parse_legacy_detection_string(detection_text: str) -> List[Dict[str, Any]]:
        """
        Parse legacy detection string like 'person (0.85), car (0.72)' into structured items.

        Args:
            detection_text: Comma-separated string from detection_history.

        Returns:
            List of dicts with 'class_name' and 'confidence'.
        """
        if not detection_text or detection_text == "No objects detected.":
            return []

        pattern = re.compile(r"([a-zA-Z0-9_\-\s]+)\s*\(([0-9.]+)\)")
        matches = pattern.findall(detection_text)

        parsed = []
        for class_name, conf_str in matches:
            try:
                parsed.append({
                    "class_name": class_name.strip(),
                    "confidence": float(conf_str),
                })
            except ValueError:
                continue
        return parsed

    @classmethod
    def aggregate_history(cls, history_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate analytics across a collection of historical detection rows.

        Args:
            history_rows: List of rows retrieved from detection_history.

        Returns:
            Aggregated dictionary with total images, total objects, and top classes.
        """
        total_images = len(history_rows)
        class_frequencies: Dict[str, int] = {}
        all_confidences: List[float] = []

        for row in history_rows:
            raw_text = row.get("detected_objects", "")
            items = cls.parse_legacy_detection_string(raw_text)
            for item in items:
                name = item["class_name"]
                class_frequencies[name] = class_frequencies.get(name, 0) + 1
                all_confidences.append(item["confidence"])

        total_detected_items = sum(class_frequencies.values())
        avg_conf = (
            round(sum(all_confidences) / len(all_confidences), 4)
            if all_confidences
            else 0.0
        )

        sorted_classes = sorted(
            class_frequencies.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "total_images_processed": total_images,
            "total_objects_detected": total_detected_items,
            "overall_average_confidence": avg_conf,
            "top_detected_classes": sorted_classes[:10],
            "class_distribution": class_frequencies,
        }


vision_analytics = VisionAnalytics()

