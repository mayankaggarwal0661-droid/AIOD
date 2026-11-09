"""Unit tests for Vision Analytics."""

import unittest
from aiod.core.models import DetectionBox, DetectionResult
from aiod.analytics.vision_analytics import VisionAnalytics


class TestVisionAnalytics(unittest.TestCase):
    """Test suite for Vision Analytics."""

    def test_analyze_result_empty(self):
        result = DetectionResult(image_path="test.jpg", boxes=[], inference_time_ms=10.0)
        stats = VisionAnalytics.analyze_result(result)
        self.assertEqual(stats["total_objects"], 0)
        self.assertEqual(stats["average_confidence"], 0.0)

    def test_analyze_result_with_detections(self):
        boxes = [
            DetectionBox(0, "person", 0.90, (0, 0, 10, 10)),
            DetectionBox(0, "person", 0.80, (10, 10, 20, 20)),
            DetectionBox(2, "car", 0.70, (20, 20, 30, 30)),
        ]
        result = DetectionResult(image_path="test.jpg", boxes=boxes, inference_time_ms=25.5)
        stats = VisionAnalytics.analyze_result(result)

        self.assertEqual(stats["total_objects"], 3)
        self.assertEqual(stats["counts_by_class"]["person"], 2)
        self.assertEqual(stats["counts_by_class"]["car"], 1)
        self.assertEqual(stats["average_confidence"], 0.80)

    def test_parse_legacy_string(self):
        legacy = "person (0.92), car (0.75), dog (0.88)"
        parsed = VisionAnalytics.parse_legacy_detection_string(legacy)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["class_name"], "person")
        self.assertEqual(parsed[0]["confidence"], 0.92)


if __name__ == "__main__":
    unittest.main()

