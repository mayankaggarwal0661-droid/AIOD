"""Unit tests for DetectionService verifying class filtering, confidence thresholds, and persistence."""

import unittest
from pathlib import Path
import tempfile
from PIL import Image

from aiod.core.detector import YOLODetector
from aiod.database.connection import DatabaseConnectionManager
from aiod.database.repository import DetectionRepository
from aiod.services.detection_service import DetectionService
from aiod.tests.test_detector import MockYOLOModel


class TestDetectionService(unittest.TestCase):
    """Test suite for DetectionService class filtering, thresholds, and storage."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create test image
        self.test_img = self.temp_path / "sample.jpg"
        Image.new("RGB", (640, 480), color=(150, 100, 50)).save(self.test_img)

        # Mock detector with known boxes: 2 persons (0.95, 0.82), 1 car (0.74), 1 dog (0.45)
        self.mock_model = MockYOLOModel()
        self.detector = YOLODetector(model_instance=self.mock_model)

        # Isolated SQLite database repository
        self.db_path = self.temp_path / "service_test.db"
        self.conn_mgr = DatabaseConnectionManager()
        self.conn_mgr.get_connection = lambda: (
            __import__("sqlite3").connect(str(self.db_path)),
            "sqlite",
        )
        self.repo = DetectionRepository(connection_manager=self.conn_mgr)
        self.repo.init_schema()

        self.service = DetectionService(detector=self.detector, repository=self.repo)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detect_only_no_class_filter(self):
        """Verify detect_only with no class filter detects all objects above confidence."""
        result = self.service.detect_only(self.test_img, conf=0.5)
        self.assertEqual(result.total_count, 3)
        self.assertIn("person", result.counts)
        self.assertIn("car", result.counts)

    def test_detect_only_empty_class_filter(self):
        """Verify empty or whitespace class filters allow all classes to be detected."""
        # Empty list
        r1 = self.service.detect_only(self.test_img, conf=0.5, classes=[])
        self.assertEqual(r1.total_count, 3)

        # Whitespace strings in list
        r2 = self.service.detect_only(self.test_img, conf=0.5, classes=["", "   "])
        self.assertEqual(r2.total_count, 3)

    def test_detect_only_with_specific_class_filter(self):
        """Verify filtering for specific classes (e.g. ['car', 'dog'])."""
        result = self.service.detect_only(self.test_img, conf=0.25, classes=["car", "dog"])
        self.assertEqual(result.total_count, 2)
        classes_found = {b.class_name for b in result.boxes}
        self.assertEqual(classes_found, {"car", "dog"})

    def test_detect_and_save_with_classes_argument(self):
        """Verify detect_and_save accepts classes argument and saves correctly to database."""
        annotated_path = self.temp_path / "annotated_sample.jpg"

        result, saved, backend = self.service.detect_and_save(
            image_path=self.test_img,
            conf=0.5,
            classes=["person"],
            annotated_image_path=annotated_path,
        )

        self.assertTrue(saved)
        self.assertEqual(backend, "sqlite")
        self.assertEqual(result.total_count, 2)
        self.assertEqual(list(result.counts.keys()), ["person"])

        # Check database records
        recent = self.repo.get_recent_detections(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["total_detected_objects"], 2)
        self.assertEqual(recent[0]["annotated_image_path"], str(annotated_path))

        # Check child objects table
        detection_id = recent[0]["id"]
        objects = self.repo.get_objects_for_detection(detection_id)
        self.assertEqual(len(objects), 2)
        for obj in objects:
            self.assertEqual(obj["class_name"], "person")

    def test_detect_and_save_different_confidence_thresholds(self):
        """Verify threshold adjustment influences detected count and database record."""
        # High confidence 0.90 -> only 1 person at 0.95 matches
        res_high, saved_high, _ = self.service.detect_and_save(
            image_path=self.test_img,
            conf=0.90,
        )
        self.assertTrue(saved_high)
        self.assertEqual(res_high.total_count, 1)

        # Lower confidence 0.40 -> all 4 mock detections match
        res_low, saved_low, _ = self.service.detect_and_save(
            image_path=self.test_img,
            conf=0.40,
        )
        self.assertTrue(saved_low)
        self.assertEqual(res_low.total_count, 4)


if __name__ == "__main__":
    unittest.main()

