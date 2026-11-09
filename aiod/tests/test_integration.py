"""End-to-end integration tests for AIOD and launch script workflows."""

import unittest
import tempfile
from pathlib import Path
from PIL import Image

from aiod.core.detector import get_detector
from aiod.services.detection_service import detection_service
from aiod.database.repository import detection_repository


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end tests for the complete detection, rendering, and persistence flow."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Create a sample test image
        cls.test_image = cls.temp_path / "integration_test.jpg"
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        img.save(cls.test_image)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_standalone_detection_pipeline(self):
        """Simulates AIOD.py execution."""
        detector = get_detector()
        result = detector.detect(self.test_image, conf=0.25)
        self.assertIsNotNone(result)
        self.assertEqual(result.image_path, str(self.test_image.resolve()))
        self.assertGreater(result.inference_time_ms, 0)
        self.assertTrue(isinstance(result.formatted_summary, str))

    def test_database_detection_pipeline(self):
        """Simulates AIOD_Database.py execution."""
        result, saved, backend = detection_service.detect_and_save(self.test_image, conf=0.25)
        self.assertIsNotNone(result)
        self.assertTrue(saved)
        self.assertIn(backend, ("mysql", "sqlite"))

        # Verify recent records contain the entry
        history = detection_repository.get_recent_history(limit=5)
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["image_path"], str(self.test_image.resolve()))


if __name__ == "__main__":
    unittest.main()

