"""Unit tests for the database repository and connections."""

import unittest
import tempfile
import sqlite3
from pathlib import Path

from aiod.core.models import DetectionBox, DetectionResult
from aiod.database.connection import DatabaseConnectionManager
from aiod.database.repository import DetectionRepository


class TestDatabaseRepository(unittest.TestCase):
    """Test database persistence using isolated SQLite database."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        # Initialize manager pointing to temp SQLite
        self.conn_mgr = DatabaseConnectionManager()
        self.conn_mgr.get_connection = lambda: (
            sqlite3.connect(str(self.db_path)),
            "sqlite",
        )
        self.repo = DetectionRepository(connection_manager=self.conn_mgr)
        self.repo.init_schema()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_detection(self):
        sample_box = DetectionBox(
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox=(10.0, 20.0, 100.0, 150.0),
        )
        sample_result = DetectionResult(
            image_path="test_image.jpg",
            boxes=[sample_box],
            inference_time_ms=45.2,
        )

        detection_id, backend = self.repo.save_detection(sample_result)
        self.assertGreater(detection_id, 0)
        self.assertEqual(backend, "sqlite")

        history = self.repo.get_recent_history(limit=10)
        self.assertEqual(len(history), 1)
        self.assertIn("person (0.88)", history[0]["detected_objects"])

        # Verify child objects retrieval
        objects = self.repo.get_objects_for_detection(detection_id)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["class_name"], "person")
        self.assertEqual(objects[0]["confidence"], 0.88)


if __name__ == "__main__":
    unittest.main()
