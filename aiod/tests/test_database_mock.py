"""Unit tests for DetectionRepository using Mock MySQL connections and isolated SQLite."""

import unittest
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path
import tempfile
import sqlite3

from aiod.core.models import DetectionBox, DetectionResult
from aiod.database.connection import DatabaseConnectionManager
from aiod.database.repository import DetectionRepository
from aiod.utils.exceptions import DatabaseError


# =====================================================================
# MOCK MYSQL COMPONENTS FOR DETERMINISTIC UNIT TESTING
# =====================================================================

class MockMySQLCursor:
    """Mock MySQL cursor tracking queries, params, and lifecycle."""

    def __init__(self, connection):
        self.connection = connection
        self.lastrowid = 42
        self.closed = False
        self.executed_queries: List[Tuple[str, Any]] = []
        self.executed_many: List[Tuple[str, List[Any]]] = []
        self._fake_fetchall_data: List[Any] = []
        self._fake_fetchone_data: Optional[Any] = None

    def execute(self, query: str, params: Optional[Tuple] = None):
        self.executed_queries.append((query.strip(), params))
        # Handle SHOW COLUMNS query simulation
        if "SHOW COLUMNS FROM" in query:
            self._fake_fetchall_data = [
                ("id", "int"),
                ("image_path", "varchar(500)"),
                ("detected_objects", "text"),
                ("detection_time", "timestamp"),
            ]
        # Handle SELECT queries
        elif "SELECT id, image_path" in query and "WHERE id =" in query:
            self._fake_fetchone_data = (
                42, "test.jpg", "annotated_test.jpg", "yolo26n.pt", 0.25, 12.5, 2, "person (0.90), car (0.80)", "2026-11-05 12:00:00"
            )
        elif "SELECT id, detection_id" in query:
            self._fake_fetchall_data = [
                (1, 42, 0, "person", 0.90, 10.0, 20.0, 50.0, 80.0),
                (2, 42, 2, "car", 0.80, 100.0, 150.0, 300.0, 250.0),
            ]

    def executemany(self, query: str, seq_of_params: List[Tuple]):
        self.executed_many.append((query.strip(), seq_of_params))

    def fetchall(self):
        return self._fake_fetchall_data

    def fetchone(self):
        return self._fake_fetchone_data

    def close(self):
        self.closed = True


class MockMySQLConnection:
    """Mock MySQL connection tracking transaction state and connection lifecycle."""

    def __init__(self, fail_on_execute=False):
        self.fail_on_execute = fail_on_execute
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.cursor_instance = MockMySQLCursor(self)

    def is_connected(self):
        return not self.closed

    def cursor(self):
        if self.fail_on_execute:
            raise RuntimeError("Simulated connection drop or database failure")
        return self.cursor_instance

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class MockConnectionManager(DatabaseConnectionManager):
    """Overrides get_connection to return mock MySQL connections."""

    def __init__(self, mock_conn: MockMySQLConnection):
        super().__init__()
        self.mock_conn = mock_conn
        self._schema_initialized = True

    def get_connection(self):
        return self.mock_conn, "mysql"


# =====================================================================
# TEST SUITE: MOCK DATABASE UNIT TESTS
# =====================================================================

class TestDatabaseRepositoryMock(unittest.TestCase):
    """Test suite using mock MySQL connection to verify queries, transactions, and lifecycle."""

    def setUp(self):
        self.mock_conn = MockMySQLConnection()
        self.mock_conn_mgr = MockConnectionManager(self.mock_conn)
        self.repo = DetectionRepository(connection_manager=self.mock_conn_mgr)

    def test_schema_initialization_with_migration(self):
        """Verify non-destructive schema initialization and column addition with mock MySQL."""
        success = self.repo.init_schema()
        self.assertTrue(success)
        self.assertTrue(self.mock_conn.committed)

        cursor = self.mock_conn.cursor_instance
        query_texts = [q for q, _ in cursor.executed_queries]

        # Verify CREATE TABLE IF NOT EXISTS was executed
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS detection_history" in q for q in query_texts))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS detected_objects" in q for q in query_texts))

        # Verify safe ALTER TABLE statements were issued for missing columns
        self.assertTrue(any("ALTER TABLE detection_history ADD COLUMN annotated_image_path" in q for q in query_texts))
        self.assertTrue(any("ALTER TABLE detection_history ADD COLUMN total_detected_objects" in q for q in query_texts))

    def test_save_detection_transaction_commit(self):
        """Verify parameterized queries, foreign key insertion, and commit on success."""
        boxes = [
            DetectionBox(0, "person", 0.92, (10.0, 20.0, 50.0, 100.0)),
            DetectionBox(2, "car", 0.85, (100.0, 120.0, 250.0, 200.0)),
        ]

        detection_id, backend = self.repo.save_detection(
            image_path="test_frame.jpg",
            annotated_image_path="annotated_frame.jpg",
            model_name="yolo26n.pt",
            confidence_threshold=0.30,
            inference_time_ms=18.4,
            boxes=boxes,
        )

        self.assertEqual(detection_id, 42)
        self.assertEqual(backend, "mysql")
        self.assertTrue(self.mock_conn.committed)
        self.assertFalse(self.mock_conn.rolled_back)
        self.assertTrue(self.mock_conn.cursor_instance.closed)
        self.assertTrue(self.mock_conn.closed)

        cursor = self.mock_conn.cursor_instance
        # Check parameterized insert query on detection_history
        history_queries = [
            (q, p) for q, p in cursor.executed_queries if "INSERT INTO detection_history" in q
        ]
        self.assertEqual(len(history_queries), 1)
        query, params = history_queries[0]
        self.assertIn("%s", query)
        self.assertEqual(params[0], "test_frame.jpg")
        self.assertEqual(params[1], "annotated_frame.jpg")
        self.assertEqual(params[2], "yolo26n.pt")
        self.assertEqual(params[3], 0.30)
        self.assertEqual(params[4], 18.4)
        self.assertEqual(params[5], 2)  # total_detected_objects

        # Check executemany on detected_objects
        self.assertEqual(len(cursor.executed_many), 1)
        obj_query, obj_params = cursor.executed_many[0]
        self.assertIn("INSERT INTO detected_objects", obj_query)
        self.assertEqual(len(obj_params), 2)
        # Check first box params: (detection_id, class_id, class_name, conf, x1, y1, x2, y2)
        self.assertEqual(obj_params[0], (42, 0, "person", 0.92, 10.0, 20.0, 50.0, 100.0))

    def test_save_detection_transaction_rollback_on_error(self):
        """Verify rollback and error handling if an exception occurs."""
        failing_conn = MockMySQLConnection()
        # Force error during execute
        failing_conn.cursor_instance.execute = lambda q, p=None: (_ for _ in ()).throw(RuntimeError("DB write failed"))
        failing_mgr = MockConnectionManager(failing_conn)
        repo = DetectionRepository(connection_manager=failing_mgr)

        with self.assertRaises(DatabaseError):
            repo.save_detection(image_path="error.jpg", boxes=[])

        self.assertTrue(failing_conn.rolled_back)
        self.assertFalse(failing_conn.committed)
        self.assertTrue(failing_conn.closed)

    def test_get_detection_by_id_mock(self):
        """Verify parameterized retrieval of a single detection."""
        record = self.repo.get_detection_by_id(42)
        self.assertIsNotNone(record)
        self.assertEqual(record["id"], 42)
        self.assertEqual(record["image_path"], "test.jpg")
        self.assertEqual(record["model_name"], "yolo26n.pt")

    def test_get_objects_for_detection_mock(self):
        """Verify retrieving granular detected objects for a given detection ID."""
        objects = self.repo.get_objects_for_detection(42)
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0]["class_name"], "person")
        self.assertEqual(objects[0]["confidence"], 0.90)
        self.assertEqual(objects[1]["class_name"], "car")


# =====================================================================
# TEST SUITE: ISOLATED SQLITE INTEGRATION TESTS
# =====================================================================

class TestDatabaseRepositorySQLite(unittest.TestCase):
    """Verify structured database operations with isolated SQLite instance."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_structured.db"

        self.conn_mgr = DatabaseConnectionManager()
        # Direct connection to temporary SQLite file
        self.conn_mgr.get_connection = lambda: (
            sqlite3.connect(str(self.db_path)),
            "sqlite",
        )
        self.repo = DetectionRepository(connection_manager=self.conn_mgr)
        self.repo.init_schema()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sqlite_full_structured_flow(self):
        """Test schema initialization, atomic insert, and relational retrieval."""
        boxes = [
            DetectionBox(0, "person", 0.95, (10.0, 20.0, 60.0, 120.0)),
            DetectionBox(16, "dog", 0.88, (150.0, 180.0, 220.0, 250.0)),
        ]

        detection_id, backend = self.repo.save_detection(
            image_path="street.jpg",
            annotated_image_path="annotated_street.jpg",
            model_name="yolo26n.pt",
            confidence_threshold=0.25,
            inference_time_ms=22.3,
            boxes=boxes,
        )

        self.assertGreater(detection_id, 0)
        self.assertEqual(backend, "sqlite")

        # Retrieve master record
        detection = self.repo.get_detection_by_id(detection_id)
        self.assertIsNotNone(detection)
        self.assertEqual(detection["image_path"], "street.jpg")
        self.assertEqual(detection["annotated_image_path"], "annotated_street.jpg")
        self.assertEqual(detection["total_detected_objects"], 2)

        # Retrieve child objects
        child_objects = self.repo.get_objects_for_detection(detection_id)
        self.assertEqual(len(child_objects), 2)
        self.assertEqual(child_objects[0]["class_name"], "person")
        self.assertEqual(child_objects[1]["class_name"], "dog")

        # Retrieve combined
        combined = self.repo.get_detection_with_objects(detection_id)
        self.assertIsNotNone(combined)
        self.assertEqual(len(combined["objects"]), 2)


if __name__ == "__main__":
    unittest.main()

