"""Comprehensive automated stability and regression test suite using pytest.

Covers:
- Confidence threshold validation
- Invalid image paths and directories
- Corrupted and zero-byte image files
- Detection result formatting and spatial math
- Object counting and class breakdown
- Class filtering (case-insensitive, empty, unknown)
- Batch image processing resilience
- Database atomic insertion and schema migration
- Database transaction rollback behavior
- Database connection failures and fallback
- Missing model weights error handling
"""

import sqlite3
import dataclasses
import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from aiod.core.detector import YOLODetector, get_detector
from aiod.core.models import DetectionBox, DetectionResult
from aiod.database.connection import DatabaseConnectionManager
from aiod.database.repository import DetectionRepository
from aiod.services.batch_service import BatchProcessor, BatchSummary
from aiod.services.detection_service import DetectionService
from aiod.utils.exceptions import InvalidImageError, ModelError, DatabaseError
from aiod.utils.image_utils import validate_image_file, discover_images
from aiod.tests.test_detector import MockYOLOModel
from aiod.tests.test_database_mock import MockMySQLConnection, MockConnectionManager


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def temp_workspace(tmp_path):
    """Provides an isolated directory for test artifacts."""
    return tmp_path


@pytest.fixture
def sample_image(tmp_path):
    """Generates a valid test JPEG image."""
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (640, 480), color=(100, 150, 200)).save(img_path)
    return img_path


@pytest.fixture
def mock_detector():
    """Provides a fast, deterministic YOLO detector using MockYOLOModel."""
    model = MockYOLOModel()
    return YOLODetector(model_instance=model)


@pytest.fixture
def isolated_db_repo(tmp_path):
    """Provides a DetectionRepository backed by an isolated SQLite database."""
    db_file = tmp_path / "test_isolated.db"
    mgr = DatabaseConnectionManager()
    mgr.get_connection = lambda: (sqlite3.connect(str(db_file)), "sqlite")
    repo = DetectionRepository(connection_manager=mgr)
    repo.init_schema()
    return repo


# =====================================================================
# 1. CONFIDENCE & IOU THRESHOLD VALIDATION
# =====================================================================

@pytest.mark.parametrize("valid_conf", [0.0, 0.05, 0.25, 0.5, 0.95, 1.0])
def test_confidence_threshold_valid(mock_detector, sample_image, valid_conf):
    """Verify all valid confidence values between 0.0 and 1.0 succeed."""
    result = mock_detector.detect(sample_image, conf=valid_conf)
    assert isinstance(result, DetectionResult)


@pytest.mark.parametrize("invalid_conf", [-0.01, -0.5, 1.01, 1.5, 100.0])
def test_confidence_threshold_invalid(mock_detector, sample_image, invalid_conf):
    """Verify out-of-bounds confidence thresholds raise ValueError."""
    with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
        mock_detector.detect(sample_image, conf=invalid_conf)


@pytest.mark.parametrize("invalid_iou", [-0.1, 1.05, 2.0])
def test_iou_threshold_invalid(mock_detector, sample_image, invalid_iou):
    """Verify out-of-bounds IoU thresholds raise ValueError."""
    with pytest.raises(ValueError, match="IoU threshold must be between 0.0 and 1.0"):
        mock_detector.detect(sample_image, iou=invalid_iou)


# =====================================================================
# 2. INVALID IMAGE PATHS
# =====================================================================

def test_missing_image_path(mock_detector, temp_workspace):
    """Verify non-existent file path raises InvalidImageError."""
    missing_file = temp_workspace / "ghost_image.jpg"
    with pytest.raises(InvalidImageError, match="does not exist"):
        mock_detector.detect(missing_file)


def test_directory_passed_as_image_path(mock_detector, temp_workspace):
    """Verify passing a directory path raises InvalidImageError."""
    sub_dir = temp_workspace / "sub_folder"
    sub_dir.mkdir()
    with pytest.raises(InvalidImageError, match="not a file"):
        mock_detector.detect(sub_dir)


@pytest.mark.parametrize("bad_ext", ["data.txt", "script.py", "archive.zip", "doc.pdf"])
def test_unsupported_file_extension(mock_detector, temp_workspace, bad_ext):
    """Verify unsupported extensions are rejected before inference."""
    bad_file = temp_workspace / bad_ext
    bad_file.write_text("dummy text content")
    with pytest.raises(InvalidImageError, match="Unsupported file format"):
        mock_detector.detect(bad_file)


def test_none_or_empty_path(mock_detector):
    """Verify empty or None image path is rejected."""
    with pytest.raises(InvalidImageError, match="No image path provided"):
        mock_detector.detect("")


# =====================================================================
# 3. CORRUPTED & EMPTY IMAGE FILES
# =====================================================================

def test_zero_byte_image_file(mock_detector, temp_workspace):
    """Verify 0-byte image file raises InvalidImageError."""
    empty_file = temp_workspace / "empty.png"
    empty_file.write_bytes(b"")
    with pytest.raises(InvalidImageError, match="empty \\(0 bytes\\)"):
        mock_detector.detect(empty_file)


def test_corrupted_image_header(mock_detector, temp_workspace):
    """Verify file with .jpg extension containing random bytes is rejected."""
    corrupted = temp_workspace / "corrupted.jpg"
    corrupted.write_bytes(b"NOT_A_VALID_JPEG_HEADER_BYTES_123456789")
    with pytest.raises(InvalidImageError, match="Corrupted or invalid image data"):
        mock_detector.detect(corrupted)


# =====================================================================
# 4. DETECTION RESULT FORMATTING & SPATIAL METRICS
# =====================================================================

def test_detection_result_formatting_with_objects(mock_detector, sample_image):
    """Verify DetectionResult formatted strings and bounding box math."""
    result = mock_detector.detect(sample_image, conf=0.5)

    assert result.total_count == 3
    assert result.formatted_summary.startswith("Detected Objects: ")
    assert "person (0.95)" in result.formatted_summary
    assert "car (0.74)" in result.formatted_summary

    # Check box properties
    box: DetectionBox = result.boxes[0]
    assert box.class_name == "person"
    assert box.class_id == 0
    assert box.x1 == 10.0
    assert box.y1 == 15.0
    assert box.x2 == 100.0
    assert box.y2 == 200.0
    assert box.width == 90.0
    assert box.height == 185.0
    assert box.area == 90.0 * 185.0
    assert box.label == "person (0.95)"

    # Check serializability
    d = result.to_dict()
    assert d["total_count"] == 3
    assert len(d["detections"]) == 3
    assert "inference_time_ms" in d


def test_detection_result_formatting_empty():
    """Verify formatted string when no objects are detected."""
    empty_res = DetectionResult(image_path="none.jpg", boxes=[])
    assert empty_res.total_count == 0
    assert empty_res.formatted_summary == "No objects detected."
    assert empty_res.legacy_detected_objects_string == ""


# =====================================================================
# 5. OBJECT COUNTING
# =====================================================================

def test_class_wise_counting(mock_detector, sample_image):
    """Verify class-wise object breakdown dictionary."""
    result = mock_detector.detect(sample_image, conf=0.5)

    # In mock model: 2 persons, 1 car (dog at 0.45 is filtered out at conf=0.5)
    assert result.counts == {"person": 2, "car": 1}
    assert result.total_count == 3

    # Lower confidence to 0.40 -> includes dog
    result_low = mock_detector.detect(sample_image, conf=0.40)
    assert result_low.counts == {"person": 2, "car": 1, "dog": 1}
    assert result_low.total_count == 4


# =====================================================================
# 6. CLASS FILTERING
# =====================================================================

def test_class_filtering_single_class(mock_detector, sample_image):
    """Verify class filter keeps only requested class."""
    result = mock_detector.detect(sample_image, conf=0.25, classes=["car"])
    assert result.total_count == 1
    assert result.boxes[0].class_name == "car"


def test_class_filtering_multiple_classes(mock_detector, sample_image):
    """Verify class filter with multiple classes (person, dog)."""
    result = mock_detector.detect(sample_image, conf=0.25, classes=["person", "dog"])
    assert result.total_count == 3
    class_set = {b.class_name for b in result.boxes}
    assert class_set == {"person", "dog"}


def test_class_filtering_case_insensitivity(mock_detector, sample_image):
    """Verify class filtering handles uppercase and whitespace."""
    result = mock_detector.detect(sample_image, conf=0.25, classes=["  CAR ", "PERSON"])
    assert result.total_count == 3
    class_set = {b.class_name for b in result.boxes}
    assert class_set == {"person", "car"}


def test_class_filtering_empty_filter_allows_all(mock_detector, sample_image):
    """Verify empty or whitespace-only filter allows all classes to be detected."""
    r_empty_list = mock_detector.detect(sample_image, conf=0.25, classes=[])
    assert r_empty_list.total_count == 4

    r_whitespace = mock_detector.detect(sample_image, conf=0.25, classes=["", "   "])
    assert r_whitespace.total_count == 4

    r_none = mock_detector.detect(sample_image, conf=0.25, classes=None)
    assert r_none.total_count == 4


def test_class_filtering_unknown_class_yields_zero(mock_detector, sample_image):
    """Verify searching for an unknown class returns 0 detections cleanly without error."""
    result = mock_detector.detect(sample_image, conf=0.25, classes=["flying_saucer"])
    assert result.total_count == 0


# =====================================================================
# 7. BATCH IMAGE PROCESSING
# =====================================================================

def test_batch_processing_resilience(mock_detector, temp_workspace):
    """Verify batch processing succeeds on valid images and isolates corrupt images."""
    in_dir = temp_workspace / "batch_in"
    out_dir = temp_workspace / "batch_out"
    in_dir.mkdir()

    # Create 3 valid images
    Image.new("RGB", (320, 240), color="red").save(in_dir / "valid_1.jpg")
    Image.new("RGB", (320, 240), color="green").save(in_dir / "valid_2.png")
    Image.new("RGB", (320, 240), color="blue").save(in_dir / "valid_3.bmp")

    # Create 1 corrupt image and 1 empty image
    (in_dir / "corrupted.jpg").write_bytes(b"broken file data")
    (in_dir / "empty.png").write_bytes(b"")

    processor = BatchProcessor(detector=mock_detector)
    summary: BatchSummary = processor.process_directory(
        input_directory=in_dir,
        output_directory=out_dir,
        conf=0.5,
        save_annotated=True,
    )

    assert summary.total_discovered == 5
    assert summary.total_successful == 3
    assert summary.total_failed == 2
    assert summary.total_objects_detected == 3 * 3  # 3 valid images * 3 objects each

    # Verify error details captured on failed items
    failed_names = {item.image_path.name for item in summary.failed_items}
    assert failed_names == {"corrupted.jpg", "empty.png"}
    for f in summary.failed_items:
        assert f.status == "failed"
        assert f.error_message is not None

    # Verify annotated images were generated for valid images
    annotated_files = list(out_dir.glob("annotated_*"))
    assert len(annotated_files) == 3


# =====================================================================
# 8. DATABASE ATOMIC INSERTION & RETRIEVAL
# =====================================================================

def test_database_insertion_atomic(isolated_db_repo):
    """Verify atomic persistence into detection_history and detected_objects child table."""
    boxes = [
        DetectionBox(0, "person", 0.94, (10.0, 20.0, 80.0, 160.0)),
        DetectionBox(2, "car", 0.81, (100.0, 150.0, 300.0, 250.0)),
    ]

    detection_id, backend = isolated_db_repo.save_detection(
        image_path="test_drive.jpg",
        annotated_image_path="annotated_drive.jpg",
        model_name="yolo26n.pt",
        confidence_threshold=0.30,
        inference_time_ms=19.5,
        boxes=boxes,
    )

    assert detection_id > 0
    assert backend == "sqlite"

    # Verify master record
    record = isolated_db_repo.get_detection_by_id(detection_id)
    assert record is not None
    assert record["image_path"] == "test_drive.jpg"
    assert record["annotated_image_path"] == "annotated_drive.jpg"
    assert record["model_name"] == "yolo26n.pt"
    assert record["total_detected_objects"] == 2
    assert record["confidence_threshold"] == 0.30

    # Verify child objects
    child_objects = isolated_db_repo.get_objects_for_detection(detection_id)
    assert len(child_objects) == 2
    assert child_objects[0]["class_name"] == "person"
    assert child_objects[0]["confidence"] == 0.94
    assert child_objects[0]["bbox"] == (10.0, 20.0, 80.0, 160.0)
    assert child_objects[1]["class_name"] == "car"


# =====================================================================
# 9. DATABASE ROLLBACK BEHAVIOR
# =====================================================================

def test_database_rollback_on_failure():
    """Verify transactions rollback on error and leave connections closed."""
    mock_conn = MockMySQLConnection()
    # Force cursor failure on second query
    mock_conn.cursor_instance.executemany = lambda q, p: (_ for _ in ()).throw(RuntimeError("Disk full error"))
    mgr = MockConnectionManager(mock_conn)
    repo = DetectionRepository(connection_manager=mgr)

    boxes = [DetectionBox(0, "person", 0.9, (0, 0, 10, 10))]

    with pytest.raises(DatabaseError, match="Failed to persist detection"):
        repo.save_detection(image_path="test.jpg", boxes=boxes)

    # Verify rollback was triggered, commit was not, and connection closed
    assert mock_conn.rolled_back is True
    assert mock_conn.committed is False
    assert mock_conn.closed is True


# =====================================================================
# 10. DATABASE CONNECTION FAILURES & FALLBACK
# =====================================================================

def test_database_connection_failure_without_fallback(monkeypatch):
    """Verify DatabaseError when MySQL fails and fallback is disabled."""
    from aiod.core import config
    new_settings = dataclasses.replace(config.settings, enable_sqlite_fallback=False)
    monkeypatch.setattr("aiod.database.connection.settings", new_settings)

    mgr = DatabaseConnectionManager()

    # Force MySQL to fail
    def fail_connect(**kwargs):
        raise RuntimeError("Network unreachable")

    import mysql.connector
    monkeypatch.setattr(mysql.connector, "connect", fail_connect)

    with pytest.raises(DatabaseError, match="MySQL connection.*failed and SQLite fallback is disabled"):
        mgr.get_connection()


def test_database_connection_fallback_to_sqlite(monkeypatch, tmp_path):
    """Verify smooth fallback to SQLite when MySQL is unreachable and fallback is enabled."""
    from aiod.core import config
    new_settings = dataclasses.replace(
        config.settings,
        enable_sqlite_fallback=True,
        sqlite_db_path=tmp_path / "fallback.db",
    )
    monkeypatch.setattr("aiod.database.connection.settings", new_settings)

    mgr = DatabaseConnectionManager()

    # Force MySQL connect failure
    def fail_connect(**kwargs):
        raise RuntimeError("No MySQL server")

    import mysql.connector
    monkeypatch.setattr(mysql.connector, "connect", fail_connect)

    conn, backend = mgr.get_connection()
    assert backend == "sqlite"
    assert conn is not None
    conn.close()


# =====================================================================
# 11. MISSING MODEL WEIGHTS
# =====================================================================

def test_missing_model_weights_raises_model_error(temp_workspace):
    """Verify initialization with missing .pt file raises ModelError with path details."""
    missing_weights = temp_workspace / "non_existent_weights.pt"
    with pytest.raises(ModelError, match="Model weight file not found"):
        YOLODetector(model_path=missing_weights)

