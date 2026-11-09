"""Unit and integration tests for BatchProcessor and batch image workflows."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

from aiod.core.detector import YOLODetector, get_detector
from aiod.services.batch_service import BatchProcessor, BatchSummary, BatchItemResult
from aiod.utils.exceptions import InvalidImageError
from aiod.tests.test_detector import MockYOLOModel


class TestBatchProcessor(unittest.TestCase):
    """Test suite for batch image processing with both mock and actual models."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.input_dir = self.temp_path / "images_input"
        self.output_dir = self.temp_path / "images_output"
        self.input_dir.mkdir(parents=True, exist_ok=True)

        # Create valid test images of various supported formats
        self.img1 = self.input_dir / "frame_01.jpg"
        self.img2 = self.input_dir / "frame_02.png"
        self.img3 = self.input_dir / "frame_03.bmp"

        Image.new("RGB", (320, 240), color="red").save(self.img1)
        Image.new("RGB", (320, 240), color="green").save(self.img2)
        Image.new("RGB", (320, 240), color="blue").save(self.img3)

        # Use mock model for fast, deterministic unit testing
        self.mock_model = MockYOLOModel()
        self.detector = YOLODetector(model_instance=self.mock_model)
        self.processor = BatchProcessor(detector=self.detector)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_batch_all_valid_images(self):
        """Verify processing a batch of valid images across different formats."""
        summary: BatchSummary = self.processor.process_directory(
            input_directory=self.input_dir,
            output_directory=self.output_dir,
            conf=0.5,
            save_annotated=True,
        )

        self.assertEqual(summary.total_discovered, 3)
        self.assertEqual(summary.total_successful, 3)
        self.assertEqual(summary.total_failed, 0)
        self.assertGreater(summary.total_objects_detected, 0)
        self.assertGreater(summary.total_processing_time_ms, 0)
        self.assertGreater(summary.average_time_per_image_ms, 0)

        # Verify class counts
        self.assertIn("person", summary.aggregated_class_counts)
        self.assertIn("car", summary.aggregated_class_counts)

        # Verify annotated images were created
        annotated_files = list(self.output_dir.glob("annotated_*"))
        self.assertEqual(len(annotated_files), 3)

        # Verify report generation
        report = summary.summary_report()
        self.assertIn("BATCH PROCESSING SUMMARY REPORT", report)
        self.assertIn("Successfully Processed: 3", report)

    def test_resilience_to_corrupt_files_in_batch(self):
        """Verify the batch processor continues when corrupted or invalid files are encountered."""
        # Add a corrupted image file to the input folder
        corrupt_file = self.input_dir / "frame_corrupted.jpg"
        corrupt_file.write_bytes(b"garbage bytes that fail PIL verification")

        # Add an empty 0-byte image file
        empty_file = self.input_dir / "frame_empty.png"
        empty_file.write_bytes(b"")

        # Total files: 3 valid + 2 corrupt = 5 discovered
        summary = self.processor.process_directory(
            input_directory=self.input_dir,
            output_directory=self.output_dir,
            conf=0.5,
        )

        self.assertEqual(summary.total_discovered, 5)
        self.assertEqual(summary.total_successful, 3)
        self.assertEqual(summary.total_failed, 2)

        # Check failed items
        failed_names = [item.image_path.name for item in summary.failed_items]
        self.assertIn("frame_corrupted.jpg", failed_names)
        self.assertIn("frame_empty.png", failed_names)

        for failed_item in summary.failed_items:
            self.assertEqual(failed_item.status, "failed")
            self.assertIsNotNone(failed_item.error_message)

    def test_progress_callback(self):
        """Verify progress_callback is invoked for each item in the batch."""
        callback_history = []

        def on_progress(current, total, item):
            callback_history.append((current, total, item.status))

        summary = self.processor.process_directory(
            input_directory=self.input_dir,
            output_directory=self.output_dir,
            progress_callback=on_progress,
        )

        self.assertEqual(len(callback_history), 3)
        self.assertEqual(callback_history[0], (1, 3, "success"))
        self.assertEqual(callback_history[1], (2, 3, "success"))
        self.assertEqual(callback_history[2], (3, 3, "success"))

    def test_empty_directory(self):
        """Verify empty directory results in zero discovered images without errors."""
        empty_dir = self.temp_path / "empty_folder"
        empty_dir.mkdir()

        summary = self.processor.process_directory(input_directory=empty_dir)
        self.assertEqual(summary.total_discovered, 0)
        self.assertEqual(summary.total_successful, 0)
        self.assertEqual(summary.total_failed, 0)
        self.assertEqual(len(summary.items), 0)

    def test_missing_input_directory_raises_error(self):
        """Verify non-existent directory raises InvalidImageError."""
        missing = self.temp_path / "does_not_exist"
        with self.assertRaises(InvalidImageError):
            self.processor.process_directory(input_directory=missing)

    def test_class_filtering_in_batch(self):
        """Verify class filter restricts detections across the batch."""
        summary = self.processor.process_directory(
            input_directory=self.input_dir,
            classes=["car"],
            conf=0.25,
        )
        self.assertEqual(summary.total_successful, 3)
        self.assertEqual(list(summary.aggregated_class_counts.keys()), ["car"])

    def test_batch_with_actual_yolo_model(self):
        """Integration test with the real yolo26n.pt model."""
        real_detector = get_detector()
        real_processor = BatchProcessor(detector=real_detector)

        summary = real_processor.process_directory(
            input_directory=self.input_dir,
            output_directory=self.output_dir / "actual_out",
            conf=0.25,
            save_annotated=True,
        )

        self.assertEqual(summary.total_discovered, 3)
        self.assertEqual(summary.total_successful, 3)
        self.assertEqual(summary.total_failed, 0)
        self.assertGreater(summary.total_processing_time_ms, 0)


if __name__ == "__main__":
    unittest.main()

