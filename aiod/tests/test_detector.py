"""Comprehensive unit and integration tests for YOLODetector using both Mock and Real YOLO models."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
from PIL import Image

from aiod.core.detector import YOLODetector, get_detector
from aiod.core.models import DetectionResult, DetectionBox
from aiod.utils.exceptions import InvalidImageError, ModelError


# =====================================================================
# MOCK ULTRALYTICS COMPONENTS FOR PURE ISOLATED TESTING
# =====================================================================

class MockBox:
    """Simulates an Ultralytics single box entry."""

    def __init__(self, cls_id: int, conf: float, xyxy: list):
        self.cls = torch.tensor([float(cls_id)])
        self.conf = torch.tensor([conf])
        self.xyxy = torch.tensor([xyxy])


class MockUltralyticsResult:
    """Simulates an Ultralytics Results object returned by model.predict()."""

    def __init__(self, mock_boxes: list, orig_shape=(480, 640)):
        self.boxes = mock_boxes
        self.orig_shape = orig_shape
        self.speed = {"preprocess": 1.2, "inference": 15.4, "postprocess": 0.8}

    def plot(self):
        """Returns a dummy plotted BGR image array."""
        h, w = self.orig_shape
        return np.full((h, w, 3), 120, dtype=np.uint8)


class MockYOLOModel:
    """Mock YOLO model imitating Ultralytics YOLO with configurable behaviors."""

    def __init__(self, should_fail_predict=False):
        self.names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            16: "dog",
        }
        self.should_fail_predict = should_fail_predict
        self.last_predict_kwargs = {}

    def predict(self, **kwargs):
        self.last_predict_kwargs = kwargs

        if self.should_fail_predict:
            raise RuntimeError("Simulated internal GPU OOM error during inference")

        conf = kwargs.get("conf", 0.25)
        classes = kwargs.get("classes", None)

        raw_detections = [
            (0, 0.95, [10.0, 15.0, 100.0, 200.0]),  # person
            (0, 0.82, [120.0, 30.0, 180.0, 190.0]), # person
            (2, 0.74, [200.0, 50.0, 400.0, 250.0]), # car
            (16, 0.45, [50.0, 180.0, 110.0, 240.0]),# dog
        ]

        boxes = []
        for cls_id, box_conf, xyxy in raw_detections:
            if box_conf >= conf:
                if classes is None or cls_id in classes:
                    boxes.append(MockBox(cls_id, box_conf, xyxy))

        return [MockUltralyticsResult(boxes, orig_shape=(480, 640))]


# =====================================================================
# TEST SUITE: MOCK MODEL TESTS
# =====================================================================

class TestYOLODetectorMock(unittest.TestCase):
    """Tests YOLODetector logic, threshold bounds, class filtering, and input formats."""

    def setUp(self):
        self.mock_model = MockYOLOModel()
        self.detector = YOLODetector(model_instance=self.mock_model)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a sample test image file
        self.sample_image = self.temp_path / "test_frame.jpg"
        img = Image.new("RGB", (640, 480), color=(200, 100, 50))
        img.save(self.sample_image)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_threshold_validation_valid(self):
        """Verify thresholds within [0.0, 1.0] are accepted."""
        result = self.detector.detect(self.sample_image, conf=0.5, iou=0.45)
        self.assertIsInstance(result, DetectionResult)

    def test_threshold_validation_invalid_confidence(self):
        """Verify confidence outside [0.0, 1.0] raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.detect(self.sample_image, conf=-0.1)

        with self.assertRaises(ValueError):
            self.detector.detect(self.sample_image, conf=1.05)

    def test_threshold_validation_invalid_iou(self):
        """Verify IoU outside [0.0, 1.0] raises ValueError."""
        with self.assertRaises(ValueError):
            self.detector.detect(self.sample_image, iou=-0.01)

        with self.assertRaises(ValueError):
            self.detector.detect(self.sample_image, iou=1.5)

    def test_structured_detection_results(self):
        """Verify structured output fields: bounding boxes, coordinates, classes, scores."""
        result = self.detector.detect(self.sample_image, conf=0.5)

        self.assertGreater(result.total_count, 0)
        self.assertGreater(result.inference_time_ms, 0)
        self.assertIsNotNone(result.annotated_array)

        # Verify first box details
        first_box: DetectionBox = result.boxes[0]
        self.assertEqual(first_box.class_id, 0)
        self.assertEqual(first_box.class_name, "person")
        self.assertAlmostEqual(first_box.confidence, 0.95, places=2)
        self.assertEqual(first_box.bbox, (10.0, 15.0, 100.0, 200.0))
        self.assertEqual(first_box.width, 90.0)
        self.assertEqual(first_box.height, 185.0)
        self.assertEqual(first_box.area, 90.0 * 185.0)
        self.assertIn("person (0.95)", first_box.label)

    def test_class_wise_counting(self):
        """Verify class-wise object count dictionary."""
        result = self.detector.detect(self.sample_image, conf=0.5)
        # In mock data with conf=0.5: 2 persons, 1 car (dog at 0.45 is filtered out)
        self.assertEqual(result.counts.get("person"), 2)
        self.assertEqual(result.counts.get("car"), 1)
        self.assertEqual(result.total_count, 3)

    def test_class_filtering_by_name(self):
        """Verify filtering detections by object class name (e.g. ['car'])."""
        result = self.detector.detect(self.sample_image, conf=0.25, classes=["car"])
        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.boxes[0].class_name, "car")

    def test_post_filtering_by_class_method(self):
        """Verify DetectionResult.filter_by_classes()."""
        result = self.detector.detect(self.sample_image, conf=0.25)
        filtered = result.filter_by_classes(["dog"])
        self.assertEqual(filtered.total_count, 1)
        self.assertEqual(filtered.boxes[0].class_name, "dog")

    def test_post_filtering_by_min_confidence(self):
        """Verify DetectionResult.filter_by_min_confidence()."""
        result = self.detector.detect(self.sample_image, conf=0.25)
        high_conf = result.filter_by_min_confidence(0.90)
        self.assertEqual(high_conf.total_count, 1)
        self.assertAlmostEqual(high_conf.boxes[0].confidence, 0.95, places=2)

    def test_numpy_frame_input_for_video_or_webcam(self):
        """Verify frame arrays (e.g. from OpenCV or webcam) can be processed directly."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_frame(dummy_frame, conf=0.5)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.image_dimensions, (640, 480))
        self.assertGreater(result.total_count, 0)

    def test_pil_image_input(self):
        """Verify PIL Image instances can be processed directly."""
        pil_img = Image.new("RGB", (320, 240), color="green")
        result = self.detector.detect(pil_img, conf=0.5)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.image_dimensions, (320, 240))

    def test_save_annotated_image(self):
        """Verify saving annotated output image to disk."""
        result = self.detector.detect(self.sample_image, conf=0.5)
        out_file = self.temp_path / "output_annotated.jpg"

        saved_path = result.save_annotated_image(out_file)
        self.assertTrue(saved_path.exists())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_detect_and_save_convenience_method(self):
        """Verify detector.detect_and_save()."""
        out_file = self.temp_path / "subfolder" / "annotated.png"
        result, saved_path = self.detector.detect_and_save(self.sample_image, out_file)
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path, out_file.resolve())

    def test_missing_image_file_raises_error(self):
        """Verify missing image file raises InvalidImageError."""
        missing = self.temp_path / "non_existent.jpg"
        with self.assertRaises(InvalidImageError):
            self.detector.detect(missing)

    def test_corrupted_file_raises_error(self):
        """Verify corrupted image file raises InvalidImageError."""
        corrupt = self.temp_path / "corrupt.jpg"
        corrupt.write_bytes(b"corrupt header bytes not a jpeg")
        with self.assertRaises(InvalidImageError):
            self.detector.detect(corrupt)

    def test_empty_numpy_array_raises_error(self):
        """Verify empty numpy array raises InvalidImageError."""
        with self.assertRaises(InvalidImageError):
            self.detector.detect_frame(np.array([]))

    def test_missing_model_weights_raises_error(self):
        """Verify missing model weight file raises ModelError."""
        non_existent_weights = self.temp_path / "missing_model.pt"
        with self.assertRaises(ModelError):
            YOLODetector(model_path=non_existent_weights)

    def test_inference_failure_raises_model_error(self):
        """Verify internal inference exceptions are wrapped as ModelError."""
        failing_model = MockYOLOModel(should_fail_predict=True)
        failing_detector = YOLODetector(model_instance=failing_model)
        with self.assertRaises(ModelError):
            failing_detector.detect(self.sample_image)


# =====================================================================
# TEST SUITE: ACTUAL YOLO MODEL INTEGRATION TESTS (yolo26n.pt)
# =====================================================================

class TestYOLODetectorActualModel(unittest.TestCase):
    """Integration tests running against the actual yolo26n.pt weights."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Generate a test image
        cls.test_image = cls.temp_path / "actual_test.jpg"
        img = Image.new("RGB", (640, 480), color=(180, 180, 180))
        img.save(cls.test_image)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_actual_model_loaded_properly(self):
        """Verify actual yolo26n.pt model has loaded class dictionary."""
        self.assertGreater(len(self.detector.class_names), 0)
        self.assertIn(0, self.detector.class_names)
        self.assertEqual(self.detector.class_names[0], "person")

    def test_actual_inference_execution(self):
        """Verify actual YOLO model executes inference without errors."""
        result = self.detector.detect(self.test_image, conf=0.25)
        self.assertIsInstance(result, DetectionResult)
        self.assertGreater(result.inference_time_ms, 0)
        self.assertEqual(result.image_path, str(self.test_image.resolve()))
        self.assertIsInstance(result.to_dict(), dict)

    def test_actual_frame_inference(self):
        """Verify actual YOLO model processes OpenCV BGR numpy frame."""
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        result = self.detector.detect_frame(frame, conf=0.3)
        self.assertIsInstance(result, DetectionResult)
        self.assertIsNotNone(result.annotated_array)

    def test_actual_annotated_save(self):
        """Verify saving annotated image with real model output."""
        result = self.detector.detect(self.test_image, conf=0.25)
        out_file = self.temp_path / "actual_annotated.jpg"
        saved = result.save_annotated_image(out_file)
        self.assertTrue(saved.exists())
        self.assertGreater(saved.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
