"""Integration tests for web API endpoints verifying browser request flows."""

import unittest
import io
from pathlib import Path
from PIL import Image

from app import app


class TestWebAPI(unittest.TestCase):
    """Test suite for web application REST endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

        # Generate a test image bytes buffer
        img = Image.new("RGB", (320, 240), color=(200, 150, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cls.test_image_bytes = buf.getvalue()

    def test_api_detect_no_class_filter(self):
        """Verify POST /api/detect with no class filter succeeds."""
        response = self.client.post(
            "/api/detect",
            data={
                "image": (io.BytesIO(self.test_image_bytes), "test_no_filter.jpg"),
                "confidence": "0.25",
                "classes": "",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("annotated_image_url", data)
        self.assertIn("total_count", data)
        self.assertIn("counts", data)
        self.assertIn("inference_time_ms", data)
        self.assertTrue(data["saved_to_db"])

    def test_api_detect_with_class_filter(self):
        """Verify POST /api/detect with classes filter (e.g. person, car, dog) succeeds."""
        response = self.client.post(
            "/api/detect",
            data={
                "image": (io.BytesIO(self.test_image_bytes), "test_with_filter.jpg"),
                "confidence": "0.30",
                "classes": "person, car, dog",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("annotated_image_url", data)
        self.assertTrue(data["saved_to_db"])

    def test_api_detect_high_confidence_threshold(self):
        """Verify POST /api/detect with high confidence threshold succeeds."""
        response = self.client.post(
            "/api/detect",
            data={
                "image": (io.BytesIO(self.test_image_bytes), "test_high_conf.jpg"),
                "confidence": "0.85",
                "classes": "",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["saved_to_db"])

    def test_api_detect_invalid_file_rejected(self):
        """Verify POST /api/detect with invalid non-image file returns 400."""
        response = self.client.post(
            "/api/detect",
            data={
                "image": (io.BytesIO(b"not an image"), "test.txt"),
                "confidence": "0.25",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()

