"""Unit tests for GeminiVisionService."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image

from aiod.services.gemini_service import GeminiVisionService


class TestGeminiVisionService(unittest.TestCase):
    """Test suite for GeminiVisionService."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a test image
        self.test_img = self.temp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="blue").save(self.test_img)

        self.service = GeminiVisionService(api_key="mock_test_api_key_for_testing_123")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_is_configured_true(self):
        self.assertTrue(self.service.is_configured)

    def test_is_configured_false_when_empty(self):
        unconfigured = GeminiVisionService(api_key="")
        self.assertFalse(unconfigured.is_configured)

    @patch("requests.post")
    def test_analyze_scene_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "The scene displays a solid blue visual background."}
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        res = self.service.analyze_scene(
            image_path=self.test_img,
            detection_summary="1x person",
        )

        self.assertTrue(res["success"])
        self.assertIn("solid blue visual background", res["analysis"])
        self.assertEqual(res["model"], "gemini-1.5-flash")

    @patch("requests.post")
    def test_analyze_scene_api_error_handling(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "API key not valid. Please pass a valid API key."}
        }
        mock_post.return_value = mock_response

        res = self.service.analyze_scene(image_path=self.test_img)

        self.assertFalse(res["success"])
        self.assertIn("API key not valid", res["error"])

    def test_analyze_scene_missing_image(self):
        missing = self.temp_path / "non_existent.jpg"
        res = self.service.analyze_scene(image_path=missing)
        self.assertFalse(res["success"])
        self.assertIn("not found", res["error"])


if __name__ == "__main__":
    unittest.main()

