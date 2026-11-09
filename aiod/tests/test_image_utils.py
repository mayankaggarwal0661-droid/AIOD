"""Unit tests for image validation and processing utilities."""

import unittest
from pathlib import Path
import tempfile
from PIL import Image

from aiod.utils.image_utils import validate_image_file, load_image_for_display
from aiod.utils.exceptions import InvalidImageError


class TestImageUtils(unittest.TestCase):
    """Test suite for image utilities."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a valid test image
        self.valid_img_path = self.temp_path / "valid.jpg"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(self.valid_img_path)

        # Create an invalid extension file
        self.txt_path = self.temp_path / "document.txt"
        self.txt_path.write_text("not an image")

        # Create an empty image file
        self.empty_path = self.temp_path / "empty.png"
        self.empty_path.write_bytes(b"")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_valid_image(self):
        validated = validate_image_file(self.valid_img_path)
        self.assertEqual(validated, self.valid_img_path.resolve())

    def test_validate_missing_image(self):
        missing = self.temp_path / "non_existent.jpg"
        with self.assertRaises(InvalidImageError):
            validate_image_file(missing)

    def test_validate_unsupported_extension(self):
        with self.assertRaises(InvalidImageError):
            validate_image_file(self.txt_path)

    def test_validate_empty_file(self):
        with self.assertRaises(InvalidImageError):
            validate_image_file(self.empty_path)

    def test_load_image_for_display(self):
        thumb, orig_size = load_image_for_display(self.valid_img_path, max_size=(50, 50))
        self.assertEqual(orig_size, (100, 100))
        self.assertLessEqual(thumb.width, 50)
        self.assertLessEqual(thumb.height, 50)


if __name__ == "__main__":
    unittest.main()

