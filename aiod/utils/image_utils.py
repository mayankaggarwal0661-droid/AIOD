"""Image validation and processing utilities for AIOD."""

import os
from pathlib import Path
from typing import Tuple, Union
import cv2
from PIL import Image, ImageTk

from aiod.utils.exceptions import InvalidImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_images(directory: Union[str, Path], recursive: bool = False) -> list[Path]:
    """
    Discover all supported image files in a directory.

    Args:
        directory: Path to the target folder.
        recursive: Whether to search subdirectories recursively.

    Returns:
        Sorted list of Path objects for discovered image files.

    Raises:
        InvalidImageError: If the directory does not exist or is not a directory.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        raise InvalidImageError(f"Directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise InvalidImageError(f"Path is not a directory: {dir_path}")

    pattern = "**/*" if recursive else "*"
    discovered = [
        p.resolve()
        for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(discovered)


def validate_image_file(file_path: Union[str, Path]) -> Path:
    """
    Validate that the given path exists, is a supported image file, and is not corrupt.

    Args:
        file_path: Path to the image file.

    Returns:
        Resolved Path object.

    Raises:
        InvalidImageError: If the file is missing, unsupported, or corrupted.
    """
    if not file_path:
        raise InvalidImageError("No image path provided.")

    path = Path(file_path).resolve()

    if not path.exists():
        raise InvalidImageError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise InvalidImageError(f"Provided path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise InvalidImageError(
            f"Unsupported file format '{path.suffix}'. Supported formats: {supported}"
        )

    if path.stat().st_size == 0:
        raise InvalidImageError(f"Image file is empty (0 bytes): {path.name}")

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise InvalidImageError(f"Corrupted or invalid image data in {path.name}: {e}") from e

    return path


def load_image_for_display(file_path: Union[str, Path], max_size: Tuple[int, int] = (780, 420)) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Load an image from disk, convert to RGB, and generate a thumbnailed PIL Image for display.

    Args:
        file_path: Path to the validated image.
        max_size: Maximum (width, height) bounding box for the thumbnail.

    Returns:
        Tuple of (thumbnailed PIL Image, original (width, height)).
    """
    path = validate_image_file(file_path)

    try:
        with Image.open(path) as raw_img:
            original_size = raw_img.size
            img = raw_img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img, original_size
    except Exception as e:
        raise InvalidImageError(f"Failed to open image {path.name}: {e}") from e


def bgr_array_to_display_image(bgr_array, max_size: Tuple[int, int] = (780, 420)) -> Image.Image:
    """
    Convert an OpenCV BGR numpy array to an RGB PIL Image resized for display.

    Args:
        bgr_array: NumPy array in BGR format (e.g. from YOLO plot()).
        max_size: Maximum (width, height) bounding box.

    Returns:
        Thumbnailed PIL Image in RGB format.
    """
    try:
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_array)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        raise InvalidImageError(f"Failed to convert plotted detection array: {e}") from e


def to_tk_photo_image(pil_image: Image.Image) -> ImageTk.PhotoImage:
    """
    Convert a PIL Image to a Tkinter PhotoImage.

    Args:
        pil_image: PIL Image instance.

    Returns:
        ImageTk.PhotoImage object.
    """
    return ImageTk.PhotoImage(pil_image)

