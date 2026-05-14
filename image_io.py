from __future__ import annotations

from pathlib import Path

import cv2

from cv_types import ImageBGR, validate_image_bgr


class ImageProcessingError(RuntimeError):
    """Raised when image cannot be processed safely."""


def read_image(path: Path) -> ImageBGR:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageProcessingError(f"Failed to decode image: {path}")
    validate_image_bgr(image)
    return image


def write_image(path: Path, image: ImageBGR) -> None:
    validate_image_bgr(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ImageProcessingError(f"Failed to write image: {path}")
