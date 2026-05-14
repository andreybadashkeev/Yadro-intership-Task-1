from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


UInt8Array: TypeAlias = NDArray[np.uint8]
Int32Array: TypeAlias = NDArray[np.int32]

ImageBGR: TypeAlias = UInt8Array
MaskU8: TypeAlias = UInt8Array
Landmarks68: TypeAlias = Int32Array
LandmarksBatch: TypeAlias = Int32Array


def validate_image_bgr(image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
    if not image.flags.c_contiguous:
        raise ValueError("Image must be C-contiguous for predictable C/C++ interop.")


def validate_landmarks_batch(landmarks: np.ndarray) -> None:
    if landmarks.dtype != np.int32:
        raise TypeError(f"Expected int32 landmarks, got {landmarks.dtype}")
    if landmarks.ndim != 3 or landmarks.shape[1:] != (68, 2):
        raise ValueError(f"Expected landmarks shape (N, 68, 2), got {landmarks.shape}")
    if not landmarks.flags.c_contiguous:
        raise ValueError("Landmarks must be C-contiguous for predictable C/C++ interop.")


def validate_mask(mask: np.ndarray, height: int, width: int) -> None:
    if mask.dtype != np.uint8:
        raise TypeError(f"Expected uint8 mask, got {mask.dtype}")
    if mask.shape != (height, width):
        raise ValueError(f"Expected mask shape {(height, width)}, got {mask.shape}")
    if not mask.flags.c_contiguous:
        raise ValueError("Mask must be C-contiguous for predictable C/C++ interop.")
