from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from cv_types import ImageBGR, LandmarksBatch, MaskU8, validate_image_bgr, validate_landmarks_batch, validate_mask
from image_io import read_image
from image_io import write_image
from skin_mask_math import build_skin_mask

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class LandmarkDetectorLike(Protocol):
    def detect(self, image_bgr: ImageBGR) -> LandmarksBatch:
        """Return landmarks as shape (N, 68, 2), dtype int32."""


def extract_skin_mask_to_file(
    image: ImageBGR,
    detector: LandmarkDetectorLike,
    output_path: Path,
    forehead_offset_ratio: float = 0.25,
    raise_if_no_faces: bool = True,
    out_mask: MaskU8 | None = None,
    out_result: ImageBGR | None = None,
) -> Path:
    result = extract_skin_mask(
        image=image,
        detector=detector,
        forehead_offset_ratio=forehead_offset_ratio,
        raise_if_no_faces=raise_if_no_faces,
        out_mask=out_mask,
        out_result=out_result,
    )
    write_image(output_path, result)
    return output_path


def process_input_path(
    input_path: Path,
    output_path: Path,
    detector: LandmarkDetectorLike,
    forehead_offset_ratio: float = 0.25,
    raise_if_no_faces: bool = True,
    logger: logging.Logger | None = None,
) -> list[Path]:
    if input_path.is_file():
        image = read_image(input_path)
        target_output = _resolve_single_output_path(input_path=input_path, output_path=output_path)
        extract_skin_mask_to_file(
            image=image,
            detector=detector,
            output_path=target_output,
            forehead_offset_ratio=forehead_offset_ratio,
            raise_if_no_faces=raise_if_no_faces,
        )
        if logger is not None:
            logger.info("Processed 1/1 images: %s", target_output)
        return [target_output]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be file or directory: {input_path}")

    image_paths = collect_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in directory: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    total = len(image_paths)
    produced: list[Path] = []
    for index, image_path in enumerate(image_paths, start=1):
        image = read_image(image_path)
        target_output = output_path / image_path.name
        extract_skin_mask_to_file(
            image=image,
            detector=detector,
            output_path=target_output,
            forehead_offset_ratio=forehead_offset_ratio,
            raise_if_no_faces=raise_if_no_faces,
        )
        produced.append(target_output)
        if logger is not None:
            logger.info("Processed %d/%d images: %s", index, total, target_output)
    return produced


def extract_skin_mask(
    image: ImageBGR,
    detector: LandmarkDetectorLike,
    forehead_offset_ratio: float = 0.25,
    raise_if_no_faces: bool = True,
    out_mask: MaskU8 | None = None,
    out_result: ImageBGR | None = None,
) -> ImageBGR:
    validate_image_bgr(image)
    landmarks_batch = detector.detect(image)
    return extract_skin_mask_from_landmarks(
        image=image,
        landmarks_batch=landmarks_batch,
        forehead_offset_ratio=forehead_offset_ratio,
        raise_if_no_faces=raise_if_no_faces,
        out_mask=out_mask,
        out_result=out_result,
    )


def extract_skin_mask_from_landmarks(
    image: ImageBGR,
    landmarks_batch: LandmarksBatch,
    forehead_offset_ratio: float = 0.25,
    raise_if_no_faces: bool = True,
    out_mask: MaskU8 | None = None,
    out_result: ImageBGR | None = None,
) -> ImageBGR:
    validate_image_bgr(image)
    validate_landmarks_batch(landmarks_batch)

    if landmarks_batch.shape[0] == 0 and raise_if_no_faces:
        raise RuntimeError("No face detected on image.")

    height, width = image.shape[:2]
    mask = build_skin_mask(
        image_height=height,
        image_width=width,
        landmarks_batch=landmarks_batch,
        forehead_offset_ratio=forehead_offset_ratio,
        out_mask=out_mask,
    )

    if out_result is None:
        result = np.empty_like(image)
    else:
        validate_image_bgr(out_result)
        if out_result.shape != image.shape:
            raise ValueError(f"out_result shape {out_result.shape} does not match image shape {image.shape}")
        result = out_result

    validate_mask(mask, height, width)
    cv2.bitwise_and(image, image, dst=result, mask=mask)
    return result


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _resolve_single_output_path(input_path: Path, output_path: Path) -> Path:
    if output_path.exists() and output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / input_path.name

    if output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / input_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
