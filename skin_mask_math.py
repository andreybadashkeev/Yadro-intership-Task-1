from __future__ import annotations

import cv2
import numpy as np

from cv_types import Landmarks68, LandmarksBatch, MaskU8, validate_landmarks_batch, validate_mask


def build_skin_mask(
    image_height: int,
    image_width: int,
    landmarks_batch: LandmarksBatch,
    forehead_offset_ratio: float = 0.25,
    out_mask: MaskU8 | None = None,
) -> MaskU8:
    if not (0.0 < forehead_offset_ratio <= 1.0):
        raise ValueError("forehead_offset_ratio must be in range (0, 1].")
    validate_landmarks_batch(landmarks_batch)

    if out_mask is None:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
    else:
        validate_mask(out_mask, image_height, image_width)
        out_mask.fill(0)
        mask = out_mask

    face_polygon = np.empty((27, 2), dtype=np.int32)
    eye_mouth_scratch = np.empty((12, 2), dtype=np.int32)

    for points in landmarks_batch:
        _fill_face_polygon(mask, points, forehead_offset_ratio, face_polygon)

        # Reuse one scratch buffer for small polygons to avoid short-lived allocations.
        eye_mouth_scratch[:6] = points[36:42]
        cv2.fillConvexPoly(mask, eye_mouth_scratch[:6], 0)

        eye_mouth_scratch[:6] = points[42:48]
        cv2.fillConvexPoly(mask, eye_mouth_scratch[:6], 0)

        eye_mouth_scratch[:, :] = points[48:60]
        cv2.fillPoly(mask, [eye_mouth_scratch], 0)

    return mask


def _fill_face_polygon(
    mask: MaskU8,
    points: Landmarks68,
    forehead_offset_ratio: float,
    out_polygon: np.ndarray,
) -> None:
    nose_bridge = points[27]
    chin = points[8]
    dy = int(chin[1]) - int(nose_bridge[1])
    face_height = abs(dy)
    forehead_offset = int(face_height * forehead_offset_ratio)

    out_polygon[:17] = points[:17]

    out_polygon[17:22] = points[26:21:-1]
    out_polygon[17:22, 1] -= forehead_offset

    out_polygon[22:27] = points[21:16:-1]
    out_polygon[22:27, 1] -= forehead_offset

    cv2.fillPoly(mask, [out_polygon], 255)
