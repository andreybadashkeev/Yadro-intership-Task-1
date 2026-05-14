from __future__ import annotations

from pathlib import Path

import cv2
import dlib
import numpy as np

from cv_types import ImageBGR, LandmarksBatch, validate_image_bgr


class DlibLandmarkDetector:
    def __init__(self, predictor_path: Path) -> None:
        if not predictor_path.exists():
            raise FileNotFoundError(f"Predictor file not found: {predictor_path}")
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(predictor_path))

    def detect(self, image_bgr: ImageBGR) -> LandmarksBatch:
        validate_image_bgr(image_bgr)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray)

        n_faces = len(faces)
        landmarks = np.empty((n_faces, 68, 2), dtype=np.int32)
        for i, face in enumerate(faces):
            shape = self._predictor(gray, face)
            for j, point in enumerate(shape.parts()):
                landmarks[i, j, 0] = point.x
                landmarks[i, j, 1] = point.y
        return landmarks
