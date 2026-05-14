import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline import collect_images, process_input_path


def synthetic_landmarks() -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.int32)
    points[0:17] = np.array(
        [
            [30, 120], [38, 125], [46, 130], [54, 134], [62, 137], [70, 139], [78, 140], [86, 141], [94, 142],
            [102, 141], [110, 140], [118, 139], [126, 137], [134, 134], [142, 130], [150, 125], [158, 120],
        ],
        dtype=np.int32,
    )
    points[17:22] = np.array([[58, 84], [66, 80], [74, 79], [82, 80], [90, 84]], dtype=np.int32)
    points[22:27] = np.array([[110, 84], [118, 80], [126, 79], [134, 80], [142, 84]], dtype=np.int32)
    points[27] = np.array([100, 92], dtype=np.int32)
    points[8] = np.array([94, 142], dtype=np.int32)
    points[36:42] = np.array([[68, 104], [74, 102], [80, 104], [80, 110], [74, 112], [68, 110]], dtype=np.int32)
    points[42:48] = np.array([[118, 104], [124, 102], [130, 104], [130, 110], [124, 112], [118, 110]], dtype=np.int32)
    points[48:60] = np.array(
        [[78, 124], [86, 120], [96, 118], [106, 120], [116, 124], [106, 130], [96, 132], [86, 130], [82, 124], [90, 122], [102, 122], [110, 124]],
        dtype=np.int32,
    )
    return points


class FakeDetector:
    def __init__(self, landmarks_batch: np.ndarray) -> None:
        self._landmarks_batch = landmarks_batch
        self.call_count = 0

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return self._landmarks_batch


class TestBatchPaths(unittest.TestCase):
    def test_collect_images_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "a.jpg").write_bytes(b"")
            (base / "b.jpeg").write_bytes(b"")
            (base / "c.png").write_bytes(b"")
            (base / "d.bmp").write_bytes(b"")

            collected = collect_images(base)
            self.assertEqual([path.name for path in collected], ["a.jpg", "b.jpeg", "c.png"])

    def test_process_directory_outputs_same_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "input"
            output_dir = base / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            image = np.full((180, 180, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(input_dir / "one.jpg"), image)
            cv2.imwrite(str(input_dir / "two.png"), image)

            detector = FakeDetector(synthetic_landmarks()[None, ...])
            produced = process_input_path(
                input_path=input_dir,
                output_path=output_dir,
                detector=detector,
                forehead_offset_ratio=0.25,
                raise_if_no_faces=True,
            )

            self.assertEqual([path.name for path in produced], ["one.jpg", "two.png"])
            self.assertTrue((output_dir / "one.jpg").exists())
            self.assertTrue((output_dir / "two.png").exists())
            self.assertEqual(detector.call_count, 2)


if __name__ == "__main__":
    unittest.main()
