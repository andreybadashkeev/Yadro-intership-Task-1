import unittest

import numpy as np

from pipeline import extract_skin_mask_from_landmarks


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


class TestPipelineSmoke(unittest.TestCase):
    def test_pipeline_runs_with_landmarks_and_fills_out_buffers(self) -> None:
        image = np.full((180, 180, 3), 255, dtype=np.uint8)
        landmarks_batch = synthetic_landmarks()[None, ...]
        out_mask = np.empty((180, 180), dtype=np.uint8)
        out_result = np.empty_like(image)

        result = extract_skin_mask_from_landmarks(
            image=image,
            landmarks_batch=landmarks_batch,
            forehead_offset_ratio=0.25,
            raise_if_no_faces=True,
            out_mask=out_mask,
            out_result=out_result,
        )
        self.assertIs(result, out_result)
        self.assertGreater(int(np.count_nonzero(result)), 0)
        self.assertTrue(set(np.unique(out_mask).tolist()).issubset({0, 255}))

    def test_pipeline_raises_when_no_face_and_required(self) -> None:
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        empty_landmarks = np.empty((0, 68, 2), dtype=np.int32)
        with self.assertRaises(RuntimeError):
            extract_skin_mask_from_landmarks(
                image=image,
                landmarks_batch=empty_landmarks,
                forehead_offset_ratio=0.25,
                raise_if_no_faces=True,
            )


if __name__ == "__main__":
    unittest.main()
