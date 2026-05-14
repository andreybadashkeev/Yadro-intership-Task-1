import unittest

import numpy as np

from skin_mask_math import build_skin_mask


def build_synthetic_landmarks() -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.int32)

    # Jawline (0..16): wide lower ellipse.
    jaw_x = np.linspace(30, 170, 17, dtype=np.int32)
    jaw_y = np.array([140, 145, 150, 154, 157, 159, 160, 161, 162, 161, 160, 159, 157, 154, 150, 145, 140], dtype=np.int32)
    points[0:17, 0] = jaw_x
    points[0:17, 1] = jaw_y

    # Eyebrows (17..26).
    left_brow = np.array([[55, 92], [63, 88], [72, 86], [81, 88], [89, 92]], dtype=np.int32)
    right_brow = np.array([[111, 92], [119, 88], [128, 86], [137, 88], [145, 92]], dtype=np.int32)
    points[17:22] = left_brow
    points[22:27] = right_brow

    # Nose bridge top (27) and chin (8) for height estimate.
    points[27] = np.array([100, 100], dtype=np.int32)
    points[8] = np.array([100, 162], dtype=np.int32)

    # Left eye (36..41), right eye (42..47), mouth (48..59).
    points[36:42] = np.array(
        [[70, 112], [77, 109], [84, 112], [84, 118], [77, 121], [70, 118]],
        dtype=np.int32,
    )
    points[42:48] = np.array(
        [[116, 112], [123, 109], [130, 112], [130, 118], [123, 121], [116, 118]],
        dtype=np.int32,
    )
    points[48:60] = np.array(
        [
            [80, 132],
            [88, 128],
            [98, 126],
            [108, 128],
            [120, 132],
            [108, 138],
            [98, 140],
            [88, 138],
            [85, 132],
            [92, 130],
            [104, 130],
            [112, 132],
        ],
        dtype=np.int32,
    )

    return points


class TestSkinMaskMath(unittest.TestCase):
    def test_invalid_ratio_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_skin_mask(
                image_height=100,
                image_width=100,
                landmarks_batch=np.zeros((1, 68, 2), dtype=np.int32),
                forehead_offset_ratio=0.0,
            )

    def test_build_mask_returns_binary_mask_and_cuts_holes(self) -> None:
        landmarks = build_synthetic_landmarks()[None, ...]
        out_mask = np.empty((200, 200), dtype=np.uint8)
        mask = build_skin_mask(
            image_height=200,
            image_width=200,
            landmarks_batch=landmarks,
            forehead_offset_ratio=0.25,
            out_mask=out_mask,
        )
        self.assertIs(mask, out_mask)

        unique_values = set(np.unique(mask).tolist())
        self.assertTrue(unique_values.issubset({0, 255}))

        # A point inside face area is expected to be white.
        self.assertEqual(int(mask[150, 100]), 255)

        # Points inside eyes and mouth should be removed from mask.
        self.assertEqual(int(mask[115, 77]), 0)   # left eye center
        self.assertEqual(int(mask[115, 123]), 0)  # right eye center
        self.assertEqual(int(mask[132, 98]), 0)   # mouth center


if __name__ == "__main__":
    unittest.main()
