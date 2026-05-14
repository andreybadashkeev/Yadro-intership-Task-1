from __future__ import annotations

import argparse
import logging
from pathlib import Path

from landmarks_dlib import DlibLandmarkDetector
from pipeline import process_input_path


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract facial skin mask using dlib landmarks.")
    parser.add_argument("--input", default=Path("data/input"), type=Path, help="Path to input image or directory.")
    parser.add_argument("--output", default=Path("data/output"), type=Path, help="Path to output image or directory.")
    parser.add_argument(
        "--predictor",
        type=Path,
        default=Path("shape_predictor_68_face_landmarks.dat"),
        help="Path to dlib 68-landmark predictor file.",
    )
    parser.add_argument(
        "--forehead-offset-ratio",
        type=float,
        default=0.25,
        help="Fraction of face height used to expand mask to forehead.",
    )
    parser.add_argument(
        "--allow-empty-face",
        action="store_true",
        help="Do not fail if no faces are found (output will be empty mask).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args()

    detector = DlibLandmarkDetector(args.predictor)
    output_paths = process_input_path(
        input_path=args.input,
        output_path=args.output,
        detector=detector,
        forehead_offset_ratio=args.forehead_offset_ratio,
        raise_if_no_faces=not args.allow_empty_face,
        logger=LOGGER,
    )
    LOGGER.info("Skin mask extraction finished successfully. Total outputs: %d", len(output_paths))


if __name__ == "__main__":
    main()