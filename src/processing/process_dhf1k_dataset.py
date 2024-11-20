import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import shutil
import argparse

from src.config import (
    DHF1K_PATH,
    RAW_DHF1K_PATH,
    PROCESSED_DHF1K_PATH,
)

SALICON_HEIGHT = 480
SALICON_WIDTH = 640
N_MSEC_IN_SEC = 1000

DEFAULT_DISPERSION_THRESHOLD_PX = 25
DEFAULT_DURATION_THRESHOLD_MS = 100
DEFAULT_MIN_N_FIXATIONS = 5
DEFAULT_KDE_BANDWIDTH = 0.4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the DHF1K dataset")

    parser.add_argument(
        "--dispersion_threshold_px",
        type=int,
        default=DEFAULT_DISPERSION_THRESHOLD_PX,
        help="The dispersion threshold in pixels.",
    )
    parser.add_argument(
        "--duration_threshold_ms",
        type=int,
        default=DEFAULT_DURATION_THRESHOLD_MS,
        help="The duration threshold in milliseconds.",
    )
    parser.add_argument(
        "--min_n_fixations",
        type=int,
        default=DEFAULT_MIN_N_FIXATIONS,
        help="The minimum number of fixations required to generate a saliency map.",
    )
    parser.add_argument(
        "--kde_bandwidth",
        type=float,
        default=DEFAULT_KDE_BANDWIDTH,
        help="The bandwidth for the kernel density estimation.",
    )

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    dispersion_threshold_px = args.dispersion_threshold_px
    duration_threshold_ms = args.duration_threshold_ms
    min_n_fixations = args.min_n_fixations
    kde_bandwidth = args.kde_bandwidth

    # Delete exsiting processed DHF1K data
    if os.path.exists(PROCESSED_DHF1K_PATH):
        shutil.rmtree(PROCESSED_DHF1K_PATH)


if __name__ == "__main__":
    main()