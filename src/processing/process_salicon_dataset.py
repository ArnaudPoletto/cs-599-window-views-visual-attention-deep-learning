import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing
from scipy.io import loadmat
from typing import List, Dict, Optional
from scipy.stats import gaussian_kde
from concurrent.futures import ProcessPoolExecutor

from src.utils.file import get_paths_recursive
from src.config import (
    RAW_SALICON_GAZES_PATH,
    PROCESSED_SALICON_PATH,
    RAW_SALICON_IMAGES_PATH,
    RAW_SALICON_GROUND_TRUTHS_PATH,
)

SALICON_HEIGHT = 480
SALICON_WIDTH = 640
DURATION_N_FRAME = 5
N_MSEC_IN_SEC = 1000

DEFAULT_DISPERSION_THRESHOLD_PX = 25
DEFAULT_DURATION_THRESHOLD_MS = 100
DEFAULT_MIN_N_FIXATIONS = 5
DEFAULT_KDE_BANDWIDTH = 0.2


def get_dispersion(
    subject_df: pd.DataFrame, start_index: int, curr_index: int
) -> float:
    """
    Get the dispersion of the fixations.

    Args:
        subject_df (pd.DataFrame): The subject data.
        start_index (int): The start index.
        curr_index (int): The current index.

    Raises:
        ValueError: If the start index is less than zero.
        ValueError: If the current index is less than zero.
        ValueError: If the current index is less than the start index.
        ValueError: If the current index is greater than the length of the subject data.

    Returns:
        float: The dispersion.
    """
    if start_index < 0:
        raise ValueError("❌ The start index must be greater than zero.")
    if curr_index < 0:
        raise ValueError("❌ The current index must be greater than zero.")
    if curr_index < start_index:
        raise ValueError("❌ The current index must be greater than the start index.")
    if len(subject_df) < curr_index:
        raise ValueError(
            "❌ The current index must be less than the length of the subject data."
        )

    x_values = subject_df["X_px"].iloc[start_index:curr_index]
    y_values = subject_df["Y_px"].iloc[start_index:curr_index]

    x_max, x_min = x_values.max(), x_values.min()
    y_max, y_min = y_values.max(), y_values.min()

    x_dispersion = float(x_max - x_min)
    y_dispersion = float(y_max - y_min)
    total_dispersion = x_dispersion + y_dispersion

    return total_dispersion


def get_saliency_map(
    fixation_data: pd.DataFrame,
    start_frame: Optional[int],
    end_frame: Optional[int],
    kde_bandwidth: float,
    min_n_fixations: int,
    height: int = SALICON_HEIGHT,
    width: int = SALICON_WIDTH,
) -> np.ndarray:
    """
    Get the saliency map from the fixation data.

    Args:
        fixation_data (pd.DataFrame): The fixation data.
        start_frame (Optional[int]): The start frame.
        end_frame (Optional[int]): The end frame.
        kde_bandwidth (float): The bandwidth for the kernel density estimation.
        min_n_fixations (int): The minimum number of fixations required to generate a saliency map.
        width (int, optional): The width of the saliency map. Defaults to SALICON_WIDTH.
        height (int, optional): The height of the saliency map. Defaults to SALICON_HEIGHT.

    Returns:
        np.ndarray: The saliency map.

    Raises:
        ValueError: If the bandwidth for the kernel density estimation is less than or equal to zero.
        ValueError: If the minimum number of fixations required to generate a saliency map is less than or equal to zero.
        ValueError: If the width of the saliency map is less than or equal to zero.
        ValueError: If the height of the saliency map is less than or equal to zero.
    """
    if kde_bandwidth <= 0:
        raise ValueError(
            "❌ The bandwidth for the kernel density estimation must be greater than zero."
        )
    if min_n_fixations <= 0:
        raise ValueError(
            "❌ The minimum number of fixations required to generate a saliency map must be greater than zero."
        )
    if width <= 0:
        raise ValueError("❌ The width of the saliency map must be greater than zero.")
    if height <= 0:
        raise ValueError("❌ The height of the saliency map must be greater than zero.")

    fixation_data = fixation_data.copy()
    if "TimeSinceStart_ms" in fixation_data.columns:
        fixation_data["FrameId"] = fixation_data["TimeSinceStart_ms"] // N_MSEC_IN_SEC
        if start_frame is not None:
            fixation_data = fixation_data[fixation_data["FrameId"] >= start_frame]
        if end_frame is not None:
            fixation_data = fixation_data[fixation_data["FrameId"] < end_frame]
    elif start_frame is not None or end_frame is not None:
        print(
            "⚠️ Start and end frames are ignored as TimeSinceStart_ms colum is not present in the fixation data."
        )

    # Get ground truth distribution
    x_coords = fixation_data["X_px"].values
    y_coords = fixation_data["Y_px"].values

    # Return an empty saliency map if no fixations are present
    if len(x_coords) < min_n_fixations or len(y_coords) < min_n_fixations:
        return np.zeros((height, width))

    # Perform kernel density estimation
    positions = np.vstack([x_coords, y_coords])
    kde = gaussian_kde(positions, bw_method=kde_bandwidth)
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    saliency_values = kde(grid_positions).reshape(height, width)

    # Normalize the saliency map
    saliency_map = (saliency_values - saliency_values.min()) / (
        saliency_values.max() - saliency_values.min()
    )

    return saliency_map


def get_subject_fixation_data(
    subject_df: pd.DataFrame,
    group_start_timestamp: int,
    dispersion_threshold_px: int,
    duration_threshold_ms: int,
) -> List[Dict]:
    """
    Get the fixation data for a single subject.

    Args:
        subject_df (pd.DataFrame): The subject data.
        sample_id (int): The sample id.
        subject_id (int): The subject id.
        group_start_timestamp (int): The group start timestamp.
        dispersion_threshold_px (int): The dispersion threshold in pixels.
        duration_threshold_ms (int): The duration threshold in milliseconds.

    Returns:
        List[Dict]: The fixation data.

    Raises:
        ValueError: If the group start timestamp is less than zero.
        ValueError: If the dispersion threshold is less than zero.
        ValueError: If the duration threshold is less than zero.
    """
    if group_start_timestamp < 0:
        raise ValueError("❌ The group start timestamp must be greater than zero.")
    if dispersion_threshold_px < 0:
        raise ValueError("❌ The dispersion threshold must be greater than zero.")
    if duration_threshold_ms < 0:
        raise ValueError("❌ The duration threshold must be greater than zero.")

    # Process fixations for this subject
    subject_fixation_data = []
    start_index = 0
    is_fixation = False
    for curr_index in range(len(subject_df)):
        start_timestamp = subject_df["Timestamp_ms"].iloc[start_index]
        end_timestamp = subject_df["Timestamp_ms"].iloc[curr_index]
        fixation_duration = end_timestamp - start_timestamp
        time_since_start = start_timestamp - group_start_timestamp

        # Skip if the fixation duration is too short
        if fixation_duration < duration_threshold_ms:
            continue

        # Skip if the dispersion is too large
        dispersion = get_dispersion(subject_df, start_index, curr_index)
        if dispersion < dispersion_threshold_px and curr_index < len(subject_df) - 1:
            is_fixation = True
            continue

        # Add fixation data if it is a fixation at this point
        if is_fixation:
            subject_fixation_data.append(
                {
                    "TimeSinceStart_ms": time_since_start,
                    "X_px": subject_df["X_px"].iloc[start_index:curr_index].mean(),
                    "Y_px": subject_df["Y_px"].iloc[start_index:curr_index].mean(),
                }
            )
        start_index = curr_index
        is_fixation = False

    return subject_fixation_data


def is_already_processed(gaze_file_path: str) -> bool:
    gaze_file_name = os.path.basename(gaze_file_path)
    sample_id = int(gaze_file_name.split("_")[-1].split(".")[0])
    processed_path = f"{PROCESSED_SALICON_PATH}/{sample_id}"

    # Check if folder exists
    if not os.path.exists(processed_path):
        return False

    # Check if all files exist
    if not os.path.exists(f"{processed_path}/frame.jpg"):
        return False
    if not os.path.exists(f"{processed_path}/global_ground_truth.png"):
        return False
    if not os.path.exists(f"{processed_path}/global_ground_truth_from_fixations.jpg"):
        return False
    for i in range(DURATION_N_FRAME):
        if not os.path.exists(f"{processed_path}/ground_truth_{i}.jpg"):
            return False

    return True


def process_sample(
    frame_file_path: str,
    dispersion_threshold_px: int,
    duration_threshold_ms: int,
    min_n_fixations: int,
    kde_bandwidth: float,
) -> None:
    """
    Process a single sample.

    Args:
        image_file_path (str): The image file path.
        dispersion_threshold_px (int): The dispersion threshold in pixels.
        duration_threshold_ms (int): The duration threshold in milliseconds.
        min_n_fixations (int): The minimum number of fixations required to generate a saliency map.
        kde_bandwidth (float): The bandwidth for the kernel density estimation.
    """
    # Check if sample is already processed
    sample_id = int(frame_file_path.split("/")[-1].split(".")[0].split("_")[-1])
    output_folder_path = f"{PROCESSED_SALICON_PATH}/{sample_id}"
    if is_already_processed(output_folder_path):
        return

    # Write image to processed sample folder
    dst_frame_file_path = f"{output_folder_path}/frame.jpg"
    os.makedirs(os.path.dirname(dst_frame_file_path), exist_ok=True)
    frame = Image.open(frame_file_path).convert("RGB")
    frame.save(dst_frame_file_path)

    # Get gaze data
    gaze_file_path = f"{RAW_SALICON_GAZES_PATH}/{os.path.basename(frame_file_path).replace('.jpg', '.mat')}"
    if not os.path.exists(gaze_file_path):
        return
    mat_data = loadmat(gaze_file_path)
    if "gaze" not in mat_data:
        return
    gaze_data = mat_data["gaze"]

    # Process fixations
    fixation_data = []
    global_fixation_data = []
    for gaze_subject in gaze_data:
        # Get subject data an sort by timestamp
        gaze_subject = gaze_subject[0]
        subject_locations, subject_timestamps, subject_global_fixations = gaze_subject
        subject_df = pd.DataFrame(
            {
                "Timestamp_ms": subject_timestamps[:, 0],
                "X_px": subject_locations[:, 0],
                "Y_px": subject_locations[:, 1],
            }
        )
        subject_df = subject_df.sort_values(by="Timestamp_ms")
        group_start_timestamp = subject_df["Timestamp_ms"].min()

        # Process fixations for this subject
        subject_fixation_data = get_subject_fixation_data(
            subject_df=subject_df,
            group_start_timestamp=group_start_timestamp,
            dispersion_threshold_px=dispersion_threshold_px,
            duration_threshold_ms=duration_threshold_ms,
        )
        subject_global_fixation_data = [
            {"X_px": x, "Y_px": y} for x, y in subject_global_fixations
        ]
        fixation_data.extend(subject_fixation_data)
        global_fixation_data.extend(subject_global_fixation_data)

    if len(fixation_data) == 0:
        print(f"❌ No fixations found for sample {sample_id}.")
        return
    if len(global_fixation_data) == 0:
        print(f"❌ No global fixations found for sample {sample_id}.")
        return

    # Get ground truths and global ground truth from fixations
    fixation_data = pd.DataFrame(fixation_data)
    saliency_maps = [
        get_saliency_map(
            fixation_data=fixation_data,
            start_frame=start_frame,
            end_frame=start_frame + 1,
            kde_bandwidth=kde_bandwidth,
            min_n_fixations=min_n_fixations,
        )
        for start_frame in range(0, DURATION_N_FRAME)
    ]
    ground_truths = (np.array(saliency_maps) * 255).astype(np.uint8)
    global_fixation_data = pd.DataFrame(global_fixation_data)
    global_saliency_map = get_saliency_map(
        fixation_data=global_fixation_data,
        start_frame=None,
        end_frame=None,
        kde_bandwidth=kde_bandwidth,
        min_n_fixations=min_n_fixations,
    )
    global_ground_truth_from_fixations = (global_saliency_map * 255).astype(np.uint8)

    # Write global ground truth to processed sample folder
    dst_global_ground_truth_file_path = f"{output_folder_path}/global_ground_truth.jpg"
    os.makedirs(os.path.dirname(dst_global_ground_truth_file_path), exist_ok=True)
    global_ground_truth_file_path = f"{RAW_SALICON_GROUND_TRUTHS_PATH}/{os.path.basename(frame_file_path).replace('.jpg', '.png')}"
    global_ground_truth = Image.open(global_ground_truth_file_path).convert("L")
    global_ground_truth.save(dst_global_ground_truth_file_path)

    # Write global ground truth from fixations to processed sample folder
    dst_global_ground_truth_from_fixations_file_path = (
        f"{output_folder_path}/global_ground_truth_from_fixations.jpg"
    )
    os.makedirs(os.path.dirname(dst_global_ground_truth_from_fixations_file_path), exist_ok=True)
    Image.fromarray(global_ground_truth_from_fixations).save(
        dst_global_ground_truth_from_fixations_file_path
    )

    # Write ground truths to processed sample folder
    for i, ground_truth in enumerate(ground_truths):
        dst_ground_truth_file_path = f"{output_folder_path}/ground_truth_{i}.jpg"
        os.makedirs(os.path.dirname(dst_ground_truth_file_path), exist_ok=True)
        Image.fromarray(ground_truth).save(dst_ground_truth_file_path)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process the salicon dataset.")

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

    # Get gaze file paths for each sample
    image_file_paths = get_paths_recursive(
        folder_path=RAW_SALICON_IMAGES_PATH, match_pattern="*.jpg", path_type="f"
    )

    # Process fixations
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args = [
            (
                image_file_path,
                dispersion_threshold_px,
                duration_threshold_ms,
                min_n_fixations,
                kde_bandwidth,
            )
            for image_file_path in image_file_paths
        ]
        futures = executor.map(process_sample, *zip(*args))
        for _ in tqdm(
            futures, total=len(image_file_paths), desc="⌛ Processing fixations..."
        ):
            pass


if __name__ == "__main__":
    main()
