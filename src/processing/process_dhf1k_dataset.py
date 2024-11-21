import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing
from typing import List, Dict
from scipy.stats import gaussian_kde
from concurrent.futures import ProcessPoolExecutor

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from src.utils.file import get_paths_recursive
from src.config import (
    RAW_DHF1K_PATH,
    PROCESSED_DHF1K_PATH,
    RAW_EXPORTDATA_DHF1K_PATH,
    RAW_VIDEOS_DHF1K_PATH,
)

RAW_WIDTH = 1440
RAW_HEIGHT = 900
DHF1K_WIDTH = 640
DHF1K_HEIGHT = 360
N_USEC_IN_SEC = 1e6

DEFAULT_DISPERSION_THRESHOLD_PX = 25
DEFAULT_DURATION_THRESHOLD_MS = 100
DEFAULT_MIN_N_FIXATIONS = 5
DEFAULT_KDE_BANDWIDTH = 0.4
DEFAULT_TARGET_FPS = 5


def get_gaze_file_paths_dict() -> Dict[int, List[str]]:
    gaze_file_paths = get_paths_recursive(
        folder_path=RAW_EXPORTDATA_DHF1K_PATH,
        match_pattern="*.txt",
        file_type="f",
    )
    gaze_file_paths_dict = {}
    for gaze_file_path in gaze_file_paths:
        sample_id = int(gaze_file_path.split("/")[-1].split(".")[0][-3:])
        if sample_id not in gaze_file_paths_dict:
            gaze_file_paths_dict[sample_id] = []
        gaze_file_paths_dict[sample_id].append(gaze_file_path)

    return gaze_file_paths_dict


def get_saliency_map(
    fixation_data: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    kde_bandwidth: float,
    min_n_fixations: int,
) -> pd.DataFrame:
    if kde_bandwidth <= 0:
        raise ValueError(
            "❌ The bandwidth for the kernel density estimation must be greater than zero."
        )
    if min_n_fixations <= 0:
        raise ValueError(
            "❌ The minimum number of fixations required to generate a saliency map must be greater than zero."
        )

    fixation_data = fixation_data.copy()
    fixation_data["FrameId"] = fixation_data["TimeSinceStart_us"] // N_USEC_IN_SEC
    fixation_data = fixation_data[fixation_data["FrameId"] >= start_frame]
    fixation_data = fixation_data[fixation_data["FrameId"] < end_frame]

    # Get ground truth distribution
    x_coords = fixation_data["X_px"].values
    y_coords = fixation_data["Y_px"].values

    # Return an empty saliency map if no fixations are present
    if len(x_coords) < min_n_fixations or len(y_coords) < min_n_fixations:
        return np.zeros((DHF1K_HEIGHT, DHF1K_WIDTH))

    # Perform kernel density estimation
    positions = np.vstack([x_coords, y_coords])
    kde = gaussian_kde(positions, bw_method=kde_bandwidth)
    x_grid, y_grid = np.meshgrid(np.arange(DHF1K_WIDTH), np.arange(DHF1K_HEIGHT))
    grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    saliency_values = kde(grid_positions).reshape(DHF1K_HEIGHT, DHF1K_WIDTH)

    # Normalize the saliency map
    saliency_map = (saliency_values - saliency_values.min()) / (
        saliency_values.max() - saliency_values.min()
    )

    return saliency_map


def process_sample(
    sample_id: int,
    gaze_file_paths: str,
    min_n_fixations: int,
    kde_bandwidth: float,
    target_fps: int,
) -> None:
    # Get fixations
    fixation_data = None
    for gaze_file_path in gaze_file_paths:
        subject_df = pd.read_csv(gaze_file_path, sep="\t")
        subject_df = subject_df[subject_df["L Event Info"] == "Fixation"]
        subject_df = subject_df[["Time", "L POR X [px]", "L POR Y [px]"]]
        subject_df = subject_df.dropna()
        start_timestamp = subject_df["Time"].min()
        subject_df["TimeSinceStart_us"] = subject_df["Time"] - start_timestamp
        subject_df = subject_df.rename(
            columns={
                "L POR X [px]": "X_px",
                "L POR Y [px]": "Y_px",
            }
        )
        # TODO: check correct scaling
        subject_df["X_px"] = subject_df["X_px"].clip(0, RAW_WIDTH)
        subject_df["Y_px"] = subject_df["Y_px"].clip(0, RAW_HEIGHT)
        subject_df["X_px"] = subject_df["X_px"] / RAW_WIDTH * DHF1K_WIDTH
        subject_df["Y_px"] = subject_df["Y_px"] / RAW_HEIGHT * DHF1K_HEIGHT
        subject_df = subject_df[["TimeSinceStart_us", "X_px", "Y_px"]]

        if fixation_data is None:
            fixation_data = subject_df
        else:
            fixation_data = pd.concat([fixation_data, subject_df])

    # Get video duration
    video_file_path = f"{RAW_VIDEOS_DHF1K_PATH}/{sample_id:03d}.avi"
    video = cv2.VideoCapture(video_file_path)
    original_fps = int(video.get(cv2.CAP_PROP_FPS))
    duration_n_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // original_fps
    if duration_n_frame == 0:
        print(
            f"⚠️ Video {video_file_path} lasts less than 1 second: could not generate saliency maps."
        )
        return

    # Get and save frames
    dst_frame_file_path = f"{PROCESSED_DHF1K_PATH}/{sample_id}/frames"
    os.makedirs(dst_frame_file_path, exist_ok=True)

    frame_step = original_fps // target_fps
    frame_count = 0
    second_count = 0
    frame_in_second = 1
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            frame = cv2.resize(frame, (DHF1K_WIDTH, DHF1K_HEIGHT))
            frame_file_path = (
                f"{dst_frame_file_path}/{second_count}_{frame_in_second}.jpg"
            )
            cv2.imwrite(frame_file_path, frame)

            frame_in_second += 1
            if frame_in_second > target_fps:
                second_count += 1
                frame_in_second = 1

        frame_count += 1
    video.release()

    # Get ground truths
    saliency_maps = [
        get_saliency_map(
            fixation_data=fixation_data,
            start_frame=start_frame,
            end_frame=start_frame + 1,
            kde_bandwidth=kde_bandwidth,
            min_n_fixations=min_n_fixations,
        )
        for start_frame in range(0, duration_n_frame)
    ]
    ground_truths = (np.array(saliency_maps) * 255).astype(np.uint8)

    # Save ground truths
    dst_ground_truths_file_paths = [
        f"{PROCESSED_DHF1K_PATH}/{sample_id}/ground_truths/ground_truth_{i}.jpg"
        for i in range(ground_truths.shape[0])
    ]
    os.makedirs(os.path.dirname(dst_ground_truths_file_paths[0]), exist_ok=True)
    for i, ground_truth in enumerate(ground_truths):
        Image.fromarray(ground_truth).save(dst_ground_truths_file_paths[i])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the DHF1K dataset")

    parser.add_argument(
        "--min-n-fixations",
        "-n",
        type=int,
        default=DEFAULT_MIN_N_FIXATIONS,
        help="The minimum number of fixations required to generate a saliency map.",
    )
    parser.add_argument(
        "--kde-bandwidth",
        "-b",
        type=float,
        default=DEFAULT_KDE_BANDWIDTH,
        help="The bandwidth for the kernel density estimation.",
    )
    parser.add_argument(
        "--target-fps",
        "-f",
        type=int,
        default=DEFAULT_TARGET_FPS,
        help="The target frames per second for the processed videos.",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    min_n_fixations = args.min_n_fixations
    kde_bandwidth = args.kde_bandwidth
    target_fps = args.target_fps

    # Delete exsiting processed DHF1K data
    if os.path.exists(PROCESSED_DHF1K_PATH):
        shutil.rmtree(PROCESSED_DHF1K_PATH)
        print("✅ Deleted existing processed DHF1K data.")

    # Get gaze file paths for each sample
    gaze_file_paths_dict = get_gaze_file_paths_dict()

    # Process fixations
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args = [
            (
                sample_id,
                gaze_file_paths,
                min_n_fixations,
                kde_bandwidth,
                target_fps,
            )
            for sample_id, gaze_file_paths in gaze_file_paths_dict.items()
        ]
        futures = executor.map(process_sample, *zip(*args))
        for _ in tqdm(
            futures, total=len(gaze_file_paths_dict), desc="Processing fixations..."
        ):
            pass


if __name__ == "__main__":
    main()
