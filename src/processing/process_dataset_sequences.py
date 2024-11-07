import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import shutil
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from scipy.stats import gaussian_kde

from src.utils.sample_ground_truth import SampleGroundTruth
from src.utils.sample import Sample
from src.utils.file import get_files_recursive, get_ids_from_file_path, get_set_str
from src.config import (
    FPS,
    SETS_PATH,
    SAMPLES_PATH,
    FIXATION_DATA_PATH,
    RAW_GAZE_FRAME_WIDTH,
    RAW_GAZE_FRAME_HEIGHT,
    DEFAULT_DATASET_SAMPLE_FPS,
    DEFAULT_DATASET_SAMPLE_LENGTH,
    DEFAULT_DATASET_VIDEO_STOP_SEC,
    DEFAULT_DATASET_VIDEO_START_SEC,
    DEFAULT_DATASET_GROUND_TRUTH_KDE_BANDWIDTH,
)

N_NSEC_IN_SEC = 1e9


def process_sample(
    experiment_id: int,
    set_id: int,
    scene_id: int,
    start_frame: int,
    end_frame: int,
    frames: List[np.ndarray],
    next_frame: Optional[np.ndarray],
    ground_truth: SampleGroundTruth,
) -> None:
    """
    Process the given sample.

    Args:
        experiment_id (int): The experiment ID.
        set_id (int): The set ID.
        scene_id (int): The scene ID.
        start_frame (int): The start frame.
        end_frame (int): The end frame.
        frames (List[np.ndarray]): The frames.
        next_frame (np.ndarray): The next frame.
        ground_truth (np.ndarray): The ground truth.
    """
    sample = Sample(
        frames=frames, next_frame=next_frame, ground_truth=ground_truth
    )
    set_str = get_set_str(experiment_id, set_id)
    sample_path = f"{SAMPLES_PATH}/experiment{experiment_id}/{set_str}/scene{scene_id:02}/{start_frame}-{end_frame}.pkl"

    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    with open(sample_path, "wb") as f:
        pickle.dump(sample, f)


def get_sample_next_frame(
    video: cv2.VideoCapture, frame_step: int
) -> Optional[np.ndarray]:
    """
    Get the next image from the video.

    Args:
        video (cv2.VideoCapture): The video.
        frame_step (int): The frame step.

    Returns:
        np.ndarray: The next image.
    """
    curr_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
    next_pos = curr_pos + frame_step - 1
    video.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
    ret, next_frame = video.read()
    if ret:
        next_frame = next_frame
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_pos)
    else:
        next_frame = None

    return next_frame


def get_sample_ground_truth(
    fixation_data: pd.DataFrame,
    experiment_id: int,
    set_id: int,
    scene_id: int,
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
    kde_bandwidth: float,
) -> SampleGroundTruth:
    """
    Get the ground truth for the sample.
    
    Args:
        fixation_data (pd.DataFrame): The fixation data.
        experiment_id (int): The experiment ID.
        set_id (int): The set ID.
        scene_id (int): The scene ID.
        start_frame (int): The start frame.
        end_frame (int): The end frame.
        width (int): The width of the images.
        height (int): The height of the images.
        kde_bandwidth (float): The bandwidth for the kernel density estimation.
    """
    fixation_data = fixation_data.copy()
    # Filter out fixations not from the current scene
    fixation_data.rename(columns={"SequenceId": "SceneId"}, inplace=True)
    fixation_data = fixation_data[
        (fixation_data["ExperimentId"] == experiment_id)
        & (fixation_data["SetId"] == set_id)
        & (fixation_data["SceneId"] == scene_id)
    ]

    # Get fixation frame ids and filter out fixations outside the specified video range
    fixation_data["FrameId"] = fixation_data["TimeSinceStart_ns"] / N_NSEC_IN_SEC * FPS
    fixation_data = fixation_data[fixation_data["FrameId"] >= start_frame]
    fixation_data = fixation_data[fixation_data["FrameId"] < end_frame]

    # Scale the fixation coordinates
    fixation_data["X_px"] = fixation_data["X_px"] * width / RAW_GAZE_FRAME_WIDTH
    fixation_data["Y_px"] = fixation_data["Y_px"] * height / RAW_GAZE_FRAME_HEIGHT

    # Get ground truth distribution
    x_coords = fixation_data["X_px"].values
    y_coords = fixation_data["Y_px"].values

    # Return an empty saliency map if no fixations are present
    if len(x_coords) == 0 or len(y_coords) == 0:
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

def process_video_samples(
    video_path: str,
    fixation_data: pd.DataFrame,
    video_start_sec: int,
    video_stop_sec: int,
    sample_fps: int,
    sample_length: int,
    kde_bandwidth: float,
) -> None:
    """
    Process video samples.

    Args:
        video_path (str): The path to the video file.
        fixation_data (pd.DataFrame): The fixation data.
        video_start_sec (int): The minimum start time in seconds.
        video_stop_sec (int): The maximum stop time in seconds.
        sample_fps (int): The sample FPS.
        sample_length (int): The sample length.
        kde_bandwidth (float): The bandwidth for the kernel density estimation.

    Raises:
        ValueError: If the minimum start time is less than 0.
        ValueError: If the maximum stop time is less than or equal to the minimum start time.
        ValueError: If the sample FPS is less than or equal to 0.
        ValueError: If the sample FPS is greater than the video FPS.
        ValueError: If the sample FPS is not a factor of the video FPS.
    """
    if video_start_sec < 0:
        raise ValueError("❌ Minimum start time must be greater than or equal to 0.")
    if video_stop_sec <= video_start_sec:
        raise ValueError(
            "❌ Maximum stop time must be greater than the minimum start time."
        )
    if sample_fps <= 0:
        raise ValueError("❌ Sample FPS must be greater than 0.")

    # Read video and get fps
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if sample_fps > fps:
        raise ValueError("❌ Sample FPS must be less than the video FPS.")
    if fps % sample_fps != 0:
        raise ValueError(
            f"❌ Sample FPS must be a factor of the video FPS: current FPS = {fps}, sample FPS = {sample_fps}"
        )
    frame_step = int(fps / sample_fps)

    # Set video start frame
    video_start_frame = int(video_start_sec * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    # Iterate over frames
    experiment_id, set_id, scene_id = get_ids_from_file_path(video_path)
    frames = []
    curr_frame_id = video_start_frame
    while True:
        # Read frame
        ret, frame = video.read()
        if not ret:
            break

        # Break if past the stop time
        curr_time_sec = curr_frame_id / fps
        if curr_time_sec > video_stop_sec:
            break

        # Skip frame if not at the correct rate
        if curr_frame_id % frame_step != 0:
            curr_frame_id += 1
            continue

        frames.append(frame)

        # Check if the sample length has been reached
        if len(frames) == sample_length:
            start_frame = curr_frame_id - (sample_length - 1) * sample_fps
            end_frame = curr_frame_id

            # Get next image and ground truth
            next_frame = get_sample_next_frame(video=video, frame_step=frame_step)
            ground_truth = get_sample_ground_truth(
                fixation_data=fixation_data,
                experiment_id=experiment_id,
                set_id=set_id,
                scene_id=scene_id,
                start_frame=start_frame,
                end_frame=end_frame,
                width=frame.shape[1],
                height=frame.shape[0],
                kde_bandwidth=kde_bandwidth,
            )

            # Process the sample
            process_sample(
                experiment_id=experiment_id,
                set_id=set_id,
                scene_id=scene_id,
                start_frame=start_frame,
                end_frame=end_frame,
                frames=frames,
                next_frame=next_frame,
                ground_truth=ground_truth,
            )
            frames = []

        curr_frame_id += 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process dataset sequences.")

    parser.add_argument(
        "--video-start",
        type=int,
        default=DEFAULT_DATASET_VIDEO_START_SEC,
        help="The minimum start time in seconds.",
    )
    parser.add_argument(
        "--video-stop",
        type=int,
        default=DEFAULT_DATASET_VIDEO_STOP_SEC,
        help="The maximum stop time in seconds.",
    )
    parser.add_argument(
        "--sample-fps",
        type=int,
        default=DEFAULT_DATASET_SAMPLE_FPS,
        help="The sample FPS.",
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=DEFAULT_DATASET_SAMPLE_LENGTH,
        help="The sample length.",
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=DEFAULT_DATASET_GROUND_TRUTH_KDE_BANDWIDTH,
        help="The bandwidth for the kernel density estimation",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    video_start_sec = args.video_start
    video_stop_sec = args.video_stop
    sample_fps = args.sample_fps
    sample_length = args.sample_length
    kde_bandwidth = args.kde_bandwidth
    print(f"⚙️ Processing dataset sequences with the following parameters:")
    print(f"\t- Minimum start time: {video_start_sec} seconds")
    print(f"\t- Maximum stop time: {video_stop_sec} seconds")
    print(f"\t- Sample FPS: {sample_fps}")
    print(f"\t- Sample length: {sample_length} frames")
    print(f"\t- KDE bandwidth: {kde_bandwidth}")

    # Delete existing samples, remove folder
    if os.path.exists(SAMPLES_PATH):
        shutil.rmtree(SAMPLES_PATH)
        print("✅ Deleted existing samples.")

    # Get video paths
    video_paths = get_files_recursive(
        folder_path=SETS_PATH,
        match_pattern="*.mp4",
    )

    # Get and fixation data
    fixation_data = pd.read_csv(FIXATION_DATA_PATH)

    # Process each video
    for video_path in tqdm(video_paths, desc="⌛ Processing video samples..."):
        process_video_samples(
            video_path=video_path,
            fixation_data=fixation_data,
            video_start_sec=video_start_sec,
            video_stop_sec=video_stop_sec,
            sample_fps=sample_fps,
            sample_length=sample_length,
            kde_bandwidth=kde_bandwidth,
        )
    print(f"✅ Processed {len(video_paths)} video samples.")


if __name__ == "__main__":
    main()
