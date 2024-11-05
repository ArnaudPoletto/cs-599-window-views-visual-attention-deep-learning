import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from typing import List

from src.utils.file import get_files_recursive, get_ids_from_file_path, get_set_str
from src.config import SAMPLES_PATH, SETS_PATH

DEFAULT_VIDEO_START_SEC = 5
DEFAULT_VIDEO_STOP_SEC = 60
DEFAULT_SAMPLE_FPS = 5
DEFAULT_SAMPLE_LENGTH = 5


def process_sample(
    experiment_id: int,
    set_id: int,
    scene_id: int,
    start_frame: int,
    end_frame: int,
    sample: List[np.ndarray],
) -> None:
    """
    Process the given sample.

    Args:
        experiment_id (int): The experiment ID.
        set_id (int): The set ID.
        scene_id (int): The scene ID.
        start_frame (int): The start frame.
        end_frame (int): The end frame.
        sample (List[np.ndarray]): The sample frames.
    """
    sample = np.array(sample)
    set_str = get_set_str(experiment_id, set_id)
    sample_path = f"{SAMPLES_PATH}/experiment{experiment_id}/{set_str}/scene{scene_id:02}/{start_frame}-{end_frame}.npy"

    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    np.save(sample_path, sample)


def process_video_samples(
    video_path: str,
    video_start_sec: int,
    video_stop_sec: int,
    sample_fps: int,
    sample_length: int,
) -> None:
    """
    Process video samples.

    Args:
        video_path (str): The path to the video file.
        video_start_sec (int): The minimum start time in seconds.
        video_stop_sec (int): The maximum stop time in seconds.
        sample_fps (int): The sample FPS.
        sample_length (int): The sample length.

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
    curr_sample = []
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

        curr_sample.append(frame)

        # Process sample if it is complete
        if len(curr_sample) == sample_length:
            process_sample(
                experiment_id=experiment_id,
                set_id=set_id,
                scene_id=scene_id,
                start_frame=curr_frame_id - (sample_length - 1) * sample_fps,
                end_frame=curr_frame_id + sample_fps,
                sample=curr_sample,
            )
            curr_sample = []

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
        default=DEFAULT_VIDEO_START_SEC,
        help="The minimum start time in seconds.",
    )
    parser.add_argument(
        "--video-stop",
        type=int,
        default=DEFAULT_VIDEO_STOP_SEC,
        help="The maximum stop time in seconds.",
    )
    parser.add_argument(
        "--sample-fps",
        type=int,
        default=DEFAULT_SAMPLE_FPS,
        help="The sample FPS.",
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=DEFAULT_SAMPLE_LENGTH,
        help="The sample length.",
    )

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    video_start_sec = args.video_start
    video_stop_sec = args.video_stop
    sample_fps = args.sample_fps
    sample_length = args.sample_length
    print(f"⚙️ Processing dataset sequences with the following parameters:")
    print(f"\t- Minimum start time: {video_start_sec} seconds")
    print(f"\t- Maximum stop time: {video_stop_sec} seconds")
    print(f"\t- Sample FPS: {sample_fps}")
    print(f"\t- Sample length: {sample_length} frames")

    # Delete existing samples, remove folder
    if os.path.exists(SAMPLES_PATH):
        shutil.rmtree(SAMPLES_PATH)
        print("✅ Deleted existing samples.")

    # Get video paths
    video_paths = get_files_recursive(
        folder_path=SETS_PATH,
        match_pattern="*.mp4",
    )

    # Process each video
    for video_path in tqdm(video_paths, desc="⌛ Processing video samples..."):
        process_video_samples(
            video_path=video_path,
            video_start_sec=video_start_sec,
            video_stop_sec=video_stop_sec,
            sample_fps=sample_fps,
            sample_length=sample_length,
        )
    print(f"✅ Processed {len(video_paths)} video samples.")


if __name__ == "__main__":
    main()
