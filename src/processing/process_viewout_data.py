import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from scipy.stats import gaussian_kde
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from src.utils.file import get_set_str
from src.utils.file import get_paths_recursive
from src.config import (
    SETS_PATH,
    IMAGE_SIZE,
    VIEWOUT_PATH,
    RAW_VIEWOUT_PATH,
    VIEWOUT_FRAME_WIDTH,
    VIEWOUT_FRAME_HEIGHT,
    PROCESSED_VIEWOUT_PATH,
)

OUTLIER_VALUES = (3000, 1500)
BOTTOM_SCREEN_THRESHOLD_SC = 0.99
MAX_TIME_SINCE_START_SEC = 120
RESAMPLING_RATE = "25ms"  # 40 Hz

N_HNSEC_IN_NSEC = 100
N_NSEC_IN_SEC = 1e9

DEFAULT_DISPERSION_THRESHOLD_PX = 100
DEFAULT_DURATION_THRESHOLD_NS = 100
DEFAULT_MIN_N_FIXATIONS = 5
DEFAULT_KDE_BANDWIDTH = 0.2
DEFAULT_TARGET_FPS = 5
DEFAULT_START_TIME_S = 0
DEFAULT_END_TIME_S = 60

VIEWOUT_WIDTH = 6144
VIEWOUT_HEIGHT = 3072


def get_raw_data() -> pd.DataFrame:
    # Get valid source file paths
    file_paths = get_paths_recursive(
        folder_path=RAW_VIEWOUT_PATH,
        match_pattern="Exp[12]_[12][0-9][0-9][12]_*.csv",
        path_type="f",
        recursive=True,
    )
    n_files = len(file_paths)

    # Check if all files were found
    all_file_paths = get_paths_recursive(
        folder_path=RAW_VIEWOUT_PATH,
        match_pattern="*.csv",
        path_type="f",
        recursive=True,
    )
    ignored_files = set(all_file_paths) - set(file_paths)
    if len(ignored_files) > 0:
        print(
            f"➡️  Found {n_files} raw gaze data files, ignoring the following {len(ignored_files)} file(s):"
        )
        for ignored_file in ignored_files:
            print(f"\t - {ignored_file}")
    else:
        print(f"➡️  Found {n_files} raw gaze data files.")

    # Read raw gaze data
    raw_data_list = []
    for file_path in tqdm(file_paths, total=n_files, desc="⌛ Reading raw gaze data"):
        raw_file_data = pd.read_csv(file_path, sep=";")
        raw_data_list.append(raw_file_data)

    raw_data = pd.concat(raw_data_list, axis=0, ignore_index=True)

    return raw_data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    print("⌛ Processing raw gaze data.")
    data = data.copy()

    # Delete entries with NaN values
    data = data.dropna()

    # Delete false center gaze points
    data = data[
        (data["GazeX"] != OUTLIER_VALUES[0]) & (data["GazeY"] != OUTLIER_VALUES[1])
    ]

    # Rescale gaze coordinates because they only go up to 6000, 3000
    max_gaze_x = data["GazeX"].max()
    max_gaze_y = data["GazeY"].max()
    data["GazeX"] = data["GazeX"] * (VIEWOUT_FRAME_WIDTH / max_gaze_x)
    data["GazeY"] = data["GazeY"] * (VIEWOUT_FRAME_HEIGHT / max_gaze_y)

    # Add gaze screen coordinates column and rename gaze pixel coordinates column
    data["X_sc"] = data["GazeX"] / VIEWOUT_FRAME_WIDTH
    data["Y_sc"] = data["GazeY"] / VIEWOUT_FRAME_HEIGHT
    data = data.rename(columns={"GazeX": "X_px", "GazeY": "Y_px"})

    # Remove gaze entries at the bottom of the screen since they are overrepresented
    data = data[data["Y_sc"] < BOTTOM_SCREEN_THRESHOLD_SC]

    # Get experiment, session, set and participant ids
    data["ExperimentId"] = data["Id"] // 1000  # Is the thousands digit
    data["SessionId"] = data["Id"] % 10  # Is the unit digit
    data["ParticipantId"] = (data["Id"] % 1000) // 10  # Is the hundreds and tens digit
    data["SetId"] = data["SequenceSet"]

    # Change timestamp unit to nanoseconds
    data["Timestamp"] = data["Timestamp"].astype("int64")
    data["Timestamp_ns"] = data["Timestamp"] * N_HNSEC_IN_NSEC

    # Delete entries with invalid ids
    data = data[
        ((data["ExperimentId"] == 1) | (data["ExperimentId"] == 2))
        & ((data["SessionId"] == 1) | (data["SessionId"] == 2))
        & ((data["SetId"] == 0) | (data["SetId"] == 1))
    ]

    # Add time since start column
    grouped_data = data.groupby(
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"]
    )
    data["TimeSinceStart_ns"] = grouped_data["Timestamp_ns"].transform(
        lambda x: x - x.min()
    )

    # Delete entries recorded after a long time
    data = data[data["TimeSinceStart_ns"] <= MAX_TIME_SINCE_START_SEC * N_NSEC_IN_SEC]

    # Delete outlier participants
    data = data[
        ~(
            (data["ExperimentId"] == 1)
            & (data["SessionId"] == 1)
            & data["ParticipantId"].isin([2, 9, 30])
        )
    ]
    data = data[
        ~(
            (data["ExperimentId"] == 2)
            & (data["SessionId"] == 1)
            & data["ParticipantId"].isin([23])
        )
    ]

    # Delete vector gaze information and id
    data = data.drop(
        columns=[
            "VectorGazeX",
            "VectorGazeY",
            "VectorGazeZ",
            "Id",
            "SequenceSet",
            "Timestamp",
        ]
    )

    # Convert types
    data = data.astype(
        {
            "ExperimentId": "int",
            "SessionId": "int",
            "ParticipantId": "int",
            "SequenceId": "int",
            "SetId": "int",
            "X_sc": "float32",
            "Y_sc": "float32",
            "X_px": "float32",
            "Y_px": "float32",
            "Timestamp_ns": "int64",
            "TimeSinceStart_ns": "int64",
        }
    )

    # Reorder columns
    data = data[
        [
            "ExperimentId",
            "SessionId",
            "ParticipantId",
            "SequenceId",
            "SetId",
            "X_sc",
            "Y_sc",
            "X_px",
            "Y_px",
            "Timestamp_ns",
            "TimeSinceStart_ns",
        ]
    ]

    print("✅ Gaze data processed.")

    return data


def get_interpolated_data(
    data: pd.DataFrame,
    resampling_rate: str = "50ms",
) -> pd.DataFrame:
    """
    Get interpolated gaze data.

    Args:
        data (pd.DataFrame): The gaze data.

    Returns:
        pd.DataFrame: The interpolated gaze data.
    """
    data = data.copy()

    # The conversion does not give good date, but the time unit is correct
    data["DateTime"] = pd.to_datetime(data["Timestamp_ns"], unit="ns")
    data = data.groupby(
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"]
    )
    groups = [data.get_group(x) for x in data.groups]

    for i, group in tqdm(
        enumerate(groups), total=len(groups), desc="⌛ Interpolating gaze data"
    ):
        group = group.copy()
        group = group.set_index("DateTime")

        # Resample and interpolate the data
        columns_to_interpolate = ["X_sc", "Y_sc", "X_px", "Y_px"]
        resampled_group = group[columns_to_interpolate].resample(resampling_rate).mean()
        interpolated_group = resampled_group.interpolate(method="linear")
        interpolated_group = interpolated_group.reset_index()

        # Add group information
        interpolated_group["ExperimentId"] = group["ExperimentId"].iloc[0]
        interpolated_group["SessionId"] = group["SessionId"].iloc[0]
        interpolated_group["ParticipantId"] = group["ParticipantId"].iloc[0]
        interpolated_group["SequenceId"] = group["SequenceId"].iloc[0]
        interpolated_group["SetId"] = group["SetId"].iloc[0]
        interpolated_group["Timestamp_ns"] = interpolated_group["DateTime"].astype(
            "int64"
        )
        start_timestamp = interpolated_group["Timestamp_ns"].min()
        interpolated_group["TimeSinceStart_ns"] = (
            interpolated_group["Timestamp_ns"] - start_timestamp
        )

        groups[i] = interpolated_group

    interpolated_data = pd.concat(groups, axis=0, ignore_index=True)

    # Reformat data
    interpolated_data = interpolated_data.drop(columns=["DateTime"])
    interpolated_data = interpolated_data.astype(
        {
            "ExperimentId": "int",
            "SessionId": "int",
            "ParticipantId": "int",
            "SequenceId": "int",
            "SetId": "int",
            "X_sc": "float32",
            "Y_sc": "float32",
            "X_px": "float32",
            "Y_px": "float32",
            "Timestamp_ns": "int64",
            "TimeSinceStart_ns": "int64",
        }
    )

    interpolated_data = interpolated_data[
        [
            "ExperimentId",
            "SessionId",
            "ParticipantId",
            "SequenceId",
            "SetId",
            "X_sc",
            "Y_sc",
            "X_px",
            "Y_px",
            "Timestamp_ns",
            "TimeSinceStart_ns",
        ]
    ]

    return interpolated_data


def get_fixation_data_from_group(
    group: pd.DataFrame, dispersion_threshold_px: float, duration_threshold_ns: float
) -> List[Dict[str, Any]]:
    """
    Get fixation data from a single sequence of gaze data.

    Args:
        group (pd.DataFrame): The group of gaze data.
        dispersion_threshold_px (float): The dispersion threshold for fixation detection in pixels.
        duration_threshold_ns (float): The duration threshold for fixation detection in nanoseconds.

    Returns:
        List[Dict[str, Any]]: The fixation data.
    """
    # Sort the group by timestamp
    group = group.sort_values(by="Timestamp_ns")
    group_start_timestamp = group["Timestamp_ns"].min()

    fixation_data = []
    start_index = 0
    is_fixation = False
    for curr_index in range(len(group)):
        # Set window to cover the duration threshold
        start_timestamp = group["Timestamp_ns"].iloc[start_index]
        end_timestamp = group["Timestamp_ns"].iloc[curr_index]
        fixation_duration = end_timestamp - start_timestamp
        time_since_start = start_timestamp - group_start_timestamp
        if fixation_duration < duration_threshold_ns:
            continue

        dispersion = (
            group["X_px"].iloc[start_index:curr_index].max()
            - group["X_px"].iloc[start_index:curr_index].min()
            + group["Y_px"].iloc[start_index:curr_index].max()
            - group["Y_px"].iloc[start_index:curr_index].min()
        )

        # Define the window as a fixation if it does not exceed the dispersion threshold, and increase the size of the window
        if dispersion < dispersion_threshold_px and curr_index < len(group) - 1:
            is_fixation = True
            continue

        # If the threshold is exceeded, save the fixation point and start over with next points in the time-series
        if is_fixation:
            fixation_data.append(
                {
                    "ExperimentId": group["ExperimentId"].iloc[start_index],
                    "SessionId": group["SessionId"].iloc[start_index],
                    "ParticipantId": group["ParticipantId"].iloc[start_index],
                    "SequenceId": group["SequenceId"].iloc[start_index],
                    "SetId": group["SetId"].iloc[start_index],
                    "X_sc": group["X_sc"].iloc[start_index:curr_index].mean(),
                    "Y_sc": group["Y_sc"].iloc[start_index:curr_index].mean(),
                    "X_px": group["X_px"].iloc[start_index:curr_index].mean(),
                    "Y_px": group["Y_px"].iloc[start_index:curr_index].mean(),
                    "StartTimestamp_ns": start_timestamp,
                    "EndTimestamp_ns": end_timestamp,
                    "Duration_ns": fixation_duration,
                    "TimeSinceStart_ns": time_since_start,
                }
            )
        start_index = curr_index
        is_fixation = False

    return fixation_data


def get_fixation_data(
    data: pd.DataFrame,
    dispersion_threshold_px: float,
    duration_threshold_ns: float,
) -> pd.DataFrame:
    """
    Get fixation data from the gaze data.

    Args:
        data (pd.DataFrame): The gaze data.
        dispersion_threshold_px (float): The dispersion threshold for fixation detection in pixels.
        duration_threshold_ns (float): The duration threshold for fixation detection in nanoseconds.

    Returns:
        pd.DataFrame: The fixation data.
    """
    # Group the data by sequence
    data = data.copy()
    data = data.groupby(
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"]
    )
    groups = [data.get_group(group) for group in data.groups]

    # Iterate through the data to get fixations
    fixation_data = []
    for group in tqdm(groups, total=len(groups), desc="⌛ Getting fixation data"):
        group_fixation_data = get_fixation_data_from_group(
            group, dispersion_threshold_px, duration_threshold_ns
        )
        fixation_data.extend(group_fixation_data)
    fixation_data = pd.DataFrame(fixation_data)

    # Reformat data
    fixation_data = fixation_data.astype(
        {
            "ExperimentId": "int",
            "SessionId": "int",
            "ParticipantId": "int",
            "SequenceId": "int",
            "SetId": "int",
            "X_sc": "float32",
            "Y_sc": "float32",
            "X_px": "float32",
            "Y_px": "float32",
            "StartTimestamp_ns": "int64",
            "EndTimestamp_ns": "int64",
            "Duration_ns": "int64",
            "TimeSinceStart_ns": "int64",
        }
    )

    fixation_data = fixation_data[
        [
            "ExperimentId",
            "SessionId",
            "ParticipantId",
            "SequenceId",
            "SetId",
            "X_sc",
            "Y_sc",
            "X_px",
            "Y_px",
            "StartTimestamp_ns",
            "EndTimestamp_ns",
            "Duration_ns",
            "TimeSinceStart_ns",
        ]
    ]

    return fixation_data


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
    fixation_data["FrameId"] = fixation_data["TimeSinceStart_ns"] // N_NSEC_IN_SEC
    fixation_data = fixation_data[fixation_data["FrameId"] >= start_frame]
    fixation_data = fixation_data[fixation_data["FrameId"] < end_frame]

    # Scale coordinates and get ground truth distribution
    fixation_data["X_px"] = fixation_data["X_px"] / VIEWOUT_WIDTH * IMAGE_SIZE
    fixation_data["Y_px"] = fixation_data["Y_px"] / VIEWOUT_HEIGHT * IMAGE_SIZE
    x_coords = fixation_data["X_px"].values
    y_coords = fixation_data["Y_px"].values

    # Return an empty saliency map if no fixations are present
    if len(x_coords) < min_n_fixations or len(y_coords) < min_n_fixations:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    # Perform kernel density estimation
    positions = np.vstack([x_coords, y_coords])
    kde = gaussian_kde(positions, bw_method=kde_bandwidth)
    x_grid, y_grid = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))
    grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    saliency_values = kde(grid_positions).reshape(IMAGE_SIZE, IMAGE_SIZE)

    # Normalize the saliency map
    saliency_map = (saliency_values - saliency_values.min()) / (
        saliency_values.max() - saliency_values.min()
    )

    return saliency_map


def process_sequence(
    data_keys: Tuple[int, int, int],
    data_values: pd.DataFrame,
    dispersion_threshold_px: int,
    duration_threshold_ms: int,
    min_n_fixations: int,
    kde_bandwidth: float,
    target_fps: int,
    start_time_s: int,
    end_time_s: int,
) -> None:
    # Get ids and create sample folder
    experiment_id, sequence_id, set_id = data_keys
    set_str = get_set_str(experiment_id=experiment_id, set_id=set_id)
    sample_folder_path = f"{PROCESSED_VIEWOUT_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02d}"
    os.makedirs(sample_folder_path, exist_ok=True)

    # Get input modality
    modality_type = "image" if (experiment_id == 1 and set_str == "images") else "video"
    modality_path = (
        f"{SETS_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02d}"
    )
    modality_path += ".png" if modality_type == "image" else ".mp4"
    if modality_type == "video":
        # Create frames folder
        frames_path = f"{sample_folder_path}/frames"
        os.makedirs(frames_path, exist_ok=True)

        # Open video and process frames
        video = cv2.VideoCapture(modality_path)
        video_fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_interval = video_fps // target_fps
        video_duration_s = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // video_fps
        start_frame = start_time_s * video_fps
        end_time_s = min(end_time_s, video_duration_s)
        end_frame = end_time_s * video_fps
        frame_count = 0
        while True:
            ret, frame = video.read()

            # End of video or end of sequence
            if not ret or frame_count >= end_frame:
                break

            # Skip frames before start time
            if frame_count < start_frame:
                frame_count += 1
                continue

            # Save frame every frame interval
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                second_count = frame_count // video_fps
                second_frame_count = (frame_count % video_fps) // frame_interval
                frame_file_path_dst = (
                    f"{frames_path}/{second_count:05d}_{second_frame_count:05d}.jpg"
                )
                cv2.imwrite(frame_file_path_dst, frame)

            frame_count += 1
        video.release()
    else:
        frame_file_path_dst = f"{sample_folder_path}/frame.png"
        frame = cv2.imread(modality_path)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(frame_file_path_dst, frame)

    # Get ground truths
    saliency_maps = [
        get_saliency_map(
            fixation_data=data_values,
            start_frame=start_frame,
            end_frame=start_frame + 1,
            kde_bandwidth=kde_bandwidth,
            min_n_fixations=min_n_fixations,
        )
        for start_frame in range(start_time_s, end_time_s)
    ]
    ground_truths = (np.array(saliency_maps) * 255).astype(np.uint8)

    # Save ground truths
    ground_truths_folder_path = f"{sample_folder_path}/ground_truths"
    os.makedirs(ground_truths_folder_path, exist_ok=True)
    for i, ground_truth in enumerate(ground_truths):
        cv2.imwrite(
            f"{ground_truths_folder_path}/ground_truth_{i:05d}.jpg", ground_truth
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process raw gaze data.")
    parser.add_argument(
        "--dispersion-threshold-px",
        "-dit",
        type=float,
        default=DEFAULT_DISPERSION_THRESHOLD_PX,
        help="The dispersion threshold for fixation detection in pixels.",
    )
    parser.add_argument(
        "--duration-threshold-ns",
        "-dut",
        type=float,
        default=DEFAULT_DURATION_THRESHOLD_NS,
        help="The duration threshold for fixation detection in nanoseconds.",
    )
    parser.add_argument(
        "--min-n-fixations",
        type=int,
        default=DEFAULT_MIN_N_FIXATIONS,
        help="The minimum number of fixations required to generate a saliency map.",
    )
    parser.add_argument(
        "--kde-bandwidth",
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
    parser.add_argument(
        "--start-time-s",
        type=int,
        default=DEFAULT_START_TIME_S,
        help="The start time in seconds for the processed videos.",
    )
    parser.add_argument(
        "--end-time-s",
        type=int,
        default=DEFAULT_END_TIME_S,
        help="The end time in seconds for the processed videos.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dispersion_threshold_px = args.dispersion_threshold_px
    duration_threshold_ns = args.duration_threshold_ns
    min_n_fixations = args.min_n_fixations
    kde_bandwidth = args.kde_bandwidth
    target_fps = args.target_fps
    start_time_s = args.start_time_s
    end_time_s = args.end_time_s

    raw_data = get_raw_data()

    # Get processed data and write to file
    processed_data_file_path = f"{PROCESSED_VIEWOUT_PATH}/processed_data.csv"
    if not os.path.exists(processed_data_file_path):
        processed_data = process_data(raw_data)
        os.makedirs(PROCESSED_VIEWOUT_PATH, exist_ok=True)
        processed_data.to_csv(processed_data_file_path, index=False)
        print(
            f"✅ Saved processed gaze data to {Path(processed_data_file_path).resolve()} with {len(processed_data):,} entries."
        )
    else:
        processed_data = pd.read_csv(processed_data_file_path)
        print(
            f"✅ Found existing processed gaze data at {Path(processed_data_file_path).resolve()}."
        )
    del raw_data

    # Get interpolated data and write to file
    interpolated_data_file_path = f"{PROCESSED_VIEWOUT_PATH}/interpolated_data.csv"
    if not os.path.exists(interpolated_data_file_path):
        interpolated_data = get_interpolated_data(
            processed_data, resampling_rate=RESAMPLING_RATE
        )
        interpolated_data.to_csv(interpolated_data_file_path, index=False)
        print(
            f"✅ Saved interpolated gaze data to {Path(interpolated_data_file_path).resolve()} with {len(interpolated_data):,} entries."
        )
    else:
        interpolated_data = pd.read_csv(interpolated_data_file_path)
        print(
            f"✅ Found existing interpolated gaze data at {Path(interpolated_data_file_path).resolve()}."
        )
    del processed_data

    # Get fixation data and write to file
    fixation_data_file_path = f"{PROCESSED_VIEWOUT_PATH}/fixation_data.csv"
    if not os.path.exists(fixation_data_file_path):
        fixation_data = get_fixation_data(
            data=interpolated_data,
            dispersion_threshold_px=dispersion_threshold_px,
            duration_threshold_ns=duration_threshold_ns,
        )
        fixation_data.to_csv(fixation_data_file_path, index=False)
        print(
            f"✅ Saved fixation data to {Path(fixation_data_file_path).resolve()} with {len(fixation_data):,} entries."
        )
    else:
        fixation_data = pd.read_csv(fixation_data_file_path)
        print(
            f"✅ Found existing fixation data at {Path(fixation_data_file_path).resolve()}."
        )
    del interpolated_data

    fixation_data.sort_values(
        by=["ExperimentId", "SequenceId", "SetId", "TimeSinceStart_ns"], inplace=True
    )
    grouped_data = fixation_data.groupby(["ExperimentId", "SequenceId", "SetId"])
    print(f"✅ Found {len(grouped_data)} unique sequences in the fixation data.")

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args = [
            (
                group_keys,
                group_values,
                dispersion_threshold_px,
                duration_threshold_ns,
                min_n_fixations,
                kde_bandwidth,
                target_fps,
                start_time_s,
                end_time_s,
            )
            for group_keys, group_values in grouped_data
        ]
        futures = executor.map(process_sequence, *zip(*args))
        for _ in tqdm(
            futures, total=len(grouped_data), desc="⌛ Processing sequences..."
        ):
            pass


if __name__ == "__main__":
    main()
