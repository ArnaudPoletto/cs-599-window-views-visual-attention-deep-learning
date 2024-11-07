import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np
from typing import List

from src.utils.sample import Sample


class Sequence:
    """
    A sequence is a series of consecutive samples.
    """

    def __init__(self, sequence: List[Sample]) -> None:
        """
        Initialize the sequence.

        Args:
            sequence (List[Sample]): The sequence of samples.

        Raises:
            ValueError: If all samples in the sequence do not have the same length.
            ValueError: If all samples in the sequence do not have the same frame shape.
        """
        if not all(len(sample) == len(sequence[0]) for sample in sequence):
            raise ValueError(
                "❌ All samples in the sequence must have the same length."
            )
        if not all(
            sample.frames[0].shape == sequence[0].frames[0].shape for sample in sequence
        ):
            raise ValueError(
                "❌ All samples in the sequence must have the same frame shape."
            )

        self.sequence = sequence
        self.sample_length = len(sequence[0])

    def __len__(self) -> int:
        """
        The length of the sequence.

        Returns:
            int: The length of the sequence.
        """
        return len(self.sequence)

    def __iter__(self):
        return iter(self.sequence)

    def get_frames(self) -> np.ndarray:
        """
        Get the frames of the sequence.

        Returns:
            np.ndarray: The frames of the sequence.
        """
        frames = np.array([sample.frames for sample in self.sequence])

        return frames

    def set_frames(self, frames: np.ndarray) -> None:
        """
        Set the frames of the sequence.

        Args:
            frames (np.nparray): The frames to set.

        Raises:
            ValueError: If the number of frames does not match the number of samples in the sequence.
            ValueError: If the number of frames does not match the sample length.
        """
        if frames.shape[0] != len(self.sequence):
            raise ValueError(
                "❌ The number of frames must match the number of samples in the sequence."
            )
        if frames.shape[1] != self.sample_length:
            raise ValueError("❌ The number of frames must match the sample length.")

        for sample, frame in zip(self.sequence, frames):
            sample.frames = frame

    def get_next_frames(self) -> np.ndarray:
        """
        Get the next images of the sequence.

        Returns:
            np.ndarray: The next images of the sequence.
        """
        next_frames = np.array([sample.next_frame for sample in self.sequence])

        return next_frames

    def set_next_frames(self, next_frames: np.ndarray) -> None:
        """
        Set the next images of the sequence.

        Args:
            next_frames (np.ndarray): The next images to set.

        Raises:
            ValueError: If the number of next frames does not match the number of samples in the sequence.
        """
        if len(next_frames) != len(self.sequence):
            raise ValueError(
                "❌ The number of next frames must match the number of samples in the sequence."
            )

        for sample, next_frame in zip(self.sequence, next_frames):
            sample.next_frame = next_frame

    def get_ground_truths(self) -> np.ndarray:
        """
        Get the ground truths of the sequence.

        Returns:
            np.ndarray: The ground truths of the sequence.
        """
        ground_truths = np.array(
            [sample.ground_truth for sample in self.sequence]
        ).astype(np.float32)

        return ground_truths

    def set_ground_truths(self, ground_truths: np.ndarray) -> None:
        """
        Set the ground truths of the sequence.

        Args:
            ground_truths (np.ndarray): The ground truths to set.

        Raises:
            ValueError: If the number of ground truths does not match the number of samples in the sequence.
        """
        if len(ground_truths) != len(self.sequence):
            raise ValueError(
                "❌ The number of ground truths must match the number of samples in the sequence, got {len(ground_truths)} ground truths and {len(self.sequence)} samples."
            )

        for sample, ground_truth in zip(self.sequence, ground_truths):
            sample.ground_truth = ground_truth

    def get_global_ground_truth(self) -> np.ndarray:
        """
        Get the global ground truth of the sequence.

        Returns:
            np.ndarray: The global ground truth of the sequence.
        """
        global_ground_truth = np.mean(self.get_ground_truths(), axis=0)

        return global_ground_truth

    def __repr__(self) -> str:
        """
        The string representation of the sequence.

        Returns:
            str: The string representation of the sequence.
        """
        frames = self.get_frames()
        next_frames = self.get_next_frames()
        ground_truths = self.get_ground_truths()
        return f"Sequence(sequence_length={len(self)}, sample_length={self.sample_length}, frames_shape={frames.shape}, next_frames_shape={next_frames.shape}, ground_truths_shape={ground_truths.shape})"
