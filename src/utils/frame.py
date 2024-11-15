from src.utils.sample import Sample
from src.utils.sequence import Sequence
import numpy as np


class Frame:
    """
    A frame is a single image, with associated time-continuous ground truths.
    """

    def __init__(self, frame: np.ndarray, ground_truths: np.ndarray):
        """
        Initialize the frame.

        Args:
            frame (np.ndarray): The frame.
            ground_truths (np.ndarray): The ground truths.

        Raises:
            ValueError: If the frame does not have three dimensions.
            ValueError: If the ground truths do not have three dimensions.
            ValueError: If the spatial dimensions of the ground truths are different from the spatial dimensions of the frame.
        """
        if len(frame.shape) != 3:
            raise ValueError("❌ The frame must have three dimensions: (n_channels, height, width).")
        if len(ground_truths.shape) != 4:
            raise ValueError("❌ The ground truths must have four dimensions: (n_ground_truth, n_channels, height, width).")
        if ground_truths.shape[2:] != frame.shape[1:]:
            raise ValueError(
                "❌ The spatial dimensions of the ground truths must be the same as the spatial dimensions of the frame."
            )

        self.frame = frame.astype(np.uint8)
        self.ground_truths = ground_truths.astype(np.float32)

    def get_frame(self) -> np.ndarray:
        """
        Get the frame.

        Returns:
            np.ndarray: The frame.
        """
        return self.frame
    
    def set_frame(self, frame: np.ndarray) -> None:
        """
        Set the frame.

        Args:
            frame (np.ndarray): The frame.
        """
        self.frame = frame
    
    def get_ground_truths(self) -> np.ndarray:
        """
        Get the ground truths.

        Returns:
            np.ndarray: The ground truths.
        """
        return self.ground_truths
    
    def set_ground_truths(self, ground_truths: np.ndarray) -> None:
        """
        Set the ground truths.

        Args:
            ground_truths (np.ndarray): The ground truths.
        """
        self.ground_truths = ground_truths

    def to_sequence(self, sample_length: int) -> Sequence:
        """
        Convert the frame to a sequence.

        Args:
            sample_length (int): The length of each sample.

        Returns:
            Sequence: The sequence.
        """
        sequence = [
            Sample(
                frames=[self.frame] * sample_length,
                next_frame=None,
                ground_truth=ground_truth,
            )
            for ground_truth in self.ground_truths
        ]
        sequence = Sequence(sequence=sequence)

        return sequence
    
    def __repr__(self) -> str:
        return f"Frame(frame={self.frame}, ground_truths={self.ground_truths})"
