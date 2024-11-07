import numpy as np
from typing import List, Optional


class Sample:
    """
    A sample is a series of consecutive images in a sequence, with a flow image corresponding to the next image in the sequence.
    """

    def __init__(
        self,
        frames: List[np.ndarray],
        next_frame: Optional[np.ndarray],
        ground_truth: np.ndarray,
    ) -> None:
        """
        Initialize the sample.

        Args:
            frames (List[np.ndarray]): The frames.
            next_frame (Optional[np.ndarray]): The next image, i.e. the successor of the last frame. If None, it is considered as the last frame.
            ground_truth (np.ndarray): The ground truth.

        Raises:
            ValueError: If the shapes of the images in the series are not the same.
            ValueError: If the shape of the next frame is not the same as the shape of the last image in the series.
        """
        if not all(frame.shape == frames[0].shape for frame in frames):
            raise ValueError("❌ All np.ndarrays in frames must have the same shape.")
        if next_frame is not None and next_frame.shape != frames[0].shape:
            raise ValueError(
                "❌ The shape of next_frame must be the same as the shape of the last image in frames."
            )

        self.frames = np.array(frames).astype(np.uint8)
        self.next_frame = next_frame if next_frame is not None else frames[-1].copy()
        self.ground_truth = ground_truth.astype(np.float32)

    def __len__(self) -> int:
        """
        The length of the sample.

        Returns:
            int: The length of the sample.
        """
        return self.frames.shape[0]

    def __repr__(self) -> str:
        """
        The string representation of the sample.

        Returns:
            str: The string representation of the sample.
        """
        repr = f"Sample(frames_shapes={self.frames[0].shape}"
        if self.next_frame is None:
            repr += ", next_frame=None"
        else:
            repr += f", next_frame_shape={self.next_frame.shape}"
        repr += f", ground_truth_shape={self.ground_truth.shape})"

        return repr
