import numpy as np
from typing import List, Optional


class Sample:
    """
    A sample is a series of consecutive images in a sequence, with a flow image corresponding to the next image in the sequence and an associated ground truth.
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
            ValueError: If the list of frames is empty.
            ValueError: If the frames do not have three dimensions.
            ValueError: If the frames do not have the same shape.
            ValueError: If the shape of next_frame is different from the shape of the frames.
            ValueError: If the ground truth does not have three dimensions.
            ValueError: If the spatial dimensions of the ground truth are different from the spatial dimensions of the frames.
        """
        if len(frames) == 0:
            raise ValueError("❌ The list of frames must not be empty.")
        if not all([len(frame.shape) == 3 for frame in frames]):
            raise ValueError("❌ All frames must have three dimensions.")
        if not all(frame.shape == frames[0].shape for frame in frames):
            raise ValueError("❌ All frames must have the same shape.")
        if next_frame is not None and next_frame.shape != frames[0].shape:
            raise ValueError(
                "❌ The shape of next_frame must be the same as the shape of the last image in frames."
            )
        if len(ground_truth.shape) != 3:
            raise ValueError("❌ The ground truth must have three dimensions.")
        if ground_truth.shape[1:] != frames[0].shape[1:]:
            raise ValueError(
                "❌ The spatial dimensions of the ground truth must be the same as the spatial dimensions of the frames."
            )

        self.frames = np.array(frames).astype(np.uint8)
        self.next_frame = next_frame.astype(np.uint8) if next_frame is not None else frames[-1].copy()
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
