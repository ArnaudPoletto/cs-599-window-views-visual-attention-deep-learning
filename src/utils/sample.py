import numpy as np
from typing import List, Optional


class Sample:
    """
    A sample is a series of consecutive images in a sequence, with a flow image corresponding to the next image in the sequence.
    """

    def __init__(
        self,
        image_series: List[np.ndarray],
        next_image: Optional[np.ndarray],
        ground_truth: np.ndarray,
    ) -> None:
        """
        Initialize the sample.

        Args:
            image_series (List[np.ndarray]): The series of images.
            next_image (Optional[np.ndarray]): The next image, i.e. the successor of the last image in the series. If None, the last image in the series is considered the last image in the sequence.
            ground_truth (np.ndarray): The ground truth.
        """
        self.image_series = image_series
        self.next_image = (
            next_image if next_image is not None else image_series[-1].copy()
        )
        self.ground_truth = ground_truth

    def __len__(self) -> int:
        """
        The length of the sample.

        Returns:
            int: The length of the sample.
        """
        return len(self.image_series)

    def __repr__(self) -> str:
        """
        The string representation of the sample.

        Returns:
            str: The string representation of the sample.
        """
        repr = f"Sample(image_series_shapes={[image_element.shape for image_element in self.image_series]}"
        if self.next_image is None:
            repr += ", next_image=None"
        else:
            repr += f", next_image={self.next_image.shape}"
        repr += f", ground_truth_shape={self.ground_truth.shape})"

        return repr
