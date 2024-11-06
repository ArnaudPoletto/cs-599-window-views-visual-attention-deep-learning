import numpy as np


class SampleGroundTruth:
    """
    A sample ground truth is a single heatmap representing the ground truth distribution of fixations for a given sample.
    """

    def __init__(
        self, heatmap: np.ndarray
    ) -> None:
        """
        Initialize the sample.

        Args:
            heatmap (np.ndarray): The heatmap representing the ground truth distribution of fixations.
        """
        # Normalize the heatmap
        heatmap_range = heatmap.max() - heatmap.min()
        if heatmap_range == 0:
            heatmap = np.zeros_like(heatmap)
        else:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        self.heatmap = heatmap

    def __repr__(self) -> str:
        """
        The string representation of the sample.
        
        Returns:
            str: The string representation of the sample.
        """
        repr = f"SampleGroundTruth(heatmap_shape={self.heatmap.shape})"

        return repr
