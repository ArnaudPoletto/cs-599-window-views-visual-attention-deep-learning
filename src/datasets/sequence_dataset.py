import pickle
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Callable


class SequenceDataset(Dataset):
    def __init__(
        self,
        sample_paths_list: List[List[str]],
        sequence_length: int,
    ) -> None:
        """
        Initialize the sequence dataset.

        Args:
            sample_paths_list (List[List[str]]): List of lists of sample paths.
            sequence_length (int): The sequence length.

        Raises:
            ValueError: If the sequence length is invalid.
        """
        if sequence_length <= 0:
            raise ValueError("❌The sequence length must be greater than 0.")

        self.sample_paths_list = sample_paths_list
        self.sequence_length = sequence_length

        # Remove samples that are shorter than the sequence length
        self.sample_paths_list = [
            sample_paths
            for sample_paths in self.sample_paths_list
            if len(sample_paths) >= sequence_length
        ]

    def __len__(self) -> int:
        """
        The length of the dataset.
        It it defined as the total number of possible overlapping windows for each scene.

        Returns:
            int: The length of the dataset.
        """
        return sum(
            [
                len(sample_paths) - self.sequence_length + 1
                for sample_paths in self.sample_paths_list
            ]
        )

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item.

        Raises:
            ValueError: If the index is invalid.
        """
        selected_sample_paths = None
        for sample_paths in self.sample_paths_list:
            sequence_count = len(sample_paths) - self.sequence_length + 1
            if idx < sequence_count:
                selected_sample_paths = sample_paths
                break
            idx -= len(sample_paths) - self.sequence_length + 1

        if selected_sample_paths is None:
            raise ValueError(
                f"❌ Something went wrong while getting the item {idx} from the dataset."
            )

        sample_paths = selected_sample_paths[idx : idx + self.sequence_length]
        samples = [pickle.load(open(sample_path, "rb")) for sample_path in sample_paths]

        return samples
