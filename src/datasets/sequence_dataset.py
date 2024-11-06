import cv2
import pickle
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from typing import List

from src.utils.sample import Sample
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT


class SequenceDataset(Dataset):
    def __init__(
        self,
        sample_paths_list: List[List[str]],
        sequence_length: int,
        with_transforms: bool,
    ) -> None:
        """
        Initialize the sequence dataset.

        Args:
            sample_paths_list (List[List[str]]): List of lists of sample paths.
            sequence_length (int): The sequence length.
            with_transforms (bool): Whether to apply transforms to the samples.

        Raises:
            ValueError: If the sequence length is invalid.
        """
        if sequence_length <= 0:
            raise ValueError("❌The sequence length must be greater than 0.")

        self.sample_paths_list = sample_paths_list
        self.sequence_length = sequence_length
        self.with_transforms = with_transforms
        self.all_transforms = (
            A.ReplayCompose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                ]
            )
            if with_transforms
            else None
        )
        self.input_transforms = (
            A.ReplayCompose(
                [
                    A.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8
                    ),
                    A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.5), p=0.5),
                    A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
                    # TODO: normalize
                ]
            )
            if with_transforms
            else None
        )

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

    def apply_transforms(self, samples: List[Sample]) -> List[np.ndarray]:
        """
        Apply transforms to the samples.

        Args:
            samples (List[Sample]): The samples.

        Returns:
            List[Sample]: The transformed samples.
        """
        # Get transforms to replay
        input_transform_replay = self.input_transforms(image=samples[0].image_series[0])
        input_transform_replay = input_transform_replay["replay"]
        all_replayed_transforms = self.all_transforms(image=samples[0].image_series[0])
        all_transform_replay = all_replayed_transforms["replay"]

        # Apply the stored replay to all images
        for sample in samples:
            transformed_image_series = []
            for frame in sample.image_series:
                frame = A.ReplayCompose.replay(input_transform_replay, image=frame)["image"]
                frame = A.ReplayCompose.replay(all_transform_replay, image=frame)["image"]
                transformed_image_series.append(frame)
            sample.image_series = transformed_image_series
            sample.next_image = A.ReplayCompose.replay(input_transform_replay, image=sample.next_image)["image"]
            sample.next_image = A.ReplayCompose.replay(all_transform_replay, image=sample.next_image)["image"]
            sample.ground_truth = A.ReplayCompose.replay(all_transform_replay, image=sample.ground_truth)["image"]

        return samples

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

        if self.with_transforms:
            samples = self.apply_transforms(samples)

        return samples
