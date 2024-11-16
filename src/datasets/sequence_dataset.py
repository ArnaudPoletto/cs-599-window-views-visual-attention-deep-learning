import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import pickle
import random
import numpy as np
from typing import List, Tuple
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from src.utils.random import set_seed
from src.utils.sequence import Sequence
from src.config import SEED


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
        super(SequenceDataset, self).__init__()
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

    def _apply_transforms(self, sequence: Sequence) -> Sequence:
        """
        Apply transforms to the samples.

        Args:
            sequence (Sequence): The sequence of samples.

        Returns:
            Sequence: The transformed sequence.
        """
        # Get transforms to replay
        if self.with_transforms:
            frame = sequence.get_frames()[0][0]
            input_transform_replay = self.input_transforms(image=frame)["replay"]
            all_transform_replay = self.all_transforms(image=frame)["replay"]

        resize = A.Resize(width=331, height=331)
        normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # Apply the stored replay to all images
        transformed_frames = []
        frames = sequence.get_frames()
        frames_shape = frames.shape
        frames = frames.reshape(-1, *frames_shape[2:])
        for frame in frames:
            frame = resize(image=frame)["image"]
            if self.with_transforms:
                frame = A.ReplayCompose.replay(all_transform_replay, image=frame)[
                    "image"
                ]
                frame = A.ReplayCompose.replay(input_transform_replay, image=frame)[
                    "image"
                ]
            frame = normalize(image=frame)["image"]
            transformed_frames.append(frame)
        transformed_frames = np.array(transformed_frames).reshape(
            frames_shape[:2] + transformed_frames[0].shape
        )
        sequence.set_frames(transformed_frames)

        transformed_next_frames = []
        next_frames = sequence.get_next_frames()
        for next_frame in next_frames:
            next_frame = resize(image=next_frame)["image"]
            if self.with_transforms:
                next_frame = A.ReplayCompose.replay(
                    all_transform_replay, image=next_frame
                )["image"]
                next_frame = A.ReplayCompose.replay(
                    input_transform_replay, image=next_frame
                )["image"]
            next_frame = normalize(image=next_frame)["image"]
            transformed_next_frames.append(next_frame)
        transformed_next_frames = np.array(transformed_next_frames)
        sequence.set_next_frames(transformed_next_frames)

        transformed_ground_truths = []
        ground_truths = sequence.get_ground_truths()
        for ground_truth in ground_truths:
            ground_truth = resize(image=ground_truth)["image"]
            if self.with_transforms:
                ground_truth = A.ReplayCompose.replay(
                    all_transform_replay, image=ground_truth
                )["image"]
            transformed_ground_truths.append(ground_truth)
        transformed_ground_truths = np.array(transformed_ground_truths)
        sequence.set_ground_truths(transformed_ground_truths)

        return sequence

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
        sequence = Sequence(samples)

        sequence = self._apply_transforms(sequence)

        frames = sequence.get_frames()
        frames = np.moveaxis(frames, -1, 2)  # Place channel axis at the beginning
        ground_truths = sequence.get_ground_truths()
        global_ground_truth = sequence.get_global_ground_truth()

        return frames, ground_truths, global_ground_truth


def get_dataloaders(
    sample_paths_list: List[List[str]],
    sequence_length: int,
    with_transforms: bool,
    batch_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    train_shuffle: bool,
    n_workers: int,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError(
            "❌ The sum of the train, validation, and test splits must be equal to 1."
        )

    # Check if we have enough scenes for the requested split
    num_scenes = len(sample_paths_list)
    if num_scenes < 3:
        raise ValueError(
            f"❌ Not enough scenes ({num_scenes}) to split into train, validation, and test sets. "
            "Need at least 3 scenes."
        )

    if seed is not None:
        print(f"🌱 Setting the seed to {seed} for generating dataloaders")
        set_seed(seed)

    scene_indices = list(range(num_scenes))
    random.shuffle(scene_indices)

    # Calculate initial split sizes
    val_scenes = max(1, int(val_split * num_scenes))
    test_scenes = max(1, int(test_split * num_scenes))
    train_scenes = num_scenes - val_scenes - test_scenes

    # Check if we have enough scenes for the requested split
    if train_scenes < 1:
        raise ValueError(
            "❌ Not enough scenes to maintain at least one scene for training"
        )

    train_scene_indices = scene_indices[:train_scenes]
    val_scene_indices = scene_indices[train_scenes : train_scenes + val_scenes]
    test_scene_indices = scene_indices[train_scenes + val_scenes :]

    # Create flattened lists of samples for each split while tracking scene boundaries
    train_samples = []
    val_samples = []
    test_samples = []
    for idx, scene_samples in enumerate(sample_paths_list):
        if idx in train_scene_indices:
            train_samples.append(scene_samples)
        elif idx in val_scene_indices:
            val_samples.append(scene_samples)
        elif idx in test_scene_indices:
            test_samples.append(scene_samples)

    # Create separate datasets
    train_dataset = SequenceDataset(
        sample_paths_list=train_samples,
        sequence_length=sequence_length,
        with_transforms=with_transforms,
    )
    val_dataset = SequenceDataset(
        sample_paths_list=val_samples,
        sequence_length=sequence_length,
        with_transforms=False,
    )
    test_dataset = SequenceDataset(
        sample_paths_list=test_samples,
        sequence_length=sequence_length,
        with_transforms=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=n_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    print(
        f"✅ Created dataloaders with {len(train_loader)} training, {len(val_loader)} validation, and {len(test_loader)} test batches."
    )

    return train_loader, val_loader, test_loader
