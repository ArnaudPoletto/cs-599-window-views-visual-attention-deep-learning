import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from src.utils.frame import Frame
from src.utils.random import set_seed
from src.utils.file import get_paths_recursive


class SaliconDataset(Dataset):
    def __init__(
        self,
        sample_folder_paths: List[str],
        with_transforms: bool,
    ) -> None:
        super(SaliconDataset, self).__init__()

        self.sample_folder_paths = sample_folder_paths
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
            A.Compose(
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

    def __len__(self) -> int:
        return len(self.sample_folder_paths)

    def _apply_transforms(self, frame: np.ndarray, ground_truths: np.ndarray) -> Frame:
        # Get transforms to replay
        if self.with_transforms:
            all_transform_replay = self.all_transforms(image=frame)["replay"]

        resize = A.Resize(width=331, height=331)
        normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        transformed_frame = resize(image=frame)["image"]
        if self.with_transforms:
            transformed_frame = A.ReplayCompose.replay(
                all_transform_replay, image=transformed_frame
            )["image"]
            transformed_frame = self.input_transforms(image=transformed_frame)["image"]
        transformed_frame = normalize(image=transformed_frame)["image"]

        transformed_ground_truths = []
        for ground_truth in ground_truths:
            ground_truth = resize(image=ground_truth)["image"]
            if self.with_transforms:
                ground_truth = A.ReplayCompose.replay(
                    all_transform_replay, image=ground_truth
                )["image"]
            transformed_ground_truths.append(ground_truth)
        transformed_ground_truths = np.array(transformed_ground_truths)

        return transformed_frame, transformed_ground_truths

    def __getitem__(self, index: int) -> Frame:
        sample_folder_path = self.sample_folder_paths[index]
        frame_file_path = f"{sample_folder_path}/frame.jpg"
        output_file_paths = get_paths_recursive(
            sample_folder_path, match_pattern="ground_truth_*.jpg", file_type="f"
        )
        frame = np.array(Image.open(frame_file_path).convert("RGB"))
        ground_truths = np.array(
            [
                np.array(Image.open(output_file_path).convert("L"))
                for output_file_path in output_file_paths
            ]
        )
        frame, ground_truths = self._apply_transforms(frame, ground_truths)
        frame = frame.transpose(2, 0, 1)
        ground_truths = ground_truths.astype(np.float32) / 255.0

        # Get global ground truth
        global_ground_truth = np.mean(ground_truths, axis=0)
        min_val = global_ground_truth.min()
        max_val = global_ground_truth.max()
        if min_val == max_val:
            global_ground_truth = np.zeros_like(global_ground_truth)
        else:
            global_ground_truth = (global_ground_truth - min_val) / (max_val - min_val)

        return frame, ground_truths, global_ground_truth
    

def get_dataloaders(
    sample_folder_paths: List[str],
    with_transforms: bool,
    batch_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    train_shuffle: bool,
    n_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError(
            "❌ The sum of the train, validation, and test splits must be equal to 1."
        )
    
    if seed is not None:
        print(f"🌱 Setting the seed to {seed} for generating dataloaders.")
        set_seed(seed)

    sample_indices = np.arange(len(sample_folder_paths))
    np.random.shuffle(sample_indices)

    train_samples = int(train_split * len(sample_indices))
    val_samples = int(val_split * len(sample_indices))

    train_indices = sample_indices[:train_samples]
    val_indices = sample_indices[train_samples : train_samples + val_samples]
    test_indices = sample_indices[train_samples + val_samples :]

    train_dataset = SaliconDataset(
        sample_folder_paths=[sample_folder_paths[i] for i in train_indices],
        with_transforms=with_transforms,
    )
    val_dataset = SaliconDataset(
        sample_folder_paths=[sample_folder_paths[i] for i in val_indices],
        with_transforms=False,
    )
    test_dataset = SaliconDataset(
        sample_folder_paths=[sample_folder_paths[i] for i in test_indices],
        with_transforms=False,
    )

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

    return train_loader, val_loader, test_loader
