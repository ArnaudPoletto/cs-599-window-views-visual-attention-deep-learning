import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from src.utils.random import set_seed
from src.utils.file import get_paths_recursive


class DHF1KDataset(Dataset):
    def __init__(
        self,
        sample_folder_paths: List[str],
        sequence_length: int,
        with_transforms: bool,
    ) -> None:
        super(DHF1KDataset, self).__init__()

        self.sample_folder_paths = sample_folder_paths
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

        self.samples = self._get_samples()

    def _get_samples(self) -> List[Tuple[List[str], List[str]]]:
        samples = []
        for sample_folder_path in self.sample_folder_paths:
            frames_folder_path = f"{sample_folder_path}/frames"
            ground_truths_folder_path = f"{sample_folder_path}/ground_truths"
            frames = get_paths_recursive(
                folder_path=frames_folder_path, match_pattern="*_1.jpg", file_type="f"
            )  # TODO: Only first frame for now
            ground_truths = get_paths_recursive(
                folder_path=ground_truths_folder_path,
                match_pattern="ground_truth_*.jpg",
                file_type="f",
            )

            # Remove any frames that do not have corresponding ground truth files. This mismatch 
            # typically occurs at the end of videos that have fractional-second durations (e.g., 
            # a 5.5 second video will have frames for the full duration but ground truth only up 
            # to 5.0 seconds)
            if len(frames) < len(ground_truths):
                raise ValueError(
                    f"❌ The number of frames ({len(frames)}) is less than the number of ground truths ({len(ground_truths)}) for {sample_folder_path}."
                )
            frames = frames[:len(ground_truths)]
            

            n_samples = len(frames) - self.sequence_length + 1
            if n_samples <= 0:
                continue

            for start_idx in range(n_samples):
                end_idx = start_idx + self.sequence_length
                sample = (frames[start_idx:end_idx], ground_truths[start_idx:end_idx])
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_transforms(
        self, frames: np.ndarray, ground_truths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Get transforms to replay
        if self.with_transforms:
            all_transform_replay = self.all_transforms(image=frames[0])["replay"]
            input_transform_replay = self.input_transforms(image=frames[0])["replay"]

        resize = A.Resize(width=331, height=331)  # TODO: remove hardcoded

        transformed_frames = []
        for frame in frames:
            frame = resize(image=frame)["image"]
            if self.with_transforms:
                frame = A.ReplayCompose.replay(all_transform_replay, image=frame)[
                    "image"
                ]
                frame = A.ReplayCompose.replay(input_transform_replay, image=frame)[
                    "image"
                ]
            transformed_frames.append(frame)
        transformed_frames = np.array(transformed_frames)

        transformed_ground_truths = []
        for ground_truth in ground_truths:
            ground_truth = resize(image=ground_truth)["image"]
            if self.with_transforms:
                ground_truth = A.ReplayCompose.replay(
                    all_transform_replay, image=ground_truth
                )["image"]
            transformed_ground_truths.append(ground_truth)
        transformed_ground_truths = np.array(transformed_ground_truths)

        return transformed_frames, transformed_ground_truths

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample = self.samples[index]
        frames, ground_truths = sample

        frames = np.array(
            [np.array(Image.open(frame).convert("RGB")) for frame in frames]
        )
        ground_truths = np.array(
            [
                np.array(Image.open(ground_truth).convert("L"))
                for ground_truth in ground_truths
            ]
        )
        frames, ground_truths = self._apply_transforms(frames, ground_truths)
        frames = frames.transpose(0, 3, 1, 2)
        ground_truths = ground_truths.astype(np.float32) / 255.0

        # Get global ground truth
        global_ground_truth = np.mean(ground_truths, axis=0)
        min_val = global_ground_truth.min()
        max_val = global_ground_truth.max()
        if min_val == max_val:
            global_ground_truth = np.zeros_like(global_ground_truth)
        else:
            global_ground_truth = (global_ground_truth - min_val) / (max_val - min_val)

        return frames, ground_truths, global_ground_truth
    

def get_dataloaders(
    sample_folder_paths: List[str],
    sequence_length: int,
    with_transforms: bool,
    batch_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    train_shuffle: bool,
    n_workers: int,
    seed: Optional[int],
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

    train_dataset = DHF1KDataset(
        sample_folder_paths=[sample_folder_paths[i] for i in train_indices],
        sequence_length=sequence_length,
        with_transforms=with_transforms,
    )
    val_dataset = DHF1KDataset(
        sample_folder_paths=[sample_folder_paths[i] for i in val_indices],
        sequence_length=sequence_length,
        with_transforms=with_transforms,
    )
    test_dataset = DHF1KDataset(
        sample_folder_paths=[sample_folder_paths[i] for i in test_indices],
        sequence_length=sequence_length,
        with_transforms=with_transforms,
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