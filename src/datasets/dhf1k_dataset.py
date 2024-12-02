import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
import random
import numpy as np
from PIL import Image
from natsort import natsorted
import lightning.pytorch as pl
from torchvision import transforms
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from src.utils.random import set_seed
from src.utils.file import get_paths_recursive
from src.config import SEQUENCE_LENGTH, IMAGE_SIZE


class DHF1KDataset(Dataset):
    def __init__(
        self,
        sample_folder_paths: List[str],
        with_transforms: bool,
    ) -> None:
        super(DHF1KDataset, self).__init__()

        self.sample_folder_paths = sample_folder_paths
        self.with_transforms = with_transforms
        self.samples = self._get_samples()
        self.input_transforms = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ]
        )

    def _get_samples(self) -> List[Tuple[List[str], List[str]]]:
        samples = []
        for sample_folder_path in self.sample_folder_paths:
            # Get frames and ground truth paths
            frames_folder_path = f"{sample_folder_path}/frames"
            ground_truths_folder_path = f"{sample_folder_path}/ground_truths"
            frames = get_paths_recursive(
                folder_path=frames_folder_path, match_pattern="*_1.jpg", path_type="f"
            )  # TODO: Only first frame for now
            ground_truths = get_paths_recursive(
                folder_path=ground_truths_folder_path,
                match_pattern="ground_truth_*.jpg",
                path_type="f",
            )

            # Sort frames and ground truths paths
            frames = natsorted(frames)
            ground_truths = natsorted(ground_truths)

            # Remove any frames that do not have corresponding ground truth files. This mismatch
            # typically occurs at the end of videos that have fractional-second durations (e.g.,
            # a 5.5 second video will have frames for the full duration but ground truth only up
            # to 5.0 seconds)
            if len(frames) < len(ground_truths):
                raise ValueError(
                    f"❌ The number of frames ({len(frames)}) is less than the number of ground truths ({len(ground_truths)}) for {sample_folder_path}."
                )
            frames = frames[: len(ground_truths)]

            n_samples = len(frames) - SEQUENCE_LENGTH + 1
            if n_samples <= 0:
                continue

            for start_idx in range(n_samples):
                end_idx = start_idx + SEQUENCE_LENGTH
                sample = (frames[start_idx:end_idx], ground_truths[start_idx:end_idx])
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_transforms(
        self, frames: List[np.ndarray], ground_truths: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Resize frames and ground truths
        frames = [TF.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)) for frame in frames]
        ground_truths = [
            TF.resize(gt, (IMAGE_SIZE, IMAGE_SIZE)) for gt in ground_truths
        ]

        if self.with_transforms:
            do_flip = random.random() > 0.5
            do_rotate = random.random() > 0.5
            angle = random.uniform(-15, 15) if do_rotate else 0
            brightness_factor = random.uniform(0.9, 1.1)
            contrast_factor = random.uniform(0.9, 1.1)
            saturation_factor = random.uniform(0.9, 1.1)
            hue_factor = random.uniform(-0.05, 0.05)
            sigma = random.uniform(0.1, 0.5)

            transformed_frames = []
            transformed_ground_truths = []
            for frame, ground_truth in zip(frames, ground_truths):
                # Apply flip
                if do_flip:
                    frame = TF.hflip(frame)
                    ground_truth = TF.hflip(ground_truth)

                # Apply rotation
                if do_rotate:
                    frame = TF.rotate(frame, angle, fill=0)
                    ground_truth = TF.rotate(ground_truth, angle, fill=0)

                # Apply color transforms
                frame = TF.adjust_brightness(frame, brightness_factor)
                frame = TF.adjust_contrast(frame, contrast_factor)
                frame = TF.adjust_saturation(frame, saturation_factor)
                frame = TF.adjust_hue(frame, hue_factor)

                # Apply Gaussian blur
                frame = TF.gaussian_blur(frame, kernel_size=3, sigma=sigma)

                transformed_frames.append(frame)
                transformed_ground_truths.append(ground_truth)

            frames = transformed_frames
            ground_truths = transformed_ground_truths

        return frames, ground_truths

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        frames, ground_truths = sample

        # Get frames and ground truths and apply transforms
        frames = [Image.open(frame).convert("RGB") for frame in frames]
        ground_truths = [
            Image.open(ground_truth).convert("L") for ground_truth in ground_truths
        ]
        frames, ground_truths = self._apply_transforms(frames, ground_truths)

        # Convert to torch tensors
        frames = [TF.to_tensor(frame).float() for frame in frames]
        frames = torch.stack(frames, axis=0)
        ground_truths = [TF.to_tensor(ground_truth).float() for ground_truth in ground_truths]
        ground_truths = torch.stack(ground_truths, axis=0).squeeze(1)

        # Get global ground truth
        global_ground_truth = torch.mean(ground_truths, axis=0)
        min_val = global_ground_truth.min()
        max_val = global_ground_truth.max()
        if min_val == max_val:
            global_ground_truth = torch.zeros_like(global_ground_truth)
        else:
            global_ground_truth = (global_ground_truth - min_val) / (max_val - min_val)

        return frames, ground_truths, global_ground_truth


class DHF1KDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_folder_paths: List[str],
        batch_size: int,
        train_split: float,
        val_split: float,
        test_split: float,
        with_transforms: bool,
        n_workers: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.sample_folder_paths = sample_folder_paths
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.with_transforms = with_transforms
        self.n_workers = n_workers
        self.seed = seed

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not np.isclose(self.train_split + self.val_split + self.test_split, 1.0):
            raise ValueError(
                "❌ The sum of the train, validation, and test splits must be equal to 1."
            )

        if self.seed is not None:
            print(f"🌱 Setting the seed to {self.seed} for generating dataloaders.")
            set_seed(self.seed)

        # Split indices
        sample_indices = np.arange(len(self.sample_folder_paths))
        np.random.shuffle(sample_indices)

        train_samples = int(self.train_split * len(sample_indices))
        val_samples = int(self.val_split * len(sample_indices))

        train_indices = sample_indices[:train_samples]
        val_indices = sample_indices[train_samples : train_samples + val_samples]
        test_indices = sample_indices[train_samples + val_samples :]

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = DHF1KDataset(
                sample_folder_paths=[
                    self.sample_folder_paths[i] for i in train_indices
                ],
                with_transforms=self.with_transforms,
            )
            self.val_dataset = DHF1KDataset(
                sample_folder_paths=[self.sample_folder_paths[i] for i in val_indices],
                with_transforms=False,
            )

        if stage == "test" or stage is None:
            self.test_dataset = DHF1KDataset(
                sample_folder_paths=[self.sample_folder_paths[i] for i in test_indices],
                with_transforms=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True,
        )
