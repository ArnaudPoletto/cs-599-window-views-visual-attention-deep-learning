import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import torch
import random
import numpy as np
from PIL import Image
import albumentations as A
import lightning.pytorch as pl
from torchvision import transforms
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

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
        self.input_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ]) if with_transforms else None

    def __len__(self) -> int:
        return len(self.sample_folder_paths)

    def _apply_transforms(self, frame: Image.Image, ground_truths: List[Image.Image]) -> Tuple[Image.Image, List[Image.Image]]:
        # TODO: remove the hard-coded values
        frame = TF.resize(frame, (331, 331))
        ground_truths = [TF.resize(gt, (331, 331)) for gt in ground_truths]

        if self.with_transforms:
            # Apply random horizontal flip
            if random.random() > 0.5:
                frame = TF.hflip(frame)
                ground_truths = [TF.hflip(gt) for gt in ground_truths]
            
            # Apply random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                frame = TF.rotate(frame, angle, fill=0)
                ground_truths = [TF.rotate(gt, angle, fill=0) for gt in ground_truths]
            
            # Apply color transforms only to input frame
            if self.input_transforms:
                frame = self.input_transforms(frame)

        return frame, ground_truths

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_folder_path = self.sample_folder_paths[index]
        frame_file_path = f"{sample_folder_path}/frame.jpg"
        output_file_paths = get_paths_recursive(
            sample_folder_path, match_pattern="ground_truth_*.jpg", path_type="f"
        )
        frame = Image.open(frame_file_path).convert("RGB")
        ground_truths = [
                Image.open(output_file_path).convert("L")
                for output_file_path in output_file_paths
            ]
        frame, ground_truths = self._apply_transforms(frame, ground_truths)

        # Convert to numpy arrays
        frame = np.array(frame).transpose(2, 0, 1).astype(np.float32)
        ground_truths = np.array(ground_truths).astype(np.float32) / 255.0

        # Get global ground truth
        global_ground_truth = np.mean(ground_truths, axis=0)
        min_val = global_ground_truth.min()
        max_val = global_ground_truth.max()
        if min_val == max_val:
            global_ground_truth = np.zeros_like(global_ground_truth)
        else:
            global_ground_truth = (global_ground_truth - min_val) / (max_val - min_val)

        return frame, ground_truths, global_ground_truth

class SaliconDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_folder_paths: List[str],
        batch_size: int,
        train_split: float,
        val_split: float,
        test_split: float,
        with_transforms: bool,
        n_workers: int,
        seed: Optional[int] = None
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
            self.train_dataset = SaliconDataset(
                sample_folder_paths=[self.sample_folder_paths[i] for i in train_indices],
                with_transforms=self.with_transforms,
            )
            self.val_dataset = SaliconDataset(
                sample_folder_paths=[self.sample_folder_paths[i] for i in val_indices],
                with_transforms=False,
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = SaliconDataset(
                sample_folder_paths=[self.sample_folder_paths[i] for i in test_indices],
                with_transforms=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
        )