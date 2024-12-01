import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import random
import torch
import numpy as np
from PIL import Image
import lightning.pytorch as pl
from torchvision import transforms
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from src.utils.random import set_seed
from src.utils.file import get_paths_recursive
from src.config import IMAGE_SIZE


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
        ])

    def __len__(self) -> int:
        return len(self.sample_folder_paths)

    def _apply_transforms(
        self, frame: np.ndarray, ground_truths: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Resize frames and ground truths
        frame = TF.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
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

            # Apply flip
            if do_flip:
                frame = TF.hflip(frame)
                transformed_ground_truths = []
                for ground_truth in ground_truths:
                    ground_truth = TF.hflip(ground_truth)
                    transformed_ground_truths.append(ground_truth)
                ground_truths = transformed_ground_truths

            # Apply rotation
            if do_rotate:
                frame = TF.rotate(frame, angle, fill=0)
                transformed_ground_truths = []
                for ground_truth in ground_truths:
                    ground_truth = TF.rotate(ground_truth, angle, fill=0)
                    transformed_ground_truths.append(ground_truth)
                ground_truths = transformed_ground_truths

            # Apply color transforms
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            frame = TF.adjust_saturation(frame, saturation_factor)
            frame = TF.adjust_hue(frame, hue_factor)

            # Apply Gaussian blur
            frame = TF.gaussian_blur(frame, kernel_size=3, sigma=sigma)

        return frame, ground_truths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        sample_folder_path = self.sample_folder_paths[index]

        # Get frames and ground truths and apply transforms
        frame_file_path = f"{sample_folder_path}/frame.jpg"
        output_file_paths = get_paths_recursive(
            sample_folder_path, match_pattern="ground_truth_*.jpg", path_type="f"
        )
        print("unsorted output_file_paths", output_file_paths)
        output_file_paths = sorted(output_file_paths)
        print(">>sorted output_file_paths", output_file_paths)
        frame = Image.open(frame_file_path).convert("RGB")
        ground_truths = [
                Image.open(output_file_path).convert("L")
                for output_file_path in output_file_paths
            ]
        frame, ground_truths = self._apply_transforms(frame, ground_truths)

        # Convert to torch tensors
        frame = TF.to_tensor(frame).float()
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