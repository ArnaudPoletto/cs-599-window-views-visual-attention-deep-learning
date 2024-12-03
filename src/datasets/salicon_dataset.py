import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
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
from src.config import IMAGE_SIZE, RAW_SALICON_IMAGES_PATH, PROCESSED_SALICON_PATH


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
        self, 
        frame: np.ndarray, 
        ground_truths: List[np.ndarray],
        global_ground_truth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                global_ground_truth = TF.hflip(global_ground_truth)

            # Apply rotation
            if do_rotate:
                frame = TF.rotate(frame, angle, fill=0)
                transformed_ground_truths = []
                for ground_truth in ground_truths:
                    ground_truth = TF.rotate(ground_truth, angle, fill=0)
                    transformed_ground_truths.append(ground_truth)
                ground_truths = transformed_ground_truths
                global_ground_truth = TF.rotate(global_ground_truth, angle, fill=0)

            # Apply color transforms
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            frame = TF.adjust_saturation(frame, saturation_factor)
            frame = TF.adjust_hue(frame, hue_factor)

            # Apply Gaussian blur
            frame = TF.gaussian_blur(frame, kernel_size=3, sigma=sigma)

        return frame, ground_truths, global_ground_truth

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        sample_folder_path = self.sample_folder_paths[index]
        sample_id = int(Path(sample_folder_path).name)

        # Get frame
        frame_file_path = f"{sample_folder_path}/frame.jpg"
        frame = Image.open(frame_file_path).convert("RGB")
        frame = TF.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = TF.to_tensor(frame).float()

        # Get ground truths if available...
        ground_truth_file_paths = get_paths_recursive(
            sample_folder_path, match_pattern="ground_truth_*.jpg", path_type="f"
        )
        ground_truth_file_paths = natsorted(ground_truth_file_paths)
        global_ground_truth_file_path = f"{sample_folder_path}/global_ground_truth.png"

        # ...otherwise return the frame only, typically for the challenge test set
        if len(ground_truth_file_paths) == 0 or not os.path.exists(global_ground_truth_file_path):
            return frame, torch.zeros(1), torch.zeros(1), sample_id

        ground_truths = [
                Image.open(output_file_path).convert("L")
                for output_file_path in ground_truth_file_paths
            ]
        ground_truths = [
            TF.resize(gt, (IMAGE_SIZE, IMAGE_SIZE)) for gt in ground_truths
        ]
        global_ground_truth = Image.open(global_ground_truth_file_path).convert("L")
        global_ground_truth = TF.resize(global_ground_truth, (IMAGE_SIZE, IMAGE_SIZE))
        frame, ground_truths, global_ground_truth = self._apply_transforms(frame, ground_truths, global_ground_truth)

        # Convert to torch tensors and normalize ground truths
        ground_truths = [TF.to_tensor(ground_truth).float() for ground_truth in ground_truths]
        ground_truths = [ground_truth / ground_truth.max() for ground_truth in ground_truths]
        ground_truths = torch.stack(ground_truths, axis=0).squeeze(1)
        global_ground_truth = TF.to_tensor(global_ground_truth).float().squeeze(0)
        global_ground_truth = global_ground_truth / global_ground_truth.max()

        return frame, ground_truths, global_ground_truth, sample_id

class SaliconDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_folder_paths: List[str],
        batch_size: int,
        train_split: float,
        val_split: float,
        test_split: float,
        use_challenge_split: bool,
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
        self.use_challenge_split = use_challenge_split
        self.with_transforms = with_transforms
        self.n_workers = n_workers
        self.seed = seed
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _get_challenge_split_dict(self):
        image_file_paths = get_paths_recursive(RAW_SALICON_IMAGES_PATH, match_pattern="*.jpg", path_type="f")
        image_file_paths = natsorted(image_file_paths)
        challenge_split_dict = {"train": [], "val": [], "test": []}
        for image_file_path in image_file_paths:
            image_file_name = os.path.basename(image_file_path)
            category = "test"
            if "train" in image_file_name:
                category = "train"
            elif "val" in image_file_name:
                category = "val"
            else:
                category = "test"

            sample_id = int(image_file_name.split(".")[0].split("_")[-1])
            sample_folder_path = f"{PROCESSED_SALICON_PATH}/{sample_id}"
            challenge_split_dict[category].append(sample_folder_path)

        return challenge_split_dict
    
    def setup(self, stage: Optional[str] = None):
        if not np.isclose(self.train_split + self.val_split + self.test_split, 1.0):
            raise ValueError(
                "❌ The sum of the train, validation, and test splits must be equal to 1."
            )
        
        if self.seed is not None:
            print(f"🌱 Setting the seed to {self.seed} for generating dataloaders.")
            set_seed(self.seed)

        if self.use_challenge_split:
            print("📚 Using the SALICON challenge split.")
            challenge_split_dict = self._get_challenge_split_dict()
            train_sample_folder_paths = challenge_split_dict["train"]
            val_sample_folder_paths = challenge_split_dict["val"]
            test_sample_folder_paths = challenge_split_dict["test"]
        else:
            print("📚 Using the custom split.")
            # Split indices
            sample_indices = np.arange(len(self.sample_folder_paths))
            np.random.shuffle(sample_indices)

            train_samples = int(self.train_split * len(sample_indices))
            val_samples = int(self.val_split * len(sample_indices))

            train_indices = sample_indices[:train_samples]
            val_indices = sample_indices[train_samples : train_samples + val_samples]
            test_indices = sample_indices[train_samples + val_samples :]

            train_sample_folder_paths = [self.sample_folder_paths[i] for i in train_indices]
            val_sample_folder_paths = [self.sample_folder_paths[i] for i in val_indices]
            test_sample_folder_paths = [self.sample_folder_paths[i] for i in test_indices]

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = SaliconDataset(
                sample_folder_paths=train_sample_folder_paths,
                with_transforms=self.with_transforms,
            )
            self.val_dataset = SaliconDataset(
                sample_folder_paths=val_sample_folder_paths,
                with_transforms=False,
            )
            
        if stage in ["test", "predict"] or stage is None:
            self.test_dataset = SaliconDataset(
                sample_folder_paths=test_sample_folder_paths,
                with_transforms=False,
            )

        print(f"📊 Setup datasets with sizes:")
        if self.train_dataset is not None:
            print(f"  - Train: {len(self.train_dataset)} samples")
        if self.val_dataset is not None:
            print(f"  - Validation: {len(self.val_dataset)} samples")
        if self.test_dataset is not None:
            print(f"  - Test: {len(self.test_dataset)} samples")

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
    
    def predict_dataloader(self):
        return self.test_dataset