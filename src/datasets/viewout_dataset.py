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
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from src.utils.random import set_seed
from src.utils.file import get_paths_recursive
from src.config import SEQUENCE_LENGTH, IMAGE_SIZE, PROCESSED_VIEWOUT_PATH


class ViewOutDataset(Dataset):
    def __init__(
        self,
        sample_folder_paths: List[List[str]],
        with_transforms: bool,
    ) -> None:
        super(ViewOutDataset, self).__init__()

        self.sample_folder_paths = sample_folder_paths
        self.with_transforms = with_transforms
        self.samples = self._get_samples()

    def _get_samples(self) -> List[Tuple[List[str], List[str]]]:
        samples = []
        for sample_folder_path in self.sample_folder_paths:
            # Get frames and ground_truth paths
            frames_folder_path = f"{sample_folder_path}/frames"
            ground_truths_folder_path = f"{sample_folder_path}/ground_truths"
            frames = get_paths_recursive(
                folder_path=frames_folder_path,
                match_pattern="*_00000.jpg",
                path_type="f",
                recursive=False,
            )
            ground_truths = get_paths_recursive(
                folder_path=ground_truths_folder_path,
                match_pattern="ground_truth_*.jpg",
                path_type="f",
                recursive=False,
            )

            # Sort frames and ground_truth paths
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
        self,
        frames: List[np.ndarray],
        ground_truths: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.with_transforms:
            do_hflip = random.random() > 0.5
            do_vflip = random.random() > 1.0
            do_rotate = random.random() > 1.0
            do_zoom = random.random() > 1.0
            angle = random.uniform(-15, 15) if do_rotate else 0
            zoom_factor = random.uniform(1.0, 1.2)
            brightness_factor = random.uniform(0.9, 1.1)
            contrast_factor = random.uniform(0.9, 1.1)
            saturation_factor = random.uniform(0.9, 1.1)
            hue_factor = random.uniform(-0.05, 0.05)
            do_blur = random.random() > 0.5
            sigma = random.uniform(0.1, 0.5)

            # Apply flips
            if do_hflip:
                transformed_frames = []
                for frame in frames:
                    frame = TF.hflip(frame)
                    transformed_frames.append(frame)
                frames = transformed_frames
                transformed_ground_truths = []
                for ground_truth in ground_truths:
                    ground_truth = TF.hflip(ground_truth)
                    transformed_ground_truths.append(ground_truth)
                ground_truths = transformed_ground_truths
            if do_vflip:
                transformed_frames = []
                for frame in frames:
                    frame = TF.vflip(frame)
                    transformed_frames.append(frame)
                frames = transformed_frames
                transformed_ground_truths = []
                for ground_truth in ground_truths:
                    ground_truth = TF.vflip(ground_truth)
                    transformed_ground_truths.append(ground_truth)
                ground_truths = transformed_ground_truths

            # Apply rotation
            if do_rotate:
                transformed_frames = []
                for frame in frames:
                    frame = TF.rotate(frame, angle, fill=0)
                    transformed_frames.append(frame)
                frames = transformed_frames
                transformed_ground_truths = []
                for ground_truth in ground_truths:
                    ground_truth = TF.rotate(ground_truth, angle, fill=0)
                    transformed_ground_truths.append(ground_truth)
                ground_truths = transformed_ground_truths

            # Apply zoom
            if do_zoom:
                w, h = frame.size
                crop_w = int(w / zoom_factor)
                crop_h = int(h / zoom_factor)
                left = (w - crop_w) // 2
                top = (h - crop_h) // 2
                frames = [
                    TF.resized_crop(frame, top, left, crop_h, crop_w, (h, w))
                    for frame in frames
                ]
                ground_truths = [
                    TF.resized_crop(gt, top, left, crop_h, crop_w, (h, w))
                    for gt in ground_truths
                ]

            # Apply color transforms
            transformed_frames = []
            for frame in frames:
                frame = TF.adjust_brightness(frame, brightness_factor)
                frame = TF.adjust_contrast(frame, contrast_factor)
                frame = TF.adjust_saturation(frame, saturation_factor)
                frame = TF.adjust_hue(frame, hue_factor)
                if do_blur:
                    frame = TF.gaussian_blur(frame, kernel_size=3, sigma=sigma)
                transformed_frames.append(frame)
            frames = transformed_frames

        return frames, ground_truths

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        frame_file_paths, ground_truth_file_paths = sample
        sample_id = index

        # Get frames
        frame_file_paths = natsorted(frame_file_paths)
        frames = [
            Image.open(frame_file_path).convert("RGB")
            for frame_file_path in frame_file_paths
        ]
        frames = [TF.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)) for frame in frames]

        # Get ground truths
        ground_truth_file_paths = natsorted(ground_truth_file_paths)
        ground_truths = [
            Image.open(ground_truth_file_path).convert("L")
            for ground_truth_file_path in ground_truth_file_paths
        ]
        ground_truths = [
            TF.resize(ground_truth, (IMAGE_SIZE, IMAGE_SIZE))
            for ground_truth in ground_truths
        ]

        # Apply transforms
        frames, ground_truths = self._apply_transforms(frames, ground_truths)

        # Convert to torch tensors
        frames = [TF.to_tensor(frame).float() for frame in frames]
        frames = torch.stack(frames, axis=0)
        ground_truths = [
            TF.to_tensor(ground_truth).float() for ground_truth in ground_truths
        ]
        ground_truths = [
            ground_truth / ground_truth.max() for ground_truth in ground_truths
        ]
        ground_truths = torch.stack(ground_truths, axis=0).squeeze(1)

        # Get global ground truth
        global_ground_truth = torch.mean(ground_truths, axis=0)
        global_ground_truth = global_ground_truth / global_ground_truth.max()

        return frames, ground_truths, global_ground_truth, sample_id


class ViewOutDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        with_transforms: bool,
        n_workers: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.with_transforms = with_transforms
        self.n_workers = n_workers
        self.seed = seed

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        train_sample_folder_paths, val_sample_folder_paths = self._get_splitted_sample_folder_paths()
        self.train_sample_folder_paths = train_sample_folder_paths
        self.val_sample_folder_paths = val_sample_folder_paths

    def _get_splitted_sample_folder_paths(self) -> List[str]:
        sample_folder_paths_1 = natsorted(get_paths_recursive(
            folder_path=f"{PROCESSED_VIEWOUT_PATH}/experiment1/videos",
            match_pattern="*",
            path_type="d",
            recursive=False,
        ))
        sample_folder_paths_2 = natsorted(get_paths_recursive(
            folder_path=f"{PROCESSED_VIEWOUT_PATH}/experiment2/clear",
            match_pattern="*",
            path_type="d",
            recursive=False,
        ))
        sample_folder_paths_3 = natsorted(get_paths_recursive(
            folder_path=f"{PROCESSED_VIEWOUT_PATH}/experiment2/overcast",
            match_pattern="*",
            path_type="d",
            recursive=False,
        ))

        train_indexes_1 = [2, 4, 5, 6, 7, 8, 9, 10, 12, 14]
        val_indexes_1 = [0, 1, 3, 11, 13]
        train_indexes_23 = [2, 3, 4, 5, 7, 9]
        val_indexes_23 = [0, 1, 6, 8]

        train_sample_folder_paths = [
            sample_folder_paths_1[i] for i in train_indexes_1
        ] + [
            sample_folder_paths_2[i] for i in train_indexes_23
        ] + [
            sample_folder_paths_3[i] for i in train_indexes_23
        ]
        val_sample_folder_paths = [
            sample_folder_paths_1[i] for i in val_indexes_1
        ] + [
            sample_folder_paths_2[i] for i in val_indexes_23
        ] + [
            sample_folder_paths_3[i] for i in val_indexes_23
        ]

        return train_sample_folder_paths, val_sample_folder_paths

    def setup(self, stage: Optional[str] = None):
        if self.seed is not None:
            print(f"🌱 Setting the seed to {self.seed} for generating dataloaders.")
            set_seed(self.seed)

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ViewOutDataset(
                sample_folder_paths=self.train_sample_folder_paths,
                with_transforms=self.with_transforms,
            )
            self.val_dataset = ViewOutDataset(
                sample_folder_paths=self.val_sample_folder_paths,
                with_transforms=False,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ViewOutDataset(
                sample_folder_paths=self.val_sample_folder_paths,
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
