import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import numpy as np
from PIL import Image
from typing import List
import albumentations as A
from torch.utils.data import Dataset

from src.utils.frame import Frame
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
        ground_truths = ground_truths[:, np.newaxis, :, :]

        return frame, ground_truths
