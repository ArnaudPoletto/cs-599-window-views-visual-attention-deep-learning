import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import timm
import torch
from torch import nn
from typing import List

from src.config import IMAGE_SIZE, IMAGE_ENCODER_MODEL_NAME, IMAGE_ENCODER_PRETRAINED


class ImageEncoder(nn.Module):
    """
    An image encoder that optionally uses a pretrained model to extract features from an image.
    """

    def __init__(
        self,
        freeze: bool,
        model_name: str = IMAGE_ENCODER_MODEL_NAME,
        pretrained: bool = IMAGE_ENCODER_PRETRAINED,
    ) -> None:
        """
        Initialize the image encoder.

        Args:
            model_name (str): The name of the model to use.
            pretrained (bool): Whether to use a pretrained model to extract features from the image. Defaults to True.
            freeze (bool): Whether to freeze the model's parameters. Defaults to True.
        """
        super(ImageEncoder, self).__init__()

        self.pretrained = pretrained
        self.freeze = freeze
        self.model_name = model_name

        self.pnas = timm.create_model(
            model_name=model_name, pretrained=pretrained, features_only=True
        )
        self.feature_channels_list = self._get_feature_channels_list()
        self.feature_sizes = self._get_feature_sizes()

        if freeze:
            for param in self.pnas.parameters():
                param.requires_grad = False

    def _get_feature_channels_list(self) -> List[int]:
        """
        Get the number of channels of the features extracted by the model, from lower-level to higher-level features.
        Example: [96, 270, 1080, 2160, 4320] means that the first lower-level feature has 96 channels, the second 270, and so on.
        """
        return [feature_info["num_chs"] for feature_info in self.pnas.feature_info]

    def _get_feature_sizes(self) -> List[int]:
        """
        Get the spatial size of the features extracted by the model, from lower-level to higher-level features.
        Example: [165, 83, 42, 21, 11] means that the first lower-level feature has a spatial size of 165x165, the second 83x83, and so on.

        Returns:
            List[int]: The sizes of the features extracted by the model.
        """
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        image_features = self.pnas(x)
        feature_sizes = [image_feature.size()[-1] for image_feature in image_features]

        return feature_sizes

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the image encoder.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            List[torch.Tensor]: The features extracted from the image, from lower-level to higher-level features.
        """
        if x.shape[-1] != IMAGE_SIZE or x.shape[-2] != IMAGE_SIZE:
            raise ValueError(
                f"❌ Input image size must be {IMAGE_SIZE}x{IMAGE_SIZE}, but got {x.shape[-2]}x{x.shape[-1]}"
            )

        image_features = self.pnas(x)

        return image_features
