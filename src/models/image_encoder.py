import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import timm
import torch
from torch import nn
from typing import List

MODEL_NAME = "pnasnet5large"
IMAGE_SIZE = 331

class ImageEncoder(nn.Module):
    """
    An image encoder that optionally uses a pretrained model to extract features from an image.
    """
    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        """
        Initializes the image encoder.
        
        Args:
            model_name (str): The name of the pretrained model to use.
            pretrained (bool): Whether to use pretrained weights.
        """
        super(ImageEncoder, self).__init__()

        self.pnas = timm.create_model(
            model_name=MODEL_NAME, pretrained=pretrained, features_only=True
        )

        self.feature_channels = [feature_info["num_chs"] for feature_info in self.pnas.feature_info]
        self.feature_sizes = self._get_feature_sizes()

    def _get_feature_sizes(self) -> List[int]:
        """
        Returns the sizes of the features extracted by the model.
        
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
            List[torch.Tensor]: The features extracted from the image.
        """
        image_features = self.pnas(x)

        return image_features
