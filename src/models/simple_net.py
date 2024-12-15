import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import timm
import torch
from torch import nn
from typing import List

from src.config import IMAGE_SIZE, SIMPLE_NET_N_LEVELS, SIMPLE_NET_MODEL_NAME, SIMPLE_NET_PRETRAINED


class SimpleNet(nn.Module):

    def __init__(
        self,
        freeze: bool,
        n_levels: int = SIMPLE_NET_N_LEVELS,
        model_name: str = SIMPLE_NET_MODEL_NAME,
        pretrained: bool = SIMPLE_NET_PRETRAINED,
    ) -> None:
        if n_levels < 1:
            raise ValueError(f"❌ Number of levels must be at least 1, but got {n_levels}")
        
        super(SimpleNet, self).__init__()

        self.freeze = freeze
        self.n_levels = n_levels
        self.model_name = model_name
        self.pretrained = pretrained

        self.simple_net = timm.create_model(
            model_name=model_name, pretrained=pretrained, features_only=True
        )
        self.feature_channels_list = self._get_feature_channels_list()
        self.feature_sizes = self._get_feature_sizes()

        if freeze:
            for param in self.simple_net.parameters():
                param.requires_grad = False

    def _get_feature_channels_list(self) -> List[int]:
        return [feature_info["num_chs"] for feature_info in self.simple_net.feature_info][: self.n_levels]

    def _get_feature_sizes(self) -> List[int]:
        with torch.no_grad():
            x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            image_features = self.forward(x)
            feature_sizes = [image_feature.size()[-1] for image_feature in image_features]

        return feature_sizes

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[-1] != IMAGE_SIZE or x.shape[-2] != IMAGE_SIZE:
            raise ValueError(
                f"❌ Input image size must be {IMAGE_SIZE}x{IMAGE_SIZE}, but got {x.shape[-2]}x{x.shape[-1]}"
            )

        image_features = self.simple_net(x)[: self.n_levels]

        return image_features
