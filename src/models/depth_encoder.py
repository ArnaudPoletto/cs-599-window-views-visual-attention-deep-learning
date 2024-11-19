import torch
import numpy as np
from torch import nn
from typing import List
from transformers import AutoModelForDepthEstimation

MODEL_NAME = "depth-anything/Depth-Anything-V2-Base-hf"
HIDDEN_LAYER_INDICES = [0, 6, 12]
IMAGE_SIZE = 256
HIDDEN_DIMENSION = 768
FEATURE_SIZE = 16

class DepthEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        hidden_layer_indices: List[int] = HIDDEN_LAYER_INDICES,
    ) -> None:
        super(DepthEncoder, self).__init__()

        self.pretrained = pretrained
        self.freeze = freeze
        self.hidden_layer_indices = hidden_layer_indices
        self.feature_channels = self._get_feature_channels()
        self.feature_sizes = self._get_feature_sizes()

        if pretrained:
            self.depth_anything = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME, output_hidden_states=True)
        else:
            self.depth_anything = AutoModelForDepthEstimation.from_config(config=MODEL_NAME, output_hidden_states=True)

        if freeze:
            for param in self.depth_anything.parameters():
                param.requires_grad = False

    def _get_feature_channels(self) -> List[int]:
        return [HIDDEN_DIMENSION] * len(self.hidden_layer_indices)
    
    def _get_feature_sizes(self) -> List[int]:
        return [FEATURE_SIZE] * len(self.hidden_layer_indices)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != IMAGE_SIZE or x.shape[-2] != IMAGE_SIZE:
            raise ValueError(f"❌ Input image size must be {IMAGE_SIZE}x{IMAGE_SIZE}, but got {x.shape[-2]}x{x.shape[-1]}.")
        
        batch_size = x.size(0)
        output = self.depth_anything(x)
        hidden_states = output.hidden_states
        predicted_depth = output.predicted_depth
        feature_maps = []
        for i, hidden_state in enumerate(hidden_states):
            if i not in self.hidden_layer_indices:
                continue

            patch_embeddings = hidden_state[:, 1:, :]
            feature_map = patch_embeddings.reshape(batch_size, FEATURE_SIZE, FEATURE_SIZE, -1).permute(0, 3, 1, 2)
            feature_maps.append(feature_map)

        return feature_maps, predicted_depth