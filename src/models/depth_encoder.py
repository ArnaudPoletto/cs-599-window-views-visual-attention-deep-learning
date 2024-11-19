import torch
import numpy as np
from torch import nn
from typing import List
from transformers import AutoModelForDepthEstimation

MODEL_NAME = "depth-anything/Depth-Anything-V2-Base-hf"

class DepthEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super(DepthEncoder, self).__init__()

        self.pretrained = pretrained
        self.freeze = freeze

        if pretrained:
            self.depth_anything = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME, output_hidden_states=True)
        else:
            self.depth_anything = AutoModelForDepthEstimation.from_config(config=MODEL_NAME, output_hidden_states=True)

        if freeze:
            for param in self.depth_anything.parameters():
                param.requires_grad = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.depth_anything(x)
        predicted_depth = output.predicted_depth

        return predicted_depth