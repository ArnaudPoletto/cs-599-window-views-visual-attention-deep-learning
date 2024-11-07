import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import timm
from torch import nn

MODEL_NAME = "pnasnet5large"


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: bool = True,
    ):
        super(ImageEncoder, self).__init__()

        self.pnas = timm.create_model(
            model_name=model_name, pretrained=pretrained, features_only=True
        )

    def forward(self, x):
        image_features = self.pnas(x)

        return image_features
