import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
from typing import List
from torch import nn

IMAGE_SIZE = 331


class ImageDecoder(nn.Module):
    """
    An image decoder that decodes features into an image.
    """

    def __init__(
        self,
        features_channels: List[int],  # [96, 270, 1080, 2160, 4320]
        hidden_channels: List[int],  # [128, 96, 270, 256, 512]
        features_sizes: List[int],  # [165, 83, 42, 21, 11]
        output_channels: int,
    ) -> None:
        """
        Initializes the image decoder.

        Args:
            features_channels (List[int]): The number of channels of the features to decode.
            hidden_channels (List[int]): The number of hidden channels to use.
            features_sizes (List[int]): The sizes of the features to decode.
            output_channels (int): The number of output channels.
        """
        super(ImageDecoder, self).__init__()

        self.features_channels = features_channels
        self.hidden_channels = hidden_channels
        self.features_sizes = features_sizes

        # The decoder layers
        in_channels = features_channels[::-1]
        inc_channels = [0] + hidden_channels[::-1]
        out_channels = hidden_channels[::-1]
        sizes = features_sizes[::-1][1:]
        self.deconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c + inc_c,
                        out_channels=out_c,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_c),
                    nn.ReLU(inplace=True),
                    nn.Upsample(
                        size=(size, size), mode="bilinear", align_corners=False
                    ),
                )
                for in_c, inc_c, out_c, size in zip(
                    in_channels, inc_channels, out_channels, sizes
                )
            ]
        )

        # The final layer
        final_in_channels = hidden_channels[0]
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=final_in_channels,
                out_channels=final_in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=final_in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_in_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the image decoder.

        Args:
            features (List[torch.Tensor]): The features to decode.

        Returns:
            torch.Tensor: The decoded image.
        """
        for i, (feature, deconv) in zip(features[::-1], self.deconvs):
            if i > 0:
                x = torch.cat((x, feature), 1)
            x = deconv(x)

        x = self.final_layer(x)

        return x.squeeze(1)
