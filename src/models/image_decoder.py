import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
from typing import List
from torch import nn

from src.config import IMAGE_SIZE


class ImageDecoder(nn.Module):
    """
    An image decoder that decodes features into an image.
    """

    def __init__(
        self,
        features_channels_list: List[int],
        hidden_channels_list: List[int],
        output_channels: int,
        dropout_rate: float,
        with_final_sigmoid: bool,
    ) -> None:
        """
        Initialize the image decoder.

        Args:
            features_channels (List[int]): The number of channels of the features to decode.
            hidden_channels (List[int]): The number of hidden channels to use, i.e. the number of output channels of the decoder layers.
            output_channels (int): The number of output channels.
            with_final_sigmoid (bool): Whether to apply a sigmoid activation to the output.
        """
        super(ImageDecoder, self).__init__()

        self.features_channels_list = features_channels_list
        self.hidden_channels_list = hidden_channels_list
        self.output_channels = output_channels
        self.dropout_rate = dropout_rate
        self.with_final_sigmoid = with_final_sigmoid

        # Get the decoder layers
        in_channels_list = [features_channels_list[-1]] + hidden_channels_list[1:][::-1]
        inc_channels_list = [features_channels_list[-2]] + features_channels_list[:-2][
            ::-1
        ]
        out_channels_list = hidden_channels_list[::-1]
        self.decoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels + inc_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(num_groups=ImageDecoder._get_num_groups(out_channels, 32), num_channels=out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                )
                for in_channels, inc_channels, out_channels in zip(
                    in_channels_list, inc_channels_list, out_channels_list
                )
            ]
        )

        # Get the final layer
        final_channels = out_channels_list[-1]
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=final_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(num_groups=ImageDecoder._get_num_groups(final_channels, 32), num_channels=final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=output_channels,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
            nn.Sigmoid() if with_final_sigmoid else nn.Identity(),
        )

    @staticmethod
    def _get_num_groups(num_channels, max_groups):
        num_groups = min(max_groups, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        return num_groups

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the image decoder.

        Args:
            features (List[torch.Tensor]): The features to decode.

        Returns:
            torch.Tensor: The decoded image.
        """
        # Start with the deepest feature
        x = xs[-1]

        # Decode features with skip connections
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Get skip connection feature
            y = xs[-(i + 2)]

            # Upsample current feature to match skip connection size
            x = nn.functional.interpolate(
                x, size=y.shape[-2:], mode="bilinear", align_corners=False
            )

            # Concatenate and process
            x = torch.cat([x, y], dim=1)
            x = decoder_layer(x)

        # Final upsampling to target size and processing
        output = nn.functional.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False
        )
        output = self.final_layer(output)

        return output
