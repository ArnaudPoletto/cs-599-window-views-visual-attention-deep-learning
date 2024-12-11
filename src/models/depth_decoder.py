import torch
from torch import nn
from typing import List

from src.config import IMAGE_SIZE


class DepthDecoder(nn.Module):
    def __init__(
        self,
        features_channels_list: List[int],
        hidden_channels_list: List[int],
        features_sizes: List[int],
        output_channels: int,
        dropout_rate: float,
    ) -> None:
        super(DepthDecoder, self).__init__()

        # Save parameters
        self.features_channels_list = features_channels_list
        self.hidden_channels_list = hidden_channels_list
        self.features_sizes = features_sizes
        self.output_channels = output_channels
        self.dropout_rate = dropout_rate

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
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                )
                for in_channels, inc_channels, out_channels in zip(
                    in_channels_list, inc_channels_list, out_channels_list
                )
            ]
        )

        # Final layer
        final_channels = out_channels_list[-1]
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=final_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the depth decoder.

        Args:
            xs (List[torch.Tensor]): List of feature tensors to decode, ordered from lowest
                                   to highest resolution

        Returns:
            torch.Tensor: The decoded depth map
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
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        output = self.final_layer(output)

        return output
