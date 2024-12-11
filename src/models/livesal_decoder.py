import torch
import torch.nn as nn
from typing import List, Optional

from src.config import IMAGE_SIZE, SEQUENCE_LENGTH

class LiveSALDecoder(nn.Module):
    def __init__(
        self,
        features_channels_list: List[int],
        hidden_channels_list: List[int],
        output_channels: int,
        dropout_rate: float,
        with_depth_information: bool,
        use_pooled_features: bool,
    ) -> None:
        super(LiveSALDecoder, self).__init__()

        self.features_channels_list = features_channels_list
        self.hidden_channels_list = hidden_channels_list
        self.output_channels = output_channels
        self.dropout_rate = dropout_rate
        self.with_depth_information = with_depth_information
        self.use_pooled_features = use_pooled_features

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
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=output_channels,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
        )

    def forward(self, image_features_list: List[torch.Tensor], depth_decoded_features: Optional[torch.Tensor]) -> torch.Tensor:
        # Start with the deepest feature
        x = image_features_list[-1]

        # Decode features with skip connections
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Get skip connection feature
            y = image_features_list[-(i + 2)]

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

        if not self.use_pooled_features:
            output = output.squeeze(1)
            batch_size_sequence_length, height, width = output.shape
            output = output.view(-1, SEQUENCE_LENGTH, height, width)

        return output