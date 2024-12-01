import torch
from torch import nn
from typing import List

from src.config import SEQUENCE_LENGTH, IMAGE_SIZE


class SpatioTemporalMixingModule(nn.Module):
    """
    A spatio-temporal mixing module that combines the encoded features with the temporal and global features
    in the TempSAL model.
    """

    def __init__(
        self,
        hidden_channels_list: List[int],
        feature_channels_list: List[int],
    ) -> None:
        """
        Initialize the spatio-temporal mixing module.

        Args:
            hidden_channels_list (List[int]): The number of hidden channels to use, i.e. the number of output channels of the mixing module layers.
            feature_channels_list (List[int]): The number of channels of the features to mix.
        """
        super(SpatioTemporalMixingModule, self).__init__()

        self.hidden_channels_list = hidden_channels_list
        self.feature_channels_list = feature_channels_list

        # Get the decoder layers
        in_channels_list = [feature_channels_list[-1]] + hidden_channels_list[1:][::-1]
        inc_channels_list = [feature_channels_list[-2]] + feature_channels_list[:-2][
            ::-1
        ]
        inc_channels_list = [inc_channels_list[0]] + [
            inc_channels + SEQUENCE_LENGTH + 1 for inc_channels in inc_channels_list[1:]
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
                        bias=True,
                    ),
                    nn.ReLU(),
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
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=1,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        encoded_features_list: List[torch.Tensor],
        temporal_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the spatio-temporal mixing module.

        Args:
            encoded_features (torch.Tensor): The encoded features.
            temporal_features (torch.Tensor): The temporal features.
            global_features (torch.Tensor): The global features.

        Returns:
            torch.Tensor: The output of the spatio-temporal mixing module.
        """

        # Decode the features
        # Start with the last 2 encoded features and go backwards, concatenating the temporal and global saliency
        # features with the encoded features for the next steps
        x = encoded_features_list[-1]
        for i, decoder_layer in enumerate(self.decoder_layers):
            y = encoded_features_list[-(i + 2)]
            # Start with the first layer, where we only need to resize the encoded features
            if i == 0:
                x = nn.functional.interpolate(
                    x, size=y.shape[-2:], mode="bilinear", align_corners=False
                )
                x = torch.cat([x, y], dim=1)
            # For the other layers, we also need to resize the temporal and global features and concatenate them with
            # the encoded features
            else:
                x = nn.functional.interpolate(
                    x, size=y.shape[-2:], mode="bilinear", align_corners=False
                )
                resized_global_features = nn.functional.interpolate(
                    global_features,
                    size=y.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                resized_temporal_features = nn.functional.interpolate(
                    temporal_features,
                    size=y.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                x = torch.cat(
                    [x, y, resized_global_features, resized_temporal_features], dim=1
                )
            x = decoder_layer(x)

        # Get the final output
        output = nn.functional.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False
        )
        output = self.final_layer(output)

        return output
