import torch
import torch.nn as nn
from typing import List, Optional

from src.config import IMAGE_SIZE, SEQUENCE_LENGTH

class LiveSALDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        n_levels: int,
        depth_integration: str,
        with_depth_information: bool,
        use_pooled_features: bool,
        dropout_rate: float,
    ) -> None:
        super(LiveSALDecoder, self).__init__()

        self.hidden_channels = hidden_channels
        self.n_levels = n_levels
        self.depth_integration = depth_integration
        self.with_depth_information = with_depth_information
        self.use_pooled_features = use_pooled_features
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_channels * 2,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(
                        num_groups=LiveSALDecoder._get_num_groups(hidden_channels, 32),
                        num_channels=hidden_channels,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(
                        num_groups=LiveSALDecoder._get_num_groups(hidden_channels, 32),
                        num_channels=hidden_channels,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                )
                for _ in range(n_levels - 1)
            ]
        )

        final_layer_in_channels = hidden_channels
        if with_depth_information and depth_integration in ["late", "both"]:
            final_layer_in_channels += hidden_channels
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=final_layer_in_channels,
                out_channels=hidden_channels // 2,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
            nn.GroupNorm(
                num_groups=LiveSALDecoder._get_num_groups(hidden_channels // 2, 16),
                num_channels=hidden_channels // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels // 2,
                out_channels=1,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
        )

    @staticmethod
    def _get_num_groups(num_channels, max_groups):
        num_groups = min(max_groups, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        return num_groups

    def forward(self, image_features_list: List[torch.Tensor], depth_decoded_features: Optional[torch.Tensor]) -> torch.Tensor:
        # Decode the features
        # Start with the last 2 features and go backwards
        x = image_features_list[-1]
        for i, layer in enumerate(self.layers):
            y = image_features_list[-(i + 2)]
            x = nn.functional.interpolate(
                x, size=y.shape[-2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, y], dim=1)
            x = layer(x)

        # Get the final decoded features
        decoded_features = nn.functional.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False
        )
        if self.with_depth_information and self.depth_integration in ["late", "both"]:
            print("decoded_features.shape", decoded_features.shape)
            print("depth_decoded_features.shape", depth_decoded_features.shape)
            decoded_features = torch.cat(
                [decoded_features, depth_decoded_features], dim=1
            )
        decoded_features = self.final_layer(decoded_features)

        if not self.use_pooled_features:
            decoded_features = decoded_features.squeeze(1)
            batch_size_sequence_length, height, width = decoded_features.shape
            decoded_features = decoded_features.view(-1, SEQUENCE_LENGTH, height, width)

        return decoded_features