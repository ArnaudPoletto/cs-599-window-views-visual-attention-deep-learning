import torch
from torch import nn
from typing import List

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder
from src.config import SEQUENCE_LENGTH

IMAGE_SIZE = 331
HIDDEN_CHANNELS_LIST = [96, 270, 256, 512]

class SpatioTemporalMixingModule(nn.Module):
    def __init__(
        self,
        hidden_channels_list: List[int],
        feature_channels_list: List[int],
    ):
        super(SpatioTemporalMixingModule, self).__init__()

        self.hidden_channels_list = hidden_channels_list
        self.feature_channels_list = feature_channels_list

        in_channels_list = [feature_channels_list[-1]] + hidden_channels_list[1:][::-1]
        inc_channels_list = [feature_channels_list[-2]] + feature_channels_list[:-2][::-1]
        inc_channels_list = [inc_channels_list[0]] + [inc_channels + SEQUENCE_LENGTH + 1 for inc_channels in inc_channels_list[1:]]
        out_channels_list = hidden_channels_list[::-1]
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels + inc_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(),
                )
                for in_channels, inc_channels, out_channels in zip(in_channels_list, inc_channels_list, out_channels_list)
            ]
        )
        
        final_channels = out_channels_list[-1]
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=final_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        encoded_features: torch.Tensor,
        temporal_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        x = encoded_features[-1]
        for i, layer in enumerate(self.layers):
            y = encoded_features[-(i + 2)]
            if i == 0:
                x = nn.functional.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, y], dim=1)
            else:
                x = nn.functional.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)
                resized_global_features = nn.functional.interpolate(global_features, size=y.shape[-2:], mode='bilinear', align_corners=False)
                resized_temporal_features = nn.functional.interpolate(temporal_features, size=y.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, y, resized_global_features, resized_temporal_features], dim=1)
            x = layer(x)

        output = nn.functional.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        output = self.final_layer(output)

        return output


class TempSAL(nn.Module):
    def __init__(
        self,
        freeze_encoder: bool,
        hidden_channels_list: list[int] = HIDDEN_CHANNELS_LIST,
    ):
        super(TempSAL, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.hidden_channels_list = hidden_channels_list,
        
        self.image_encoder = ImageEncoder(
            freeze=freeze_encoder,
        )

        self.temporal_decoder = ImageDecoder(
            features_channels_list=self.image_encoder.feature_channels_list,
            hidden_channels_list=hidden_channels_list,
            features_sizes=self.image_encoder.feature_sizes,
            output_channels=SEQUENCE_LENGTH
        )
        self.global_decoder = ImageDecoder(
            features_channels_list=self.image_encoder.feature_channels_list,
            hidden_channels_list=hidden_channels_list,
            features_sizes=self.image_encoder.feature_sizes,
            output_channels=1
        )
        self.spatio_temporal_mixing_module = SpatioTemporalMixingModule(
            hidden_channels_list=hidden_channels_list,
            feature_channels_list=self.image_encoder.feature_channels_list,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_features = self.image_encoder(x)
        temporal_features = self.temporal_decoder(encoded_features)
        temporal_output = temporal_features
        global_features = self.global_decoder(encoded_features)
        global_output = self.spatio_temporal_mixing_module(
            encoded_features=encoded_features,
            temporal_features=temporal_features,
            global_features=global_features,
        )
        global_output = global_output.squeeze(1)

        return temporal_output, global_output
