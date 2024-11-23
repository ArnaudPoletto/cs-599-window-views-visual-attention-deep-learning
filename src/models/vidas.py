import math
import torch
import torch.nn as nn
from typing import Tuple, List


SALIENCY_OUT_CHANNELS = 64
ATTENTION_OUT_CHANNELS = 16


class DSAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        saliency_out_channels: int,
        attention_out_channels: int,
    ) -> None:
        super(DSAM, self).__init__()

        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.saliency_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=saliency_out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.attention_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=attention_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=attention_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=attention_out_channels,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        )
        # TODO: this upscale should scale with depth, i.e. do * 2 or * 4 etc...
        self.upsample = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Temporal average pooling
        x_pool = self.temporal_pool(x).squeeze(2)

        # Generate saliency features
        saliency_maps = self.saliency_conv(x_pool)

        # Generate attention maps
        attention = self.attention_layer(x_pool)
        attention_map = torch.softmax(attention.flatten(2), dim=2).view_as(attention)
        activation_map = self.upsample(attention)  # TODO: not upsampled as needed yet

        # Enhance features
        enhanced = (1 + attention_map.unsqueeze(2)) * x

        return enhanced, saliency_maps, activation_map


class ViDaSEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_shape: Tuple[int, int, int],
        hidden_channels_list: List[int],
        kernel_sizes: List[int],
        use_max_pooling_list: List[bool],
        saliency_out_channels: int,
        attention_out_channels: int,
    ) -> None:
        super(ViDaSEncoder, self).__init__()

        if len(hidden_channels_list) != len(kernel_sizes) or len(
            hidden_channels_list
        ) != len(use_max_pooling_list):
            raise ValueError(
                f"❌ Length of hidden_channels_list, kernel_sizes and use_max_pooling must be equal, got {len(hidden_channels_list)}, {len(kernel_sizes)} and {len(use_max_pooling_list)}."
            )

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.hidden_channels_list = hidden_channels_list
        self.kernel_sizes = kernel_sizes
        self.use_max_pooling_list = use_max_pooling_list

        in_channels_list = [input_channels] + hidden_channels_list[:-1]
        out_channels_list = hidden_channels_list
        self.encoder_blocks = nn.ModuleList()
        for in_channels, out_channels, kernel_size, use_max_pooling in zip(
            in_channels_list, out_channels_list, kernel_sizes, use_max_pooling_list
        ):
            if use_max_pooling:
                encoder_block = nn.Sequential(
                    nn.MaxPool3d(kernel_size=(1, 2, 2)),
                    self._make_encoder_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=self._get_padding(kernel_size),
                    ),
                )
            else:
                encoder_block = self._make_encoder_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 2, 2),
                    padding=self._get_padding(kernel_size),
                )
            self.encoder_blocks.append(encoder_block)

        self.dsams = nn.ModuleList(
            [
                DSAM(
                    in_channels=hidden_channels,
                    saliency_out_channels=saliency_out_channels,
                    attention_out_channels=attention_out_channels,
                )
                for hidden_channels in hidden_channels_list
            ]
        )

    def _get_padding(self, kernel_size: int) -> int:
        return kernel_size // 2

    def _make_encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Tuple[int, int, int],
        padding: int,
    ) -> nn.Sequential:
        encoder_block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )

        return encoder_block

    def get_saliency_map_shapes(self) -> List[Tuple[int, int]]:
        saliency_map_shapes = [
            math.ceil(self.input_shape[1] / 2 ** (i + 1))
            for i in range(len(self.hidden_channels_list))
        ]
        saliency_map_shapes = [(s, s) for s in saliency_map_shapes]

        return saliency_map_shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"❌ Input tensor has {x.shape[1]} channels, expected {self.input_channels}."
            )
        if x.shape[2:] != self.input_shape:
            raise ValueError(
                f"❌ Input tensor has shape {x.shape[2:]}, expected {self.input_shape}."
            )
        
        saliency_maps_list = []
        for encoder_block, dsam in zip(self.encoder_blocks, self.dsams):
            x = encoder_block(x)
            x, saliency_maps, _ = dsam(x)
            saliency_maps_list.append(saliency_maps)

        return saliency_maps_list


class ViDaSDecoder(nn.Module):
    def __init__(
        self,
        saliency_map_shapes: List[Tuple[int, int]],
        saliency_out_channels: int,
    ) -> None:
        super(ViDaSDecoder, self).__init__()

        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        for saliency_map_shape in reversed(saliency_map_shapes[:-1]):
            upsample = nn.Upsample(
                size=saliency_map_shape, mode="bilinear", align_corners=False
            )
            self.upsamples.append(upsample)

            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=saliency_out_channels * 2,
                    out_channels=saliency_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=saliency_out_channels),
            )
            self.convs.append(conv)

    def forward(self, saliency_maps_list: List[torch.Tensor]) -> torch.Tensor:
        x = saliency_maps_list[-1]
        y = saliency_maps_list[-2]

        for i, (upsample, conv) in enumerate(zip(self.upsamples, self.convs)):
            x = upsample(x)
            x = torch.cat([x, y], dim=1)
            x = conv(x)
            if i < len(self.upsamples) - 1:
                y = saliency_maps_list[-(i + 3)]

        return x


class ViDaS(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        input_shape: Tuple[int, int, int] = (5, 331, 331),
        hidden_channels_list: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [5, 3, 3, 3],
        use_max_pooling_list: List[bool] = [False, True, False, False],
        saliency_out_channels: int = SALIENCY_OUT_CHANNELS,
        attention_out_channels: int = ATTENTION_OUT_CHANNELS,
    ) -> None:
        super(ViDaS, self).__init__()

        self.encoder = ViDaSEncoder(
            input_channels=input_channels,
            input_shape=input_shape,
            hidden_channels_list=hidden_channels_list,
            kernel_sizes=kernel_sizes,
            use_max_pooling_list=use_max_pooling_list,
            saliency_out_channels=saliency_out_channels,
            attention_out_channels=attention_out_channels,
        )
        saliency_map_shapes = self.encoder.get_saliency_map_shapes()
        self.decoder = ViDaSDecoder(
            saliency_map_shapes=saliency_map_shapes,
            saliency_out_channels=saliency_out_channels,
        )

        self.final_layer = nn.Sequential(
            nn.Upsample(size=input_shape[1:], mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=saliency_out_channels,
                out_channels=saliency_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=saliency_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=saliency_out_channels,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to match 3D input: channels come before sequence length
        x = x.transpose(1, 2)

        saliency_maps_list = self.encoder(x)
        decoded_maps = self.decoder(saliency_maps_list)
        output = self.final_layer(decoded_maps)

        return output.squeeze(1)
