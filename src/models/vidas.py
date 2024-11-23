import math
import torch
import torch.nn as nn
from typing import Tuple, List

from src.models.depth_estimator import DepthEstimator


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
        use_max_poolings: List[bool],
        saliency_out_channels: int,
        attention_out_channels: int,
    ) -> None:
        super(ViDaSEncoder, self).__init__()

        if len(hidden_channels_list) != len(kernel_sizes) or len(
            hidden_channels_list
        ) != len(use_max_poolings):
            raise ValueError(
                f"❌ Length of hidden_channels_list, kernel_sizes and use_max_poolings must be equal, got {len(hidden_channels_list)}, {len(kernel_sizes)} and {len(use_max_poolings)}."
            )

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.hidden_channels_list = hidden_channels_list
        self.kernel_sizes = kernel_sizes
        self.use_max_poolings = use_max_poolings

        in_channels_list = [input_channels] + hidden_channels_list[:-1]
        out_channels_list = hidden_channels_list
        self.encoder_blocks = nn.ModuleList()
        for in_channels, out_channels, kernel_size, use_max_pooling in zip(
            in_channels_list, out_channels_list, kernel_sizes, use_max_poolings
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
        input_channels: int,
        output_channels: int,
        input_shape: Tuple[int, int, int],
        hidden_channels_list: List[int],
        kernel_sizes: List[int],
        use_max_poolings: List[bool],
        saliency_out_channels: int,
        attention_out_channels: int,
        with_depth_information: bool,
    ) -> None:
        super(ViDaS, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_shape = input_shape
        self.hidden_channels_list = hidden_channels_list
        self.kernel_sizes = kernel_sizes
        self.use_max_poolings = use_max_poolings
        self.saliency_out_channels = saliency_out_channels
        self.attention_out_channels = attention_out_channels
        self.with_depth_information = with_depth_information

        self.image_encoder = ViDaSEncoder(
            input_channels=input_channels,
            input_shape=input_shape,
            hidden_channels_list=hidden_channels_list,
            kernel_sizes=kernel_sizes,
            use_max_poolings=use_max_poolings,
            saliency_out_channels=saliency_out_channels,
            attention_out_channels=attention_out_channels,
        )
        saliency_map_shapes = self.encoder.get_saliency_map_shapes()
        self.image_decoder = ViDaSDecoder(
            saliency_map_shapes=saliency_map_shapes,
            saliency_out_channels=saliency_out_channels,
        )
        self.image_upsample = nn.Upsample(
            size=input_shape[1:], mode="bilinear", align_corners=False
        )

        self.register_buffer(
            "image_mean",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
            persistent=False,
        )

        if with_depth_information:
            self.depth_estimator = DepthEstimator(freeze=True)
            self.depth_encoder = ViDaSEncoder(
                input_channels=1,
                input_shape=input_shape,
                hidden_channels_list=hidden_channels_list,
                kernel_sizes=kernel_sizes,
                use_max_poolings=use_max_poolings,
                saliency_out_channels=saliency_out_channels,
                attention_out_channels=attention_out_channels,
            )
            saliency_map_shapes = self.depth_encoder.get_saliency_map_shapes()
            self.depth_decoder = ViDaSDecoder(
                saliency_map_shapes=saliency_map_shapes,
                saliency_out_channels=saliency_out_channels,
            )
            self.depth_upsample = nn.Upsample(
                size=input_shape[1:], mode="bilinear", align_corners=False
            )

            self.register_buffer(
                "depth_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "depth_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )

        final_layer_in_channels = saliency_out_channels
        if with_depth_information:
            final_layer_in_channels *= 2
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=final_layer_in_channels,
                out_channels=saliency_out_channels // 2,
                kernel_size=3,
                padding=1,
                groups=saliency_out_channels // 2,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=saliency_out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=saliency_out_channels // 2,
                out_channels=output_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def _normalize_input(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        if x.max() > 1.0:
            x = x.float() / 255.0

        normalized_x = (x - mean) / std

        return normalized_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_depth_information:
            # Estimate depths and forward depth
            batch_size, sequence_length, channels, height, width = x.shape
            x_depth = x.clone().view(-1, channels, height, width)
            x_depth = self._normalize_input(x_depth, self.depth_mean, self.depth_std)
            depth_estimations = self.depth_estimator(x_depth)
            depth_estimations = depth_estimations.view(
                batch_size, sequence_length, height, width
            )
            depth_estimations = depth_estimations.unsqueeze(2).transpose(1, 2)
            depth_saliency_maps_list = self.depth_encoder(depth_estimations)
            depth_decoded_maps = self.depth_decoder(depth_saliency_maps_list)
            depth_decoded_maps = self.depth_upsample(depth_decoded_maps)

        # Forward image
        x_image = self._normalize_input(x, self.image_mean, self.image_std)
        x_image = x_image.transpose(1, 2)
        image_saliency_maps_list = self.image_encoder(x_image)
        image_decoded_maps = self.image_decoder(image_saliency_maps_list)
        image_decoded_maps = self.image_upsample(image_decoded_maps)

        # Final layer
        if self.with_depth_information:
            decoded_maps = torch.cat([image_decoded_maps, depth_decoded_maps], dim=1)
        else:
            decoded_maps = image_decoded_maps
        output = self.final_layer(decoded_maps)

        if self.output_channels == 1:
            output = output.squeeze(1)

        return output
