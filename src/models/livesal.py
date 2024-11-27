import torch
from torch import nn
from typing import List, Tuple

from src.models.image_encoder import ImageEncoder
from src.models.depth_estimator import DepthEstimator
from src.models.graph_processor import GraphProcessor
from src.models.spatio_temporal_mixing_module import SpatioTemporalMixingModule
from src.config import SEQUENCE_LENGTH, IMAGE_SIZE


class LiveSAL(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        neighbor_radius: int,
        n_iterations: int,
        with_graph_processing: bool,
        freeze_encoder: bool,
        with_depth_information: bool,
        dropout_rate: float,
    ):
        super(LiveSAL, self).__init__()

        self.hidden_channels = hidden_channels
        self.neighbor_radius = neighbor_radius
        self.n_iterations = n_iterations
        self.with_graph_processing = with_graph_processing
        self.freeze_encoder = freeze_encoder
        self.with_depth_information = with_depth_information
        self.dropout_rate = dropout_rate

        # Get normalization parameters for encoder/estimator inputs
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

        self.image_encoder = ImageEncoder(
            freeze=freeze_encoder,
        )
        self.fusion_level = len(self.image_encoder.feature_sizes)

        if with_depth_information:
            self.depth_estimator = DepthEstimator(
                freeze=freeze_encoder,
            )

        self.image_projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=max(in_ch // 2, hidden_channels),
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(max(in_ch // 2, hidden_channels)),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(
                        in_channels=max(in_ch // 2, hidden_channels),
                        out_channels=hidden_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                )
                for in_ch in self.image_encoder.feature_channels_list
            ]
        )

        if with_graph_processing:
            last_feature_size = self.image_encoder.feature_sizes[-1]
            self.graph_processor = GraphProcessor(
                hidden_channels=hidden_channels,
                neighbor_radius=neighbor_radius,
                fusion_size=last_feature_size,
                n_iterations=n_iterations,
                dropout_rate=self.dropout_rate,
            )

        self.temporal_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_channels * 2,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                )
                for _ in range(self.fusion_level - 1)
            ]
        )
        self.final_temporal_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels // 2,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        self.final_global_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=SEQUENCE_LENGTH,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def _normalize_input(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        x = x.clone()
        normalized_x = ((x / 255.0) - mean) / std

        return normalized_x

    def _get_image_features_list(
        self, x: torch.Tensor, is_image: bool
    ) -> List[torch.Tensor]:
        # Flatten batch_size and sequence_length for video inputs to pass through the encoder
        if not is_image:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)

        # Normalize and get image features
        x_image = self._normalize_input(x, self.image_mean, self.image_std)
        image_features_list = self.image_encoder(x_image)

        return image_features_list

    def _get_image_projected_features_list(
        self,
        image_features_list: List[torch.Tensor],
        is_image: bool,
    ) -> List[torch.Tensor]:
        image_projected_features_list = []
        for image_features, image_projection_layer in zip(
            image_features_list, self.image_projection_layers
        ):
            image_projected_features = image_projection_layer(image_features)

            if is_image:
                image_projected_features = image_projected_features.unsqueeze(1).repeat(
                    1, SEQUENCE_LENGTH, 1, 1, 1
                )
                batch_size, sequence_length, channels, height, width = image_projected_features.shape
                image_projected_features = image_projected_features.view(-1, channels, height, width)
            image_projected_features_list.append(image_projected_features)

        return image_projected_features_list
    
    def _get_graph_features(
        self, image_features: torch.Tensor, graph_processor: GraphProcessor
    ) -> torch.Tensor:
        batch_size_sequence_length, channels, height, width = image_features.shape
        transformed_features = (
            image_features.view(-1, SEQUENCE_LENGTH, channels, height, width)
            .transpose(0, 1)
            .contiguous()
        )

        graph_features = graph_processor(transformed_features)
        graph_features = (
            graph_features.transpose(0, 1)
            .contiguous()
            .view(batch_size_sequence_length, channels, height, width)
        )

        # Add residual connection
        graph_features = graph_features + image_features

        return graph_features

    def _get_temporal_features(
        self,
        image_features_list: List[torch.Tensor],
    ) -> torch.Tensor:
        # Decode the features
        # Start with the last 2 features and go backwards
        x = image_features_list[-1]
        for i, temporal_layer in enumerate(self.temporal_layers):
            y = image_features_list[-(i + 2)]
            x = nn.functional.interpolate(
                x, size=y.shape[-2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, y], dim=1)
            x = temporal_layer(x)

        # Get the final decoded features
        decoded_features = nn.functional.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        decoded_features = self.final_temporal_layer(decoded_features).squeeze(1)

        # Get temporal output from decoded features
        batch_size_sequence_length, height, width = decoded_features.shape
        temporal_features = decoded_features.view(-1, SEQUENCE_LENGTH, height, width)

        return temporal_features

    def _get_global_output(
        self,
        temporal_features: torch.Tensor,
    ) -> torch.Tensor:
        global_output = self.final_global_layer(temporal_features).squeeze(1)

        return global_output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in [4, 5]:
            raise ValueError(
                f"❌ Input tensor must be of shape (batch_size, channels, height, width) or (batch_size, sequence_length, channels, height, width), got {x.shape}."
            )
        if x.dim() == 5 and x.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"❌ Input tensor must have {SEQUENCE_LENGTH} channels, got {x.shape[1]}."
            )
        is_image = x.dim() == 4

        # Get image features
        image_features_list = self._get_image_features_list(x, is_image)

        # Project features and get skip features
        image_features_list = self._get_image_projected_features_list(
            image_features_list=image_features_list,
            is_image=is_image,
        )

        # Process features if needed
        if self.with_graph_processing:
            image_features_list[-1] = self._get_graph_features(
                image_features=image_features_list[-1],
                graph_processor=self.graph_processor,
            )

        # Get temporal output
        temporal_features = self._get_temporal_features(
            image_features_list=image_features_list,
        )
        temporal_output = self.sigmoid(temporal_features)

        global_output = self._get_global_output(
            temporal_features=temporal_features,
        )

        return temporal_output, global_output
