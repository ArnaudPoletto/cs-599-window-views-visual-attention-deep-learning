import torch
from torch import nn
from typing import List, Optional

from src.models.depth_decoder import DepthDecoder
from src.models.depth_encoder import DepthEncoder
from src.models.image_encoder import ImageEncoder
from src.models.depth_estimator import DepthEstimator
from src.models.graph_processor import GraphProcessor
from src.models.livesal_decoder import LiveSALDecoder
from src.models.spatio_temporal_mixing_module import SpatioTemporalMixingModule
from src.config import SEQUENCE_LENGTH, IMAGE_SIZE


class LiveSAL(nn.Module):
    def __init__(
        self,
        image_n_levels: int,
        freeze_encoder: bool,
        freeze_temporal_pipeline: bool,
        hidden_channels: int,
        neighbor_radius: int,
        n_iterations: int,
        image_hidden_channels_list: List[int],
        depth_hidden_channels_list: List[int],
        output_type: str,
        dropout_rate: float,
        with_graph_processing: bool,
        with_graph_edge_features: bool,
        with_graph_positional_embeddings: bool,
        with_graph_directional_kernels: bool,
        with_depth_information: bool,
        eps: float = 1e-6,
    ) -> None:
        if image_n_levels < 1:
            raise ValueError(f"❌ Invalid number of image levels: {image_n_levels}")
        if neighbor_radius < 1:
            raise ValueError(f"❌ Invalid neighbor radius: {neighbor_radius}")
        if n_iterations < 1:
            raise ValueError(f"❌ Invalid number of iterations: {n_iterations}")
        if output_type not in ["temporal", "global", "global_direct"]:
            raise ValueError(f"❌ Invalid output type: {output_type}")
        if freeze_temporal_pipeline and output_type == "temporal":
            raise ValueError("❌ Cannot freeze the temporal pipeline when output type is temporal.")
        
        super(LiveSAL, self).__init__()

        self.image_n_levels = image_n_levels
        self.freeze_encoder = freeze_encoder
        self.freeze_temporal_pipeline = freeze_temporal_pipeline
        self.hidden_channels = hidden_channels
        self.neighbor_radius = neighbor_radius
        self.n_iterations = n_iterations
        projection_channels_list = [8, 16, 32, 64, 128] # TODO: change this
        self.projection_channels_list = projection_channels_list
        self.image_hidden_channels_list = image_hidden_channels_list
        self.depth_hidden_channels_list = depth_hidden_channels_list
        self.dropout_rate = dropout_rate
        self.with_graph_processing = with_graph_processing
        self.with_graph_edge_features = with_graph_edge_features
        self.with_graph_positional_embeddings = with_graph_positional_embeddings
        self.with_graph_directional_kernels = with_graph_directional_kernels
        self.with_depth_information = with_depth_information
        self.output_type = output_type
        self.eps = eps

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

        if with_depth_information:
            self.depth_estimator = DepthEstimator(freeze=True)
            self.depth_encoder = DepthEncoder(
                hidden_channels=hidden_channels,
            )
            depth_last_layer_channels = self.depth_encoder.features_channels_list[-1]
            depth_last_layer_size = self.depth_encoder.features_sizes[-1]
            self.depth_graph_processor = GraphProcessor(
                channels=depth_last_layer_channels,
                size=depth_last_layer_size,
                neighbor_radius=neighbor_radius,
                n_iterations=n_iterations,
                dropout_rate=dropout_rate,
                with_edge_features=with_graph_edge_features,
                with_positional_embeddings=with_graph_positional_embeddings,
                with_directional_kernels=with_graph_directional_kernels,
            )
            self.depth_decoder = DepthDecoder(
                features_channels_list=self.depth_encoder.features_channels_list,
                hidden_channels_list=depth_hidden_channels_list,
                features_sizes=self.depth_encoder.features_sizes,
                output_channels=hidden_channels,
                dropout_rate=dropout_rate,
            )

        self.image_encoder = ImageEncoder(
            freeze=freeze_encoder or freeze_temporal_pipeline,
            n_levels=image_n_levels,
        )

        projection_in_channels_list = self.image_encoder.feature_channels_list
        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=(in_channels + out_channels) // 2,
                    kernel_size=1,
                    bias=True,
                ),
                nn.GroupNorm(LiveSAL.get_n_groups((in_channels + out_channels) // 2), (in_channels + out_channels) // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(
                    in_channels=(in_channels + out_channels) // 2,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.GroupNorm(LiveSAL.get_n_groups(out_channels), out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
            )
            for in_channels, out_channels in zip(projection_in_channels_list, projection_channels_list)
        ])

        image_last_layer_channels = projection_channels_list[-1]
        image_last_layer_size = self.image_encoder.feature_sizes[-1]
        if with_graph_processing:
            self.image_graph_processor = GraphProcessor(
                channels=image_last_layer_channels,
                size=image_last_layer_size,
                neighbor_radius=neighbor_radius,
                n_iterations=n_iterations,
                dropout_rate=self.dropout_rate,
                with_edge_features=with_graph_edge_features,
                with_positional_embeddings=with_graph_positional_embeddings,
                with_directional_kernels=with_graph_directional_kernels,
            )

        depth_channels = None
        if with_depth_information:
            depth_channels = self.depth_encoder.features_channels_list[-1]
        self.temporal_decoder = LiveSALDecoder(
            features_channels_list=projection_channels_list,
            hidden_channels_list=image_hidden_channels_list,
            depth_channels=depth_channels,
            output_channels=1,
            dropout_rate=dropout_rate,
            with_depth_information=with_depth_information,
            use_pooled_features=False,
        )

        self.global_decoder = LiveSALDecoder(
            features_channels_list=projection_channels_list,
            hidden_channels_list=image_hidden_channels_list,
            depth_channels=depth_channels,
            output_channels=1,
            dropout_rate=dropout_rate,
            with_depth_information=with_depth_information,
            use_pooled_features=True,
        )

        self.spatio_temporal_mixing_module = SpatioTemporalMixingModule(
            hidden_channels_list=image_hidden_channels_list,
            feature_channels_list=projection_channels_list,
            dropout_rate=dropout_rate,
        )

        self.sigmoid = nn.Sigmoid()

        if self.freeze_temporal_pipeline:
            if with_depth_information:
                    for param in self.depth_encoder.parameters():
                        param.requires_grad = False
                    for param in self.depth_graph_processor.parameters():
                        param.requires_grad = False
                    for param in self.depth_decoder.parameters():
                        param.requires_grad = False
            if with_graph_processing:
                for param in self.image_graph_processor.parameters():
                    param.requires_grad = False
            for param in self.temporal_decoder.parameters():
                param.requires_grad = False

        if self.output_type == "temporal":
            for param in self.global_decoder.parameters():
                param.requires_grad = False
            for param in self.spatio_temporal_mixing_module.parameters():
                param.requires_grad = False

        if self.output_type == "global_direct":
            for param in self.temporal_decoder.parameters():
                param.requires_grad = False
            for param in self.spatio_temporal_mixing_module.parameters():
                param.requires_grad = False

    @staticmethod
    def get_n_groups(n_channels: int, min_factor: float = 4) -> int:
        max_groups = max(1, n_channels // min_factor)
        
        # Try to find the largest divisor of num_channels that is smaller than num_channels
        for groups in range(max_groups, 0, -1):
            if n_channels % groups == 0:
                return groups
        
        return n_channels

    def _normalize_input(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        x = x.clone()
        normalized_x = (x - mean) / (std + self.eps)

        return normalized_x
    
    def _normalize_spatial_dimensions(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        batch_size, channels , height, width = x.size()
        x = x.view(batch_size, channels, -1)
        x = x / (x.max(dim=2, keepdim=True)[0] + self.eps)
        x = x.view(batch_size, channels, height, width)

        return x

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

        # Project features
        for i, projection_layer in enumerate(self.projection_layers):
            image_features_list[i] = projection_layer(image_features_list[i])

        # Expand features for image inputs to match the sequence length
        if is_image:
            new_image_features_list = []
            for image_features in image_features_list:
                image_features = image_features.unsqueeze(1).repeat(
                    1, SEQUENCE_LENGTH, 1, 1, 1
                )
                batch_size, sequence_length, channels, height, width = image_features.shape
                image_features = image_features.view(-1, channels, height, width)
                new_image_features_list.append(image_features)
            image_features_list = new_image_features_list

        return image_features_list
    
    def _get_graph_features(
        self, features: torch.Tensor, graph_processor: GraphProcessor
    ) -> torch.Tensor:
        batch_size_sequence_length, channels, height, width = features.shape
        transformed_features = (
            features.view(-1, SEQUENCE_LENGTH, channels, height, width)
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
        graph_features = graph_features + features

        return graph_features

    def _get_depth_features_list(
        self, x: torch.Tensor, is_image: bool
    ) -> torch.Tensor:
        if not is_image:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)

        # Normalize and get depth features
        x_depth = self._normalize_input(x, self.depth_mean, self.depth_std)
        depth_estimation = self.depth_estimator(x_depth)
        depth_features_list = self.depth_encoder(depth_estimation)

        if is_image:
            new_depth_features_list = []
            for depth_features in depth_features_list:
                depth_features = depth_features.unsqueeze(1).repeat(
                    1, SEQUENCE_LENGTH, 1, 1, 1
                )
                batch_size, sequence_length, channels, height, width = (
                    depth_features.shape
                )
                depth_features = depth_features.view(
                    -1, channels, height, width
                )
                new_depth_features_list.append(depth_features)
            depth_features_list = new_depth_features_list

        return depth_features_list
    
    def _forward_temporal_pipeline(
        self,
        x: torch.Tensor,
        is_image: bool,
    ):
        # Get image features
        image_features_list = self._get_image_features_list(x, is_image)

        # Process features if needed
        if self.with_graph_processing:
            image_features_list[-1] = self._get_graph_features(
                features=image_features_list[-1],
                graph_processor=self.image_graph_processor,
            )

        if self.with_depth_information:
            depth_encoded_features_list = self._get_depth_features_list(x, is_image)
            if self.with_graph_processing:
                depth_encoded_features_list[-1] = self._get_graph_features(
                    features=depth_encoded_features_list[-1],
                    graph_processor=self.depth_graph_processor,
                )
            depth_decoded_features = self.depth_decoder(depth_encoded_features_list)
        else:
            depth_decoded_features = None

        # Get temporal output
        temporal_features = self.temporal_decoder(image_features_list, depth_decoded_features)
        temporal_output = self.sigmoid(temporal_features)
        temporal_output = self._normalize_spatial_dimensions(temporal_output)

        return image_features_list, depth_decoded_features, temporal_features, temporal_output

    def _forward_global_pipeline(self, image_features_list: List[torch.Tensor], depth_decoded_features: Optional[torch.Tensor], temporal_features: torch.Tensor):
        # Reshape tensors
        image_features_list = [image_features.view(-1, SEQUENCE_LENGTH, *image_features.shape[1:]) for image_features in image_features_list]
        # Pool temporal features
        pooled_image_features_list = [torch.mean(image_features, dim=1) for image_features in image_features_list]
        if depth_decoded_features is not None:
            depth_decoded_features = depth_decoded_features.view(-1, SEQUENCE_LENGTH, *depth_decoded_features.shape[1:])
            pooled_depth_decoded_features = torch.mean(depth_decoded_features, dim=1)
        else:
            pooled_depth_decoded_features = None

        # Get global features
        global_features = self.global_decoder(pooled_image_features_list, pooled_depth_decoded_features)

        # Get global output from spatio-temporal mixing module
        global_output = self.spatio_temporal_mixing_module(
            encoded_features_list=pooled_image_features_list,
            temporal_features=temporal_features,
            global_features=global_features,
        )
        global_output = self._normalize_spatial_dimensions(global_output).squeeze(1)

        return global_output
    
    def _forward_global_direct_pipeline(self, x: torch.Tensor, is_image: bool):
        # Get image features
        image_features_list = self._get_image_features_list(x, is_image)

        # Process features if needed
        if self.with_graph_processing:
            image_features_list[-1] = self._get_graph_features(
                features=image_features_list[-1],
                graph_processor=self.image_graph_processor,
            )

        if self.with_depth_information:
            depth_encoded_features_list = self._get_depth_features_list(x, is_image)
            if self.with_graph_processing:
                depth_encoded_features_list[-1] = self._get_graph_features(
                    features=depth_encoded_features_list[-1],
                    graph_processor=self.depth_graph_processor,
                )
            depth_decoded_features = self.depth_decoder(depth_encoded_features_list)
        else:
            depth_decoded_features = None

        # Reshape tensors
        image_features_list = [image_features.view(-1, SEQUENCE_LENGTH, *image_features.shape[1:]) for image_features in image_features_list]
        # Pool temporal features
        pooled_image_features_list = [torch.mean(image_features, dim=1) for image_features in image_features_list]
        if depth_decoded_features is not None:
            depth_decoded_features = depth_decoded_features.view(-1, SEQUENCE_LENGTH, *depth_decoded_features.shape[1:])
            pooled_depth_decoded_features = torch.mean(depth_decoded_features, dim=1)
        else:
            pooled_depth_decoded_features = None

        # Get global features
        global_features = self.global_decoder(pooled_image_features_list, pooled_depth_decoded_features)
        global_output = self.sigmoid(global_features)
        global_output = self._normalize_spatial_dimensions(global_output).squeeze(1)

        return global_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.dim() not in [4, 5]:
            raise ValueError(
                f"❌ Input tensor must be of shape (batch_size, channels, height, width) or (batch_size, sequence_length, channels, height, width), got {x.shape}."
            )
        if x.dim() == 5 and x.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"❌ Video input tensor must have {SEQUENCE_LENGTH} channels, got {x.shape[1]}."
            )
        if x.shape[-3] != 3:
            raise ValueError(
                f"❌ Input tensor must have 3 channels, got {x.shape[1]}."
            )
        if x.shape[-2] != IMAGE_SIZE or x.shape[-1] != IMAGE_SIZE:
            raise ValueError(
                f"❌ Input tensor must have spatial dimensions of {IMAGE_SIZE}x{IMAGE_SIZE}, got {x.shape[-2]}x{x.shape[-1]}."
            )
        is_image = x.dim() == 4

        if self.output_type == "global":
            image_features_list, depth_decoded_features, temporal_features, _ = self._forward_temporal_pipeline(x, is_image)
            global_output = self._forward_global_pipeline(image_features_list, depth_decoded_features, temporal_features)
            return None, global_output
        elif self.output_type == "temporal":
            _, _, _, temporal_output = self._forward_temporal_pipeline(x, is_image)
            return temporal_output, None
        elif self.output_type == "global_direct":
            global_output = self._forward_global_direct_pipeline(x, is_image)
            return None, global_output