import math
import torch
from torch import nn
from typing import List, Tuple, Optional

from src.models.image_encoder import ImageEncoder
from src.models.depth_encoder import DepthEncoder


class ConvGRU(nn.Module):
    def __init__(
        self, input_channels: int, hidden_channels: int, kernel_size: int, padding: int
    ):
        super(ConvGRU, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Reset gate
        self.conv_zr = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Update gate
        self.conv_h = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)

        # Compute reset and update gates
        zr = self.sigmoid(self.conv_zr(combined))
        z, r = torch.split(zr, self.hidden_channels, dim=1)

        # Compute candidate hidden state
        combined_r = torch.cat([x, r * h], dim=1)
        h_hat = torch.tanh(self.conv_h(combined_r))

        # Update hidden state
        h_new = (1 - z) * h + z * h_hat

        return h_new


class GraphProcessor(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        with_relative_positional_embeddings: bool,
        fusion_size: int,
        n_heads: int,
        neighbor_radius: int,
        n_iterations: int,
    ):
        super(GraphProcessor, self).__init__()

        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"❌ Hidden channels must be divisible by the number of heads, got {hidden_channels} and {n_heads}."
            )

        self.hidden_channels = hidden_channels
        self.with_relative_positional_embeddings = with_relative_positional_embeddings
        self.n_heads = n_heads
        self.neighbor_radius = neighbor_radius
        self.n_iterations = n_iterations
        self.scale = hidden_channels**-0.5
        self.head_dim = hidden_channels // n_heads

        self.intra_norm = nn.LayerNorm([hidden_channels, fusion_size, fusion_size])
        self.intra_key_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.intra_query_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.intra_value_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.intra_output = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.LayerNorm([hidden_channels, fusion_size, fusion_size]),
            nn.ReLU(inplace=True),
        )
        self.intra_alpha = nn.Parameter(torch.tensor(1.0))

        self.inter_norm = nn.LayerNorm([hidden_channels, fusion_size, fusion_size])
        if with_relative_positional_embeddings:
            self.relative_positional_embeddings = nn.Parameter(
                torch.randn(2 * neighbor_radius, 1, fusion_size, fusion_size)
            )
        inter_message_edge_in_channels = (
            hidden_channels + 2 + (1 * with_relative_positional_embeddings)
        )
        self.inter_message_edge_conv = nn.Conv2d(
            in_channels=inter_message_edge_in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.inter_query_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.inter_key_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.inter_value_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.inter_gate_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels * 2,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )
        self.inter_output = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.LayerNorm([hidden_channels, fusion_size, fusion_size]),
            nn.ReLU(inplace=True),
        )

        self.intra_inter_alpha = nn.Parameter(torch.tensor(0.5))

        self.gru = ConvGRU(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm = nn.LayerNorm(
            [hidden_channels, 11, 11]
        )  # TODO: remove hardocded size, fusion_size?

    def _compute_intra_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x = self.intra_norm(x)

        query = self.intra_query_conv(x).view(batch_size, channels, -1)
        key = self.intra_key_conv(x).view(batch_size, channels, -1)
        value = self.intra_value_conv(x).view(batch_size, channels, -1)

        attention = torch.bmm(query.transpose(1, 2), key) * self.scale
        attention = torch.softmax(attention, dim=-1)

        output = torch.bmm(attention, value.transpose(1, 2))
        output = output.transpose(1, 2).view(batch_size, channels, height, width)
        output = self.intra_output(output)
        output = self.intra_alpha * output + x

        return output

    def _get_temporal_encoding(
        self, relative_position: int, is_future: bool, device: torch.device
    ) -> torch.Tensor:
        direction = torch.tensor(1.0 if is_future else -1.0, device=device)
        distance = torch.tensor(float(abs(relative_position)), device=device)
        temporal_information = torch.stack([direction, distance])

        return temporal_information

    def _compute_inter_attention(
        self, i: int, x: torch.Tensor, neighbors: List[Tuple[int, torch.Tensor]]
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x = self.inter_norm(x)

        messages = []
        gates = []
        for j, y in neighbors:
            index = i - j + self.neighbor_radius - 1
            is_future = i < j

            # Add temporal encoding
            relative_position = (i - j) / self.neighbor_radius
            temporal_encoding = self._get_temporal_encoding(
                relative_position, is_future, x.device
            )
            temporal_encoding = temporal_encoding.view(1, -1, 1, 1).expand(
                batch_size, -1, height, width
            )
            y = torch.cat([y, temporal_encoding], dim=1)

            # Add relative positional embeddings
            if self.with_relative_positional_embeddings:
                spatial_encoding = self.relative_positional_embeddings[index].expand(
                    batch_size, -1, -1, -1
                )
                y = torch.cat([y, spatial_encoding], dim=1)
            y = self.inter_message_edge_conv(y)

            query = self.inter_query_conv(x).view(
                batch_size, self.n_heads, self.head_dim, -1
            )
            key = self.inter_key_conv(y).view(
                batch_size, self.n_heads, self.head_dim, -1
            )
            value = self.inter_value_conv(y).view(
                batch_size, self.n_heads, self.head_dim, -1
            )

            attention = torch.matmul(query.transpose(2, 3), key) * self.scale
            attention = torch.softmax(attention, dim=-1)

            message = torch.matmul(value, attention.transpose(2, 3))
            message = message.view(batch_size, channels, height, width)
            message = self.inter_output(message)
            gate = self.inter_gate_conv(torch.cat([x, message], dim=1))

            messages.append(message)
            gates.append(gate)

        if messages:
            messages = torch.stack(messages, dim=0)
            gates = torch.stack(gates, dim=0)
            gated_messages = messages * gates
            output = torch.sum(gated_messages, dim=0)
        else:
            output = torch.zeros_like(x)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length, batch_size, channels, height, width = x.shape
        h = x
        for _ in range(self.n_iterations):
            new_h = []
            for i in range(sequence_length):
                intra_output = self._compute_intra_attention(h[i])

                neighbors = [
                    (j, h[j])
                    for j in range(sequence_length)
                    if i != j and abs(i - j) <= self.neighbor_radius
                ]
                inter_output = self._compute_inter_attention(i, h[i], neighbors)
                combined_message = (
                    self.intra_inter_alpha * intra_output
                    + (1 - self.intra_inter_alpha) * inter_output
                )

                next_h = self.gru(combined_message, h[i])
                next_h = next_h + h[i]
                next_h = self.norm(next_h)
                new_h.append(next_h)

            h = torch.stack(new_h, dim=0)

        return h


class LiveSAL(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        with_absolute_positional_embeddings: bool,
        with_relative_positional_embeddings: bool,
        n_heads: int,
        neighbor_radius: int,
        n_iterations: int,
        with_graph_processing: bool,
        freeze_encoder: bool,
        with_depth_information: bool,
        fusion_level: Optional[int] = None,
    ):
        super(LiveSAL, self).__init__()

        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.with_absolute_positional_embeddings = with_absolute_positional_embeddings
        self.with_relative_positional_embeddings = with_relative_positional_embeddings
        self.n_heads = n_heads
        self.neighbor_radius = neighbor_radius
        self.n_iterations = n_iterations
        self.with_graph_processing = with_graph_processing
        self.freeze_encoder = freeze_encoder
        self.with_depth_information = with_depth_information

        # Get normalizations
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

        # Get encoder and freeze weights if needed
        self.image_encoder = ImageEncoder(
            freeze=freeze_encoder,
        )
        if with_depth_information:
            self.depth_encoder = DepthEncoder(
                freeze=freeze_encoder,
            )

        # Set fusion level and size
        if fusion_level is None:
            fusion_level = len(self.image_encoder.feature_sizes) // 2
            print(f"➡️ Default fusion level set to {fusion_level}.")
        if fusion_level < 0:
            raise ValueError(
                f"❌ Fusion level must be greater than or equal to 0, got {fusion_level}."
            )
        if fusion_level >= len(self.image_encoder.feature_sizes):
            raise ValueError(
                f"❌ Fusion level must be less than {len(self.image_encoder.feature_sizes)}, got {fusion_level}."
            )
        self.fusion_level = fusion_level
        self.fusion_size = self.image_encoder.feature_sizes[self.fusion_level]

        # Projection layers to project encoded features to a common space
        # Add 2 spatial awareness layers with depthwise separable 3x3 conv for efficiency
        self.image_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        groups=hidden_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        groups=hidden_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
                for in_ch in self.image_encoder.feature_channels
            ]
        )

        # Fusion layer to combine all features
        fusion_in_channels = hidden_channels * len(self.image_projections)
        if with_depth_information:
            fusion_in_channels += 1
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=fusion_in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        if with_absolute_positional_embeddings:
            self.absolute_positional_embeddings = nn.Parameter(
                torch.randn(
                    output_channels, hidden_channels, self.fusion_size, self.fusion_size
                )
            )

        if with_graph_processing:
            self.graph_processor = GraphProcessor(
                hidden_channels=hidden_channels,
                with_relative_positional_embeddings=with_relative_positional_embeddings,
                n_heads=n_heads,
                neighbor_radius=neighbor_radius,
                fusion_size=self.fusion_size,
                n_iterations=n_iterations,
            )

        # Decoder layers
        decoder_in_channels = hidden_channels * 2
        if with_depth_information:
            decoder_in_channels += 1
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=decoder_in_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(fusion_level - 1)
            ]
        )

        # Final layer
        self.final_layer = nn.Sequential(
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
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def _normalize_input(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, is_image: bool
    ) -> torch.Tensor:
        if x.max() > 1.0:
            x = x / 255.0
            
        normalized_x = (x - mean) / std

        return normalized_x

    def _get_image_features_list(
        self, x: torch.Tensor, is_image: bool
    ) -> List[torch.Tensor]:
        # Flatten batch_size and sequence_length for video inputs to pass through the encoder
        if not is_image:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)

        # Normalize and get image features
        x_image = self._normalize_input(x, self.image_mean, self.image_std, is_image)
        image_features_list = self.image_encoder(x_image)

        return image_features_list

    def _get_depth_features(
        self, x: torch.Tensor, is_image: bool
    ) -> List[torch.Tensor]:
        # Flatten batch_size and sequence_length for video inputs to pass through the
        if not is_image:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)

        # Normalize and get depth features
        x_depth = self._normalize_input(x, self.depth_mean, self.depth_std, is_image)
        depth_features = self.depth_encoder(x_depth).unsqueeze(1)

        # Reshape to original size
        depth_features = nn.functional.interpolate(
            depth_features,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        return depth_features

    def _get_projected_image_features_list(
        self,
        image_features_list: List[torch.Tensor],
        depth_features: Optional[torch.Tensor],
        is_image: bool,
    ) -> List[torch.Tensor]:
        # Get features list
        features_list = image_features_list

        # Get projections
        projections = self.image_projections

        projected_image_features_list = []
        for features, projection_layer in zip(features_list, projections):
            projected_features = projection_layer(features)
            projected_image_features_list.append(projected_features)

        # Get skip features
        skip_features_list = projected_image_features_list[: self.fusion_level]
        
        # Add depth information if needed
        if self.with_depth_information:
            new_skip_features_list = []
            for skip_features in skip_features_list:
                if depth_features.shape[-2:] != skip_features.shape[-2:]:
                    depth_features = nn.functional.interpolate(
                        depth_features,
                        size=skip_features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                new_skip_features_list.append(torch.cat([skip_features, depth_features], dim=1))
            skip_features_list = new_skip_features_list

        # Repeat image skip features for graph processing
        if is_image:
            resized_skip_features_list = []
            for skip_features in skip_features_list:
                batch_size, channels, height, width = skip_features.shape
                skip_features = skip_features.unsqueeze(1).repeat(
                    1, self.output_channels, 1, 1, 1
                )
                skip_features = skip_features.view(
                    batch_size * self.output_channels, channels, height, width
                )
                resized_skip_features_list.append(skip_features)
            skip_features_list = resized_skip_features_list

        return projected_image_features_list, skip_features_list

    def _get_fused_features(
        self, 
        projected_image_features_list: List[torch.Tensor], 
        depth_features: Optional[torch.Tensor],
        is_image: bool,
    ) -> torch.Tensor:
        # Resize features to middle scale
        base_size = (self.fusion_size, self.fusion_size)
        resized_features_list = []
        for image_features in projected_image_features_list:
            if image_features.shape[-2:] != base_size:
                image_features = nn.functional.interpolate(
                    image_features,
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_features_list.append(image_features)

        # Add depth information if needed
        if self.with_depth_information:
            if depth_features.shape[-2:] != base_size:
                depth_features = nn.functional.interpolate(
                    depth_features,
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_features_list.append(depth_features)

        # Fuse features
        fused_features = self.fusion(torch.cat(resized_features_list, dim=1))

        # Repeat image features for graph processing
        if is_image:
            batch_size, channels, height, width = fused_features.shape
            fused_features = fused_features.unsqueeze(1).repeat(
                1, self.output_channels, 1, 1, 1
            )
            fused_features = fused_features.view(
                batch_size * self.output_channels, channels, height, width
            )

        del resized_features_list
        torch.cuda.empty_cache()

        return fused_features

    def _add_absolute_positional_embeddings(
        self, fused_features: torch.Tensor
    ) -> torch.Tensor:
        batch_size_sequence_length, channels, height, width = fused_features.shape
        fused_features_list = fused_features.view(
            -1, self.output_channels, channels, height, width
        )
        fused_features = (
            fused_features_list + self.absolute_positional_embeddings.unsqueeze(0)
        ).view(batch_size_sequence_length, channels, height, width)

        return fused_features

    def _get_graph_features(self, fused_features: torch.Tensor) -> torch.Tensor:
        batch_size_sequence_length, channels, height, width = fused_features.shape
        fused_features_list = fused_features.view(
            self.output_channels, -1, channels, height, width
        )

        graph_features_list = self.graph_processor(fused_features_list)
        graph_features = graph_features_list.view(
            batch_size_sequence_length, channels, height, width
        )

        # Add residual connection
        graph_features = graph_features + fused_features

        return graph_features

    def _decode_features(
        self,
        fused_features: torch.Tensor,
        skip_features_list: List[torch.Tensor],
        is_image: bool,
    ) -> torch.Tensor:
        # Organize features as independent samples
        batch_size_sequence_length, channels, height, width = fused_features.shape
        fused_features_list = (
            fused_features.view(-1, self.output_channels, channels, height, width)
            .transpose(0, 1)
            .contiguous()
        )
        resized_skip_features_list = []
        for skip_features in skip_features_list:
            batch_size_sequence_length, channels, height, width = skip_features.shape
            skip_features = (
                skip_features.view(-1, self.output_channels, channels, height, width)
                .transpose(0, 1)
                .contiguous()
            )
            resized_skip_features_list.append(skip_features)
        skip_features_list = [
            [
                resized_skip_features_list[j][i]
                for j in range(len(resized_skip_features_list))
            ]
            for i in range(self.output_channels)
        ]

        # Decode sample features sequentially
        decoded_features_list = []
        for fused_features, skip_features in zip(
            fused_features_list, skip_features_list
        ):
            for i, decoder_block in enumerate(self.decoder):
                skip_feat = skip_features[-(i + 1)]
                if fused_features.shape[-2:] != skip_feat.shape[-2:]:
                    fused_features = nn.functional.interpolate(
                        fused_features,
                        size=skip_feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                fused_features = torch.cat([fused_features, skip_feat], dim=1)
                fused_features = decoder_block(fused_features)
            decoded_features_list.append(fused_features)
        decoded_features_list = torch.stack(decoded_features_list, dim=0)

        return decoded_features_list

    def _get_outputs(
        self, decoded_features_list: torch.Tensor, output_shape: Tuple[int, int]
    ) -> torch.Tensor:
        # Upsample to original size and pass through final layer
        outputs = []
        for decoded_features in decoded_features_list:
            decoded_features = nn.functional.interpolate(
                decoded_features,
                size=output_shape,
                mode="bilinear",
                align_corners=False,
            )
            output = self.final_layer(decoded_features)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0).squeeze(2).transpose(0, 1).contiguous()
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)

        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in [4, 5]:
            raise ValueError(
                f"❌ Input tensor must be of shape (batch_size, channels, height, width) or (batch_size, sequence_length, channels, height, width), got {x.shape}."
            )
        if not self.with_graph_processing and x.dim() == 5:
            raise ValueError(
                "❌ Model was not initialized with graph processing, but input is of shape (batch_size, sequence_length, channels, height, width)."
            )
        if x.dim() == 5 and x.shape[1] != self.output_channels:
            raise ValueError(
                f"❌ Input tensor must have {self.output_channels} channels, got {x.shape[1]}."
            )
        is_image = x.dim() == 4

        print(">>>", x.shape)

        # Get image features
        image_features_list = self._get_image_features_list(x, is_image)

        # Get depth features if needed
        depth_features = None
        if self.with_depth_information:
            depth_features = self._get_depth_features(x, is_image)

        # Project features and get skip features
        projected_image_features_list, skip_features_list = self._get_projected_image_features_list(
            image_features_list=image_features_list,
            depth_features=depth_features,
            is_image=is_image,
        )
        del image_features_list
        torch.cuda.empty_cache()

        # Fuse features
        fused_features = self._get_fused_features(
            projected_image_features_list=projected_image_features_list, 
            depth_features=depth_features,
            is_image=is_image,
            )
        del projected_image_features_list
        torch.cuda.empty_cache()

        # Add frame embeddings if needed
        if self.with_absolute_positional_embeddings:
            fused_features = self._add_absolute_positional_embeddings(fused_features)

        # Process features if needed
        if self.with_graph_processing:
            fused_features = self._get_graph_features(fused_features)

        # Decode features
        decoded_features_list = self._decode_features(
            fused_features=fused_features, 
            skip_features_list=skip_features_list, 
            is_image=is_image
        )
        del fused_features
        torch.cuda.empty_cache()

        # Get output
        outputs = self._get_outputs(decoded_features_list, x.shape[-2:])
        del decoded_features_list
        torch.cuda.empty_cache()

        return outputs
