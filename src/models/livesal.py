import math
import torch
from torch import nn
from typing import List, Tuple

from src.models.image_encoder import ImageEncoder
from src.config import DEVICE

GRAPH_PROCESSOR_N_ITERATIONS = 3


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
        self, hidden_channels: int, n_iterations: int = GRAPH_PROCESSOR_N_ITERATIONS
    ):
        super(GraphProcessor, self).__init__()

        self.hidden_channels = hidden_channels
        self.n_iterations = n_iterations
        self.scale = hidden_channels**-0.5
        self.embedding_channels = 4  # TODO: remove hardcoded

        self.embedding_proj = nn.Sequential(
            nn.Conv2d(
                hidden_channels + self.embedding_channels, hidden_channels, 1, bias=True
            ),  # TODO: remove hardcoded
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_channels),
        )

        self.Wf = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.Wh = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.Wl = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.alpha = nn.Parameter(torch.ones(1))

        self.Wc = nn.Parameter(torch.randn(hidden_channels, hidden_channels))

        self.Wg = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.message_bn = nn.BatchNorm2d(hidden_channels)

        self.gru = ConvGRU(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            padding=2,
        )

        self.intra_weight = nn.Parameter(torch.tensor(0.5))
        self.inter_weight = nn.Parameter(torch.tensor(0.5))

        self.sigmoid = nn.Sigmoid()

    def _compute_intra_attention(self, x: torch.Tensor) -> torch.Tensor:
        ft = self.Wf(x)
        ht = self.Wh(x)
        lt = self.Wl(x)

        batch_size, channels, height, width = x.shape
        ft = ft.view(batch_size, channels, -1)
        ht = ht.view(batch_size, channels, -1)
        lt = lt.view(batch_size, channels, -1)

        attention = torch.bmm(ft.transpose(1, 2), ht)
        attention = torch.softmax(attention, dim=-1)
        context = torch.bmm(attention, lt.transpose(1, 2))
        context = context.transpose(1, 2).view(batch_size, channels, height, width)

        eii = self.alpha * context + x

        return eii

    def _compute_inter_attention(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1)
        y_flat = y.view(batch_size, channels, -1)

        yt = torch.matmul(self.Wc, y_flat)
        eij = torch.bmm(x_flat.transpose(1, 2), yt)

        return eij

    def _compute_message(self, x: torch.Tensor, eij: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1)

        eij_norm = torch.softmax(eij, dim=-1)
        mji = torch.bmm(eij_norm, x_flat.transpose(1, 2))
        mji = mji.transpose(1, 2).view(batch_size, channels, height, width)

        gji = self.gap(self.Wg(mji))
        gji = self.sigmoid(gji)

        mji_gated = gji * mji
        mji_gated = self.message_bn(mji_gated)

        return mji_gated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial node states
        sequence_length, batch_size, channels, height, width = x.shape
        h = x

        for _ in range(self.n_iterations):
            messages = []
            for i in range(sequence_length):
                intra_message = self._compute_intra_attention(h[i])
                node_messages = []
                for j in range(sequence_length):
                    if i == j:
                        continue

                    eij = self._compute_inter_attention(h[i], h[j])
                    message = self._compute_message(h[j], eij)
                    node_messages.append(message)

                node_messages = torch.stack(node_messages, dim=0)
                inter_message = torch.sum(node_messages, dim=0)
                aggregated_message = self.intra_weight * intra_message + self.inter_weight * inter_message
                messages.append(aggregated_message)

            new_h = []
            for i in range(sequence_length):
                next_h = self.gru(messages[i], h[i]) + h[i]
                new_h.append(next_h)
            h = torch.stack(new_h, dim=0)

        return h


class LiveSAL(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        with_positional_embeddings: bool,
        with_graph_processing: bool,
        freeze_encoder: bool,
    ):
        super(LiveSAL, self).__init__()

        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.with_positional_embeddings = with_positional_embeddings
        self.with_graph_processing = with_graph_processing
        self.freeze_encoder = freeze_encoder

        # Get encoder and freeze weights if needed
        self.encoder = ImageEncoder()
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection layers to project encoded features to a common space
        # Add 2 spatial awareness layers with depthwise separable 3x3 conv for efficiency
        encoder_channels = [96, 270, 1080, 2160, 4320]
        self.projection_layers = nn.ModuleList(
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
                for in_ch in encoder_channels
            ]
        )

        # Fusion layer to combine all features
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels * 5,
                out_channels=hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        if with_positional_embeddings:
            # TODO: change hardcoded positional embeddings size
            self.positional_embeddings = nn.Parameter(
                torch.randn(output_channels, hidden_channels, 21, 21)
            )

        if with_graph_processing:
            self.graph_processor = GraphProcessor(hidden_channels)

        # Decoder layers
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels * 2,
                        hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4) # TODO: remove hardcoded
            ]
        )

        # Final layer
        final_out_channels = 1 if with_graph_processing else output_channels
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
                out_channels=final_out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def _get_features_list(self, x: torch.Tensor, is_image: bool) -> List[torch.Tensor]:
        # Flatten batch_size and sequence_length for video inputs to pass through the encoder
        if self.with_graph_processing and not is_image:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)

        # Get multi-scale features
        features_list = self.encoder(x)

        return features_list

    def _get_projected_features_list(
        self,
        features_list: List[torch.Tensor],
        is_image: bool,
    ) -> List[torch.Tensor]:
        projected_features_list = []
        for features, projection_layer in zip(features_list, self.projection_layers):
            projected_features = projection_layer(features)
            projected_features_list.append(projected_features)

        # Get skip features, and repeat image skip features for graph processing
        skip_features_list = projected_features_list[:4] # TODO: remove hardcoded
        if self.with_graph_processing and is_image:
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

        return projected_features_list, skip_features_list

    def _get_fused_features(
        self, projected_features_list: List[torch.Tensor], is_image: bool
    ) -> torch.Tensor:
        # Resize features to middle scale
        base_size = projected_features_list[3].shape[-2:] # TODO: remove hardcoded
        resized_features_list = []
        for features in projected_features_list:
            if features.shape[-2:] != base_size:
                features = nn.functional.interpolate(
                    features,
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_features_list.append(features)

        # Fuse features
        fused_features = self.fusion(torch.cat(resized_features_list, dim=1))

        # Repeat image features for graph processing
        if self.with_graph_processing and is_image:
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

    def _add_positional_embeddings(self, fused_features: torch.Tensor) -> torch.Tensor:
        batch_size_sequence_length, channels, height, width = fused_features.shape
        fused_features_list = fused_features.view(
            -1, self.output_channels, channels, height, width
        )
        fused_features = (fused_features_list + self.positional_embeddings).view(
            batch_size_sequence_length, channels, height, width
        )

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
        if self.with_graph_processing:
            # Organize features as independent samples
            batch_size_sequence_length, channels, height, width = fused_features.shape
            fused_features_list = (
                fused_features.view(-1, self.output_channels, channels, height, width)
                .transpose(0, 1)
                .contiguous()
            )
            resized_skip_features_list = []
            for skip_features in skip_features_list:
                batch_size_sequence_length, channels, height, width = (
                    skip_features.shape
                )
                skip_features = (
                    skip_features.view(
                        -1, self.output_channels, channels, height, width
                    )
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
        else:
            fused_features_list = [fused_features]
            skip_features_list = [skip_features_list]

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

        # Get multi-scale features
        features_list = self._get_features_list(x, is_image)

        # Project features and get skip features
        projected_features_list, skip_features_list = self._get_projected_features_list(
            features_list, is_image
        )
        del features_list
        torch.cuda.empty_cache()

        # Fuse features
        fused_features = self._get_fused_features(projected_features_list, is_image)
        del projected_features_list
        torch.cuda.empty_cache()

        # Add frame embeddings if needed
        if self.with_positional_embeddings:
            fused_features = self._add_positional_embeddings(fused_features)

        # Process features if needed
        if self.with_graph_processing:
           fused_features = self._get_graph_features(fused_features)

        # Decode features
        decoded_features_list = self._decode_features(
            fused_features, skip_features_list, is_image
        )
        del fused_features
        torch.cuda.empty_cache()

        # Get output
        outputs = self._get_outputs(decoded_features_list, x.shape[-2:])
        del decoded_features_list
        torch.cuda.empty_cache()

        return outputs
