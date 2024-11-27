import torch
from torch import nn
from typing import List, Tuple

from src.models.conv_gru import ConvGRU

class GraphProcessor(nn.Module):
    """
    A graph processor module.
    """
    def __init__(
        self,
        hidden_channels: int,
        with_relative_positional_embeddings: bool,
        fusion_size: int,
        n_heads: int,
        neighbor_radius: int,
        n_iterations: int,
        dropout_rate: float,
    ) -> None:
        """
        Initialize the graph processor module.
        
        Args:
            hidden_channels (int): The number of hidden channels.
            with_relative_positional_embeddings (bool): Whether to use relative positional embeddings.
            fusion_size (int): The size of the fusion.
            n_heads (int): The number of heads.
            neighbor_radius (int): The neighbor radius.
            n_iterations (int): The number of iterations.
            dropout_rate (float): The dropout rate.
        """
        super(GraphProcessor, self).__init__()

        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"❌ Hidden channels must be divisible by the number of heads, got {hidden_channels} and {n_heads}."
            )

        self.hidden_channels = hidden_channels
        self.with_relative_positional_embeddings = with_relative_positional_embeddings
        self.fusion_size = fusion_size
        self.n_heads = n_heads
        self.neighbor_radius = neighbor_radius
        self.n_iterations = n_iterations
        self.dropout_rate = dropout_rate
        self.head_dim = hidden_channels // n_heads
        self.scale = hidden_channels**-0.5

        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        self.temporal_dropout = nn.Dropout3d(dropout_rate)

        # Get intra-attention components
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

        # Get inter-attention components
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
            nn.Dropout(dropout_rate),
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
            nn.Dropout(dropout_rate),
        )

        # Get the final components to combine intra- and inter-attention
        self.intra_inter_alpha = nn.Parameter(torch.tensor(0.5))
        self.gru = ConvGRU(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm = nn.LayerNorm(
            [hidden_channels, fusion_size, fusion_size]
        )

    def _compute_intra_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the intra-attention.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
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
        """
        Get the temporal encoding.
        
        Args:
            relative_position (int): The relative position.
            is_future (bool): Whether the position is in the future.
            device (torch.device): The device.
            
        Returns:
            torch.Tensor: The temporal encoding.
        """
        direction = torch.tensor(1.0 if is_future else -1.0, device=device)
        distance = torch.tensor(float(abs(relative_position)), device=device)
        temporal_information = torch.stack([direction, distance])

        return temporal_information

    def _compute_inter_attention(
        self, i: int, x: torch.Tensor, neighbors: List[Tuple[int, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute the inter-attention.
        
        Args:
            i (int): The index.
            x (torch.Tensor): The input tensor.
            neighbors (List[Tuple[int, torch.Tensor]]): The neighbors.
        """
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
        """
        Forward pass of the graph processor module.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        sequence_length, batch_size, channels, height, width = x.shape
        h = x

        if self.training:
            h = h.transpose(1, 2).contiguous()
            h = self.temporal_dropout(h)
            h = h.transpose(1, 2).contiguous()

        for _ in range(self.n_iterations):
            new_h = []
            for i in range(sequence_length):
                hi = h[i]
                if self.training:
                    hi = self.spatial_dropout(hi)

                intra_output = self._compute_intra_attention(hi)

                neighbors = [
                    (j, h[j])
                    for j in range(sequence_length)
                    if i != j and abs(i - j) <= self.neighbor_radius
                ]
                inter_output = self._compute_inter_attention(i, hi, neighbors)
                combined_message = (
                    self.intra_inter_alpha * intra_output
                    + (1 - self.intra_inter_alpha) * inter_output
                )

                next_h = self.gru(combined_message, hi)
                next_h = next_h + hi
                next_h = self.norm(next_h)
                new_h.append(next_h)

            h = torch.stack(new_h, dim=0)

        return h