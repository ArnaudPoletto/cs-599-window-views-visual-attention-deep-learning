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
        fusion_size: int,
        neighbor_radius: int,
        n_iterations: int,
        dropout_rate: float,
        with_edge_features: bool,
        with_positional_embeddings: bool,
        with_directional_kernels: bool,
    ) -> None:
        """
        Initialize the graph processor module.
        
        Args:
            hidden_channels (int): The number of hidden channels.
            fusion_size (int): The size of the fusion.
            neighbor_radius (int): The neighbor radius.
            n_iterations (int): The number of iterations.
            dropout_rate (float): The dropout rate.
            with_edge_features (bool): Whether to use edge features.
            with_positional_embeddings (bool): Whether to use positional embeddings.
            with_directional_kernels (bool): Whether to use directional kernels.
        """
        super(GraphProcessor, self).__init__()

        self.hidden_channels = hidden_channels
        self.fusion_size = fusion_size
        self.neighbor_radius = neighbor_radius
        self.n_iterations = n_iterations
        self.dropout_rate = dropout_rate
        self.with_edge_features = with_edge_features
        self.with_positional_embeddings = with_positional_embeddings
        self.with_directional_kernels = with_directional_kernels
        self.scale = hidden_channels**-0.5

        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        self.temporal_dropout = nn.Dropout3d(dropout_rate)


        # Get intra-attention components
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
            nn.ReLU(inplace=True),
        )
        self.intra_alpha = nn.Parameter(torch.tensor(0.5))

        if with_positional_embeddings:
            self.positional_embeddings = nn.Parameter(
                torch.randn(2, 1, fusion_size, fusion_size)
            )

        # Get inter-attention components
        inter_message_edge_in_channels = hidden_channels
        if with_positional_embeddings:
            inter_message_edge_in_channels += 1
        if with_edge_features:
            inter_message_edge_in_channels += 2 # TODO: adapt given edge features dimension
        self.inter_message_edge_conv = nn.Conv2d(
            in_channels=inter_message_edge_in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        if with_directional_kernels:
            self.future_inter_query_conv = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.future_inter_key_conv = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.future_inter_value_conv = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.past_inter_query_conv = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.past_inter_key_conv = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.past_inter_value_conv = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        else:
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

    def _compute_intra_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the intra-attention.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        batch_size, channels, height, width = x.shape

        query = self.intra_query_conv(x).view(batch_size, channels, -1)
        key = self.intra_key_conv(x).view(batch_size, channels, -1)
        value = self.intra_value_conv(x).view(batch_size, channels, -1)

        attention = torch.bmm(query.transpose(1, 2), key) * self.scale
        attention = attention - attention.max(dim=-1, keepdim=True)[0]
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
        temporal_encoding = torch.stack([direction, distance])

        return temporal_encoding

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

        messages = []
        gates = []
        for j, y in neighbors:
            is_future = i < j

            # Concatenate temporal encoding
            if self.with_edge_features:
                relative_position = (i - j) / self.neighbor_radius
                temporal_encoding = self._get_temporal_encoding(
                relative_position=relative_position, 
                is_future=is_future, 
                device=x.device,
                )
                temporal_encoding = temporal_encoding.view(1, -1, 1, 1).expand(
                batch_size, -1, height, width
                )
                y = torch.cat([y, temporal_encoding], dim=1)

            # Concatenate positional embeddings
            if self.with_positional_embeddings:
                positional_embeddings = self.positional_embeddings[int(is_future)].unsqueeze(0).expand(batch_size, -1, -1, -1)
                y = torch.cat([y, positional_embeddings], dim=1)

            y = self.inter_message_edge_conv(y)

            # Optionally use directional kernels
            if self.with_directional_kernels:
                inter_query_conv = self.future_inter_query_conv if is_future else self.past_inter_query_conv
                inter_key_conv = self.future_inter_key_conv if is_future else self.past_inter_key_conv
                inter_value_conv = self.future_inter_value_conv if is_future else self.past_inter_value_conv
            else:
                inter_query_conv = self.inter_query_conv
                inter_key_conv = self.inter_key_conv
                inter_value_conv = self.inter_value_conv

            # Compute attention
            query = inter_query_conv(x).view(batch_size, channels, -1)
            key = inter_key_conv(y).view(batch_size, channels, -1)
            value = inter_value_conv(y).view(batch_size, channels, -1)

            attention = torch.bmm(query.transpose(1, 2), key) * self.scale
            attention = attention - attention.max(dim=-1, keepdim=True)[0]
            attention = torch.softmax(attention, dim=-1)

            message = torch.bmm(attention, value.transpose(1, 2)).transpose(1, 2)
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
                new_h.append(next_h)

            h = torch.stack(new_h, dim=0)

        return h