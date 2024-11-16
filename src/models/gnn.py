import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import TGNMemory, TransformerConv


class SpatioTemporalGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.B = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU()

    def message_passing(self, x, edge_index):
        out = torch.zeros_like(x, device=x.device)
        in_degree = torch.zeros(x.shape[0], device=x.device)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            out[dst] += x[src]
            in_degree[dst] += 1
        in_degree = in_degree.clamp(min=1)
        out = out / in_degree.view(-1, 1, 1, 1)

        x = self.W(x) + self.B(out)
        x = self.relu(x)

        return x


class SpatioTemporalGNN(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpatioTemporalGNNLayer(
                    in_channels=in_channels, out_channels=out_channels
                )
                for in_channels, out_channels in zip(
                    channels_list[:-1], channels_list[1:]
                )
            ]
        )

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer.message_passing(x, edge_index)
        return x
