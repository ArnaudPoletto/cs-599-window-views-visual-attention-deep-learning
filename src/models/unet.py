import torch
from torch import nn

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder
from src.models.gnn import SpatioTemporalGNN
from src.config import DEVICE


class UNet(nn.Module):
    def __init__(
        self,
        freeze_encoder: bool,
    ) -> None:
        super(UNet, self).__init__()

        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.gnns = nn.ModuleList()
        encoder_channels = [96, 270, 1080, 2160, 4320]
        for channels in encoder_channels:
            channels_list = [channels, channels // 2, channels // 4, channels // 8]
            self.gnns.append(SpatioTemporalGNN(channels_list=channels_list))

    def _create_temporal_edges(self, batch_size, sequence_length):
        edges = []
        for b in range(batch_size):
            for t in range(sequence_length - 1):
                curr_idx = b * sequence_length + t
                next_idx = b * sequence_length + (t + 1)
                edges.extend([[curr_idx, next_idx], [next_idx, curr_idx]])

        return torch.tensor(edges, dtype=torch.long, device=DEVICE).t()

    def forward(self, x):
        # Flatten the first dimensions to pass every image through the encoder
        batch_size, sequence_length, sample_size, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        image_features_list = self.encoder(x)

        # Pool the features from the encoder
        processed_image_features_list = []
        for image_features, gnn in zip(image_features_list, self.gnns):
            # Reshape and pool sample features
            feature_channels, feature_height, feature_width = image_features.shape[-3:]
            image_features = image_features.view(
                batch_size * sequence_length,
                sample_size,
                feature_channels,
                feature_height,
                feature_width,
            )
            image_features = torch.mean(image_features, dim=1)

            edge_index = self._create_temporal_edges(batch_size, sequence_length)
            processed_image_features = gnn(x=image_features, edge_index=edge_index)
            processed_image_features_list.append(processed_image_features)

        # Pass the pooled features through the decoder
        output = self.decoder(processed_image_features_list)
        output = output.view(batch_size, sequence_length, height, width)

        return output
