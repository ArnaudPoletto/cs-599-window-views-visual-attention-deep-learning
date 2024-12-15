import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from src.models.simple_net import SimpleNet
from src.models.image_decoder import ImageDecoder
from src.config import SEQUENCE_LENGTH, IMAGE_SIZE

class DisjointSimpleNet(nn.Module):
    def __init__(
        self, 
        freeze_encoder: bool,
        hidden_channels_list: List[int],
        dropout_rate: float,
        eps: float = 1e-6
    ) -> None:
        super(DisjointSimpleNet, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.hidden_channels_list = hidden_channels_list
        self.dropout_rate = dropout_rate
        self.eps = eps

        # Get normalization parameters for encoder inputs
        self.register_buffer(
            "image_mean",
            torch.tensor([0.4850, 0.4560, 0.4060]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.2290, 0.2240, 0.2250]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.image_encoders = nn.ModuleList([
            SimpleNet(freeze=freeze_encoder)
            for _ in range(SEQUENCE_LENGTH)
        ])
        self.image_decoders = nn.ModuleList([
            ImageDecoder(
                features_channels_list=self.image_encoders[i].feature_channels_list,
                hidden_channels_list=hidden_channels_list,
                output_channels=1,
                dropout_rate=dropout_rate,
                with_final_sigmoid=False,
            )
            for i in range(SEQUENCE_LENGTH)
        ])

        self.sigmoid = nn.Sigmoid()

    def _normalize_input(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        x = x.clone()
        normalized_x = (x - mean) / (std + self.eps)

        return normalized_x

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # Normalize input and resize the tensor accordingly
        is_image = x.dim() == 4
        if not is_image:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)
        x_image = self._normalize_input(x, self.image_mean, self.image_std)
        if is_image:
            x_image = x_image.unsqueeze(1).repeat(1, SEQUENCE_LENGTH, 1, 1, 1)
        else:
            x_image = x_image.view(batch_size, sequence_length, channels, height, width)

        # Encode and decode each image in the sequence
        decoded_features_list = []
        for i, (image_encoder, image_decoder) in enumerate(zip(self.image_encoders, self.image_decoders)):
            x_image_i = x_image[:, i]
            encoded_features_list = image_encoder(x_image_i)
            decoded_features = image_decoder(encoded_features_list)
            decoded_features_list.append(decoded_features)

        temporal_features = torch.stack(decoded_features_list, dim=1).squeeze(2)

        temporal_output = self.sigmoid(temporal_features)
        return temporal_output, None