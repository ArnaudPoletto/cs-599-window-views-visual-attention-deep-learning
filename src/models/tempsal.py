import torch
from torch import nn

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder
from src.config import SEQUENCE_LENGTH

HIDDEN_CHANNELS_LIST = [64, 96, 270, 256, 512]


class TempSAL(nn.Module):
    def __init__(
        self,
        temporal_output: bool,
        freeze_encoder: bool,
        hidden_channels_list: list[int] = HIDDEN_CHANNELS_LIST,
    ):
        super(TempSAL, self).__init__()

        self.output_channels = SEQUENCE_LENGTH if temporal_output else 1
        self.temporal_output = temporal_output
        self.hidden_channels_list = hidden_channels_list,
        self.freeze_encoder = freeze_encoder
        
        self.encoder = ImageEncoder(
            freeze=freeze_encoder,
        )

        self.decoder = ImageDecoder(
            features_channels=self.encoder.feature_channels,
            hidden_channels=hidden_channels_list,
            features_sizes=self.encoder.feature_sizes,
            output_channels=self.output_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x
