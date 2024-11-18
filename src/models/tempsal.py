import torch
from torch import nn

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder


class TempSAL(nn.Module):
    def __init__(
        self,
        output_channels: int,
        freeze_encoder: bool,
        hidden_channels_list: list[int] = [64, 96, 270, 256, 512], # TODO: remove hardcoding
    ):
        super(TempSAL, self).__init__()

        self.output_channels = output_channels
        self.hidden_channels_list = hidden_channels_list,
        self.freeze_encoder = freeze_encoder
        
        self.encoder = ImageEncoder(
            freeze=freeze_encoder,
        )
        self.decoder = ImageDecoder(
            features_channels=self.encoder.feature_channels,
            hidden_channels=hidden_channels_list,
            features_sizes=self.encoder.feature_sizes,
            output_channels=output_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x
