import torch
from torch import nn

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder


class TempSAL(nn.Module):
    def __init__(
        self,
        output_channels: int,
        freeze_encoder: bool,
    ):
        super(TempSAL, self).__init__()

        self.output_channels = output_channels
        self.freeze_encoder = freeze_encoder
        
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder(output_channels=output_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x
