from torch import nn

from src.models.image_encoder import ImageEncoder
from src.models.image_decoder import ImageDecoder

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

    def forward(self, x):
        image_features = self.encoder(x)
        output = self.decoder(image_features)

        return output