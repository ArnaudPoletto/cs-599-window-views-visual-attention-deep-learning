import torch
from torch import nn

from src.config import IMAGE_SIZE


class DepthDecoder(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # After concatenation with skip connection, input channels double
        self.conv1 = nn.Sequential(
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

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # After concatenation with skip connection, input channels double
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                hidden_channels // 2 + hidden_channels // 4, # TODO: remove hardcoded
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Final convolution to get to single channel
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, skip_features: list[torch.Tensor]
    ) -> torch.Tensor:
        # Unpack skip features
        skip2, skip1 = skip_features

        # First upsampling + skip connection
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv1(x)

        # Second upsampling + skip connection
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)

        # Final upsampling
        x = self.up3(x)
        x = self.final_conv(x)

        return x
