import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
from torch import nn


class ImageDecoder(nn.Module):
    def __init__(
        self,
    ):
        super(ImageDecoder, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=4320,
            out_channels=2160,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.conv4 = nn.Conv2d(
            in_channels=2160 + 2160, out_channels=2160, kernel_size=3, padding=1
        )

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=2160,
            out_channels=1080,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=1080 + 1080, out_channels=1080, kernel_size=3, padding=1
        )

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=1080,
            out_channels=270,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=270 + 270, out_channels=270, kernel_size=3, padding=1
        )

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=270,
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.conv1 = nn.Conv2d(
            in_channels=96 + 96, out_channels=32, kernel_size=3, padding=1
        )

        self.final_upconv = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=0,
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        x = self.relu(self.upconv4(features[-1]))
        x = torch.cat((x, features[-2]), dim=1)
        x = self.relu(self.conv4(x))

        x = self.relu(self.upconv3(x))
        x = torch.cat((x, features[-3]), dim=1)
        x = self.relu(self.conv3(x))

        x = self.relu(self.upconv2(x))
        x = torch.cat((x, features[-4]), dim=1)
        x = self.relu(self.conv2(x))

        x = self.relu(self.upconv1(x))
        x = torch.cat((x, features[-5]), dim=1)
        x = self.relu(self.conv1(x))

        x = self.final_upconv(x)

        x = self.sigmoid(x)

        return x
