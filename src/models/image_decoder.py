import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
from torch import nn


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(
                in_channels=4320, out_channels=512, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(21, 21), mode="bilinear", align_corners=False),
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=512 + 2160,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(42, 42), mode="bilinear", align_corners=False),
        )

        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1080 + 256,
                out_channels=270,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(83, 83), mode="bilinear", align_corners=False),
        )

        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=540, 
                out_channels=96, 
                kernel_size=3, 
                padding=1, 
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(165, 165), mode="bilinear", align_corners=False),
        )

        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, 
                out_channels=128, 
                kernel_size=3, 
                padding=1, 
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(331, 331), mode="bilinear", align_corners=False),
        )

        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=128, 
                kernel_size=3, 
                padding=1, 
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, 
                out_channels=1, 
                kernel_size=3, 
                padding=1, 
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, features):
        out5 = features[-1]
        out4 = features[-2]
        out3 = features[-3]
        out2 = features[-4]
        out1 = features[-5]

        x = self.deconv_layer0(out5)

        x = torch.cat((x, out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x, out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x, out2), 1)
        x = self.deconv_layer3(x)
        x = torch.cat((x, out1), 1)
        x = self.deconv_layer4(x)

        x = self.deconv_layer5(x)

        return x.squeeze(1)
