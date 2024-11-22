# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DSAM(nn.Module):
#     def __init__(self, in_channels):
#         super(DSAM, self).__init__()
#         self.temporal_pool = nn.AdaptiveAvgPool3d((None, None, 1))
#         self.saliency_conv = nn.Conv2d(in_channels, 64, 1)
#         self.attention_conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
#         self.attention_conv2 = nn.Conv2d(in_channels // 2, 1, 1)
#         self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        
#     def forward(self, x):
#         # Temporal average pooling
#         batch, channels, time, height, width = x.size()
#         x_pool = self.temporal_pool(x).squeeze(2)
        
#         # Generate saliency features
#         saliency_maps = self.saliency_conv(x_pool)
        
#         # Generate attention maps
#         attention = F.relu(self.attention_conv1(x_pool))
#         attention = self.attention_conv2(attention)
#         attention_map = torch.softmax(attention.view(batch, -1), dim=1).view(batch, 1, height, width)
        
#         # Enhance features
#         enhanced = (1 + attention_map) * x
        
#         return enhanced, saliency_maps, attention

# class Decoder(nn.Module):
#     def __init__(self, in_channels_list):
#         super(Decoder, self).__init__()
#         self.decode_blocks = nn.ModuleList()
        
#         for i in range(len(in_channels_list)-1):
#             block = nn.Sequential(
#                 nn.ConvTranspose2d(in_channels_list[i], in_channels_list[i+1], 4, 2, 1),
#                 nn.BatchNorm2d(in_channels_list[i+1])
#             )
#             self.decode_blocks.append(block)
            
#     def forward(self, features):
#         x = features[-1]
#         for i, block in enumerate(self.decode_blocks):
#             x = block(x)
#             if i < len(features)-1:
#                 x = torch.cat([x, features[-(i+2)]], dim=1)
#         return x

# class ViDaS(nn.Module):
#     def __init__(self):
#         super(ViDaS, self).__init__()
#         self.rgb_encoder = nn.ModuleList([
#             self._make_encoder_block(3, 64),
#             self._make_encoder_block(64, 128),
#             self._make_encoder_block(128, 256),
#             self._make_encoder_block(256, 512)
#         ])
        
#         self.depth_encoder = nn.ModuleList([
#             self._make_encoder_block(1, 64),
#             self._make_encoder_block(64, 128),
#             self._make_encoder_block(128, 256),
#             self._make_encoder_block(256, 512)
#         ])
        
#         self.rgb_dsam = nn.ModuleList([DSAM(c) for c in [64, 128, 256, 512]])
#         self.depth_dsam = nn.ModuleList([DSAM(c) for c in [64, 128, 256, 512]])
        
#         self.rgb_decoder = Decoder([512, 256, 128, 64])
#         self.depth_decoder = Decoder([512, 256, 128, 64])
        
#         self.fusion = nn.Sequential(
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, 1)
#         )
        
#     def _make_encoder_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(1,2,2), padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU()
#         )
        
#     def forward(self, rgb, depth):
#         rgb_features = []
#         depth_features = []
        
#         # RGB stream
#         x_rgb = rgb
#         for enc_block, dsam in zip(self.rgb_encoder, self.rgb_dsam):
#             x_rgb = enc_block(x_rgb)
#             x_rgb, sal_rgb, _ = dsam(x_rgb)
#             rgb_features.append(sal_rgb)
            
#         # Depth stream
#         x_depth = depth
#         for enc_block, dsam in zip(self.depth_encoder, self.depth_dsam):
#             x_depth = enc_block(x_depth)
#             x_depth, sal_depth, _ = dsam(x_depth)
#             depth_features.append(sal_depth)
            
#         # Decode both streams
#         rgb_out = self.rgb_decoder(rgb_features)
#         depth_out = self.depth_decoder(depth_features)
        
#         # Fusion
#         fused = torch.cat([rgb_out, depth_out], dim=1)
#         saliency = self.fusion(fused)
        
#         return saliency

import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


DSAM_SALIENCY_OUT_CHANNELS = 64
DSAM_ATTENTION_OUT_CHANNELS = 16

class DSAM(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            saliency_out_channels: int = DSAM_SALIENCY_OUT_CHANNELS,
            attention_out_channels: int = DSAM_ATTENTION_OUT_CHANNELS,
            ) -> None:
        super(DSAM, self).__init__()

        self.temporal_pool = nn.AdaptiveAvgPool3d((None, None, 1))
        self.saliency_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=saliency_out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.attention_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=attention_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=attention_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=attention_out_channels,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=True,
            )
        )
        # TODO: this upscale should scale with depth, i.e. do * 2 or * 4 etc...
        self.upsample = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Temporal average pooling
        x_pool = self.temporal_pool(x).squeeze(2)

        # Generate saliency features
        saliency_maps = self.saliency_conv(x_pool)

        # Generate attention maps
        attention_map = self.attention_layer(x_pool)
        attention_map = torch.softmax(attention_map, dim=(2, 3))
        attention_map = self.upsample(attention_map)


        # Enhance features
        enhanced = (1 + attention_map) * x

        return enhanced, saliency_maps, attention_map
    
class ViDaSEncoder(nn.Module):
    pass

class ViDaSDecoder(nn.Module):
    def __init__(self) -> None:
        super(ViDaSDecoder, self).__init__()
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class ViDaS(nn.Module):
    def __init__(self):
        super(ViDaS, self).__init__()

        # TODO: remove hardcoding
        self.encoder_blocks = nn.ModuleList([
            self._make_encoder_block(
                in_channels=3, 
                out_channels=64, 
                kernel_size=7,
                stride=(1, 2, 2),
                padding=3,
            ),
            nn.Sequential([
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
                self._make_encoder_block(
                    in_channels=64, 
                    out_channels=128, 
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ]),
            self._make_encoder_block(
                in_channels=128, 
                out_channels=256,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            self._make_encoder_block(
                in_channels=256, 
                out_channels=512,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
        ])

        self.dsams = nn.ModuleList([DSAM(c) for c in [64, 128, 256, 512]])

    def _make_encoder_block(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int,
            stride: Tuple[int, int, int],
            padding: int,
        ) -> nn.Sequential:
        encoder_block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )

        return encoder_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode and retain saliency maps
        saliency_maps_list = []
        for encoding_block, dsam in zip(self.encoder_blocks, self.dsams):
            x = encoding_block(x)
            x, saliency_maps = dsam(x)
            saliency_maps_list.append(saliency_maps)

        # Decode
