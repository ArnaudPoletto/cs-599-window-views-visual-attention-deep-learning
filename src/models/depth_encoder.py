import torch
from torch import nn

from src.config import IMAGE_SIZE

class DepthEncoder(nn.Module):
    def __init__(self, hidden_channels: int):
        if hidden_channels % 4 != 0:
            raise ValueError("❌ Hidden channels must be divisible by 4.")
        
        super(DepthEncoder, self).__init__()

        # Split into individual layers for skip connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden_channels//4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels//4, hidden_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels//2, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.features_channels_list = [hidden_channels//4, hidden_channels//2, hidden_channels, hidden_channels]
        self.features_sizes = self._get_features_sizes()

    def _get_features_sizes(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
            x = self.forward(x)
            feature_sizes = [image_feature.size()[-1] for image_feature in x]
        
        return feature_sizes
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Store intermediate features for skip connections
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        return [x1, x2, x3, x4]