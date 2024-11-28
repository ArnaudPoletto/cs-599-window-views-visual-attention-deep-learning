import torch
from torch import nn

from src.config import IMAGE_SIZE

class DepthEncoder(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        
        # Split into individual layers for skip connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden_channels//4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels//4, hidden_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels//2, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.features_size = self._get_features_size()

    def _get_features_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
            x = self.forward(x)[0]  # Get main output only
        
        return x.shape[2]
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Store intermediate features for skip connections
        x1 = self.conv1(x)      # First skip feature
        x2 = self.conv2(x1)     # Second skip feature
        x3 = self.conv3(x2)     # Final output
        
        return x3, [x2, x1]  # Return final features and skip features