import torch
from torch import nn

class DepthEncoder(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//2, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)