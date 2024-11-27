import torch
from torch import nn


class ConvGRU(nn.Module):
    """
    A convolutional GRU module.
    """

    def __init__(
        self, input_channels: int, hidden_channels: int, kernel_size: int, padding: int
    ) -> None:
        """
        Initialize the convolutional GRU module.

        Args:
            input_channels (int): The number of input channels.
            hidden_channels (int): The number of hidden channels.
            kernel_size (int): The size of the convolutional kernel.
            padding (int): The size of the padding.
        """
        super(ConvGRU, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Get the reset gate
        self.conv_zr = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Get the update gate
        self.conv_h = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional GRU module.

        Args:
            x (torch.Tensor): The input tensor.
            h (torch.Tensor): The hidden state tensor.

        Returns:
            torch.Tensor: The new hidden state tensor.
        """
        combined = torch.cat([x, h], dim=1)

        # Compute reset and update gates
        zr = self.sigmoid(self.conv_zr(combined))
        z, r = torch.split(zr, self.hidden_channels, dim=1)

        # Compute candidate hidden state
        combined_r = torch.cat([x, r * h], dim=1)
        h_hat = torch.tanh(self.conv_h(combined_r))

        # Update hidden state
        h_new = (1 - z) * h + z * h_hat

        return h_new
