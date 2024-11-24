import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"❌ Predictions and targets must have the same shape, got {pred.shape} and {target.shape}"
            )

        # Reshape if 4D tensors (batch_size, channels/sequence_length, height, width)
        if pred.dim() == 4:
            batch_size, channels, height, width = pred.shape
            pred = pred.view(batch_size * channels, height * width)
            target = target.view(batch_size * channels, height * width)
        # Reshape if 3D tensors (batch_size, height, width)
        elif pred.dim() == 3:
            batch_size, height, width = pred.shape
            pred = pred.view(batch_size, height * width)
            target = target.view(batch_size, height * width)

        # Compute MSE
        loss = torch.mean((pred - target) ** 2)

        return loss