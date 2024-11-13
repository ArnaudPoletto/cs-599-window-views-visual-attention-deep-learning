import torch
import torch.nn as nn


class CorrelationCoefficientLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super(CorrelationCoefficientLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"❌ Predictions and targets must have the same shape, got {pred.shape} and {target.shape}"
            )

        # Reshape if 4D tensors
        if pred.dim() == 4:
            batch_size, channels, height, width = pred.shape
            pred = pred.view(batch_size * channels, height, width)
            target = target.view(batch_size * channels, height, width)

        # Flatten predictions and targets
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        # Center the data
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Calculate correlation coefficient
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt(
            (pred_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1) + self.eps
        )
        correlation = numerator / denominator

        loss = 1 - correlation.mean()

        return loss
