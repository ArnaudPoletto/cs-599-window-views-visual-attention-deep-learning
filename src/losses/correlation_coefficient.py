import torch
import torch.nn as nn


class CorrelationCoefficientLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super(CorrelationCoefficientLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"❌ Predictions and targets must have the same shape, got {pred.shape} and {target.shape}"
            )

        # Reshape tensors
        if pred.dim() == 4:
            batch_size, channels, height, width = pred.shape
            pred = pred.reshape(batch_size * channels, height * width)
            target = target.reshape(batch_size * channels, height * width)
        elif pred.dim() == 3:
            batch_size, height, width = pred.shape
            pred = pred.reshape(batch_size, height * width)
            target = target.reshape(batch_size, height * width)

        # Compute means and center the variables
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Compute variances
        pred_variance = torch.sum(pred_centered ** 2, dim=1)
        pred_variance = torch.clamp(pred_variance, min=self.eps)
        target_variance = torch.sum(target_centered ** 2, dim=1)
        target_variance = torch.clamp(target_variance, min=self.eps)

        # Compute correlation
        covariance = torch.sum(pred_centered * target_centered, dim=1)
        correlation = covariance / torch.sqrt(pred_variance * target_variance)
        correlation = torch.clamp(correlation, min=-1.0, max=1.0)

        loss = 1 - torch.mean(correlation)

        return loss
