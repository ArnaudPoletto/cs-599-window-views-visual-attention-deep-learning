import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super(KLDivLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"❌ Predictions and targets must have the same shape, got {pred.shape} and {target.shape}"
            )

        # Reshape tensors
        if pred.dim() == 4:
            batch_size, sequence_length, height, width = pred.shape
            pred = pred.view(batch_size * sequence_length, height, width)
            target = target.view(batch_size * sequence_length, height, width)
            effective_batch_size = batch_size * sequence_length
        else:
            batch_size = pred.size(0)
            height = pred.size(1)
            width = pred.size(2)
            effective_batch_size = batch_size

        # Calculate sum and expand
        sum_pred = torch.sum(pred.view(effective_batch_size, -1), 1)
        expand_pred = sum_pred.view(effective_batch_size, 1, 1).expand(effective_batch_size, height, width)

        sum_target = torch.sum(target.view(effective_batch_size, -1), 1)
        expand_target = sum_target.view(effective_batch_size, 1, 1).expand(effective_batch_size, height, width)

        # Normalize predictions and targets
        pred = pred / (expand_pred + self.eps)
        target = target / (expand_target + self.eps)

        # Reshape for loss calculation
        pred = pred.view(effective_batch_size, -1)
        target = target.view(effective_batch_size, -1)

        result = target * torch.log(self.eps + target/(pred + self.eps))

        return torch.mean(torch.sum(result, 1))
