import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, eps: float = 1e-7):
        super(KLDivLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"❌ Predictions and targets must have the same shape, got {pred.shape} and {target.shape}"
            )

        # Reshape predictions and targets if they are 4D tensors
        if pred.dim() == 4:
            batch_size, sequence_length, height, width = pred.shape
            pred = pred.view(batch_size * sequence_length, height, width)
            target = target.view(batch_size * sequence_length, height, width)

        # Prepare predictions and targets
        target = torch.softmax(target, dim=1)
        log_pred = F.log_softmax(pred / self.temperature, dim=1)

        # Calculate KL divergence
        loss = nn.KLDivLoss(reduction="batchmean", log_target=False)(log_pred, target)

        return loss * (self.temperature**2)
