import torch
from torch import nn
from typing import Dict, List, Tuple


class CombinedLoss(nn.Module):
    def __init__(
        self,
        losses: Dict[str, Tuple[nn.Module, float]] | List[Tuple[str, nn.Module, float]],
    ):
        super(CombinedLoss, self).__init__()

        if isinstance(losses, list):
            losses = {name: (loss, weight) for name, loss, weight in losses}

        self.losses = nn.ModuleDict({name: loss for name, (loss, _) in losses.items()})
        self.weights = {name: weight for name, (_, weight) in losses.items()}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        losses_dict = {}

        for name, loss_fn in self.losses.items():
            current_loss = loss_fn(pred, target)
            weighted_loss = current_loss * self.weights[name]
            total_loss += weighted_loss
            losses_dict[name] = current_loss.item()

        self.last_losses = losses_dict

        return total_loss

    def get_last_losses(self) -> Dict[str, float]:
        return self.last_losses
