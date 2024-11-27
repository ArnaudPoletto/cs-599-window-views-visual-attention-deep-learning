import torch
from torch import nn
from torch.optim import Optimizer
from torch.cuda.amp import autocast
from typing import Dict, Any

from src.trainers.trainer import Trainer

from src.config import DEVICE

class TempSALTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        use_scaler: bool,
        name: str,
    ) -> None:
        super(TempSALTrainer, self).__init__(
            model=model,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            use_scaler=use_scaler,
            name=name,
            dataset="salicon",
        )

    def _get_wandb_config(self) -> Dict[str, Any]:
        return {
            "freeze_encoder": self.model.freeze_encoder,
            "hidden_channels_list": self.model.hidden_channels_list,
            
        }

    def _get_name(
        self, optimizer: Optimizer, n_epochs: int, learning_rate: float
    ) -> str:
        name = self.name

        return name

    def _forward_pass(self, batch: tuple) -> torch.Tensor:
        frame, temporal_ground_truth, global_ground_truth = batch
        frame = frame.float().to(DEVICE)
        temporal_ground_truth = temporal_ground_truth.float().to(DEVICE)
        global_ground_truth = global_ground_truth.float().to(DEVICE)

        # Forward pass
        with autocast(enabled=self.use_scaler):
            temporal_output, global_output = self.model(frame)

        # Get loss
        temporal_loss = self.criterion(temporal_output, temporal_ground_truth)
        global_loss = self.criterion(global_output, global_ground_truth)

        return temporal_loss, global_loss, temporal_output, global_output, temporal_ground_truth, global_ground_truth