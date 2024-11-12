# This file contains the general implementation of a trainer for the UNet model.

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer

from src.trainers.trainer import Trainer

from src.config import DEVICE


class UNetTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        use_scaler: bool,
        name: str,
    ) -> None:
        super().__init__(
            model=model,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            use_scaler=use_scaler,
            name=name,
        )

    def _get_name(
        self, optimizer: Optimizer, num_epochs: int, learning_rate: float
    ) -> str:
        name = self.name

        return name

    def _forward_pass(self, batch: tuple) -> torch.Tensor:
        frames, ground_truths, global_ground_truth = batch
        frames = frames.float().to(DEVICE)
        ground_truths = ground_truths.float().to(DEVICE)
        global_ground_truth = global_ground_truth.float().to(DEVICE)

        # Forward pass
        with autocast(enabled=self.use_scaler):
            outputs = self.model(frames).squeeze(1)

        loss = self.criterion(outputs, ground_truths)

        # Compute loss
        return loss, None, None # TODO: return None for now
