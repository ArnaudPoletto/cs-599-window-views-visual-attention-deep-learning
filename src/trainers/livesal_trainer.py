import torch
from torch import nn
from typing import Dict, Any
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from src.trainers.trainer import Trainer

from src.config import DEVICE

class LiveSALTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        use_scaler: bool,
        name: str,
        dataset: str,
    ) -> None:
        super(LiveSALTrainer, self).__init__(
            model=model,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            use_scaler=use_scaler,
            name=name,
            dataset=dataset,
        )

    def _get_wandb_config(self) -> Dict[str, Any]:
        return {
            "hidden_channels": self.model.hidden_channels,
            "with_relative_positional_embeddings": self.model.with_relative_positional_embeddings,
            "neighbor_radius": self.model.neighbor_radius,
            "n_iterations": self.model.n_iterations,
            "with_graph_processing": self.model.with_graph_processing,
            "freeze_encoder": self.model.freeze_encoder,
            "with_depth_information": self.model.with_depth_information,
            "dropout_rate": self.model.dropout_rate,
            "fusion_level": self.model.fusion_level,

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

        temporal_loss = self.criterion(temporal_output, temporal_ground_truth)
        global_loss = self.criterion(global_output, global_ground_truth)

        return temporal_loss, global_loss, temporal_output, global_output, temporal_ground_truth, global_ground_truth