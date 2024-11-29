import torch
from torch import nn
from typing import Dict, Any
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from src.trainers.trainer import Trainer

from src.config import DEVICE

class ViDaSTrainer(Trainer):
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
        super(ViDaSTrainer, self).__init__(
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
            "input_channels": self.model.input_channels,
            "input_shape": self.model.input_shape,
            "hidden_channels_list": self.model.hidden_channels_list,
            "kernel_sizes": self.model.kernel_sizes,
            "use_max_poolings": self.model.use_max_poolings,
            "saliency_out_channels": self.model.saliency_out_channels,
            "attention_out_channels": self.model.attention_out_channels,
            "with_depth_information": self.model.with_depth_information,
        }

    def _get_name(
        self, optimizer: Optimizer, n_epochs: int, learning_rate: float
    ) -> str:
        name = self.name

        return name

    def _forward_pass(self, batch: tuple) -> torch.Tensor:
        frame, ground_truths, global_ground_truth = batch
        frame = frame.float().to(DEVICE)
        ground_truths = ground_truths.float().to(DEVICE)
        global_ground_truth = global_ground_truth.float().to(DEVICE)

        # Forward pass
        with autocast(enabled=self.use_scaler):
            global_output = self.model(frame)

        # Get loss
        global_loss = self.criterion(global_output, global_ground_truth)

        return None, global_loss, None, global_output, None, global_ground_truth