import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import time
import torch
import numpy as np
from torch import nn
from PIL import Image
from typing import Tuple
import lightning.pytorch as pl

from src.losses.kl_div import KLDivLoss
from src.metrics.metrics import Metrics
from src.losses.combined import CombinedLoss
from src.losses.correlation_coefficient import CorrelationCoefficientLoss
from src.config import (
    DHF1K_PATH,
    LOSS_WEIGHTS,
    SALICON_PATH, 
    VIEWOUT_PATH,
)


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        name: str,
        dataset: str,
    ) -> None:
        super(LightningModel, self).__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.name = name
        self.dataset = dataset

        # Get criterion
        kl_loss = KLDivLoss()
        corr_loss = CorrelationCoefficientLoss()
        self.criterion = CombinedLoss(
            {
                "kl": (kl_loss, LOSS_WEIGHTS["kl"]),
                "cc": (corr_loss, LOSS_WEIGHTS["cc"]),
            }
        )

        self.save_hyperparameters(ignore=['model', 'criterion'])

        self.best_eval_val_loss = float('inf')
        self.eval_train_loss = 0
        self.eval_val_loss = 0
    
    def forward(self, batch):
        pass

    def _process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, temporal_targets, global_targets, _ = batch
        
        # Forward pass through model
        temporal_output, global_output = self.model(inputs)
        
        # Calculate losses using criterion
        temporal_loss = None
        global_loss = None
        if temporal_output is not None and temporal_targets is not None:
            temporal_loss = self.criterion(temporal_output, temporal_targets)
        if global_output is not None and global_targets is not None:
            global_loss = self.criterion(global_output, global_targets)
            
        return (
            temporal_loss,
            global_loss,
            temporal_output,
            global_output,
            temporal_targets,
            global_targets,
        )
    
    def training_step(self, batch, batch_idx):
        temporal_train_loss, global_train_loss, _, _, _, _ = self._process_batch(batch)
        
        # Calculate total loss
        train_loss = 0
        if temporal_train_loss is not None:
            train_loss = train_loss + temporal_train_loss
        if global_train_loss is not None:
            train_loss = train_loss + global_train_loss
            
        # Log metrics
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, sync_dist=True)
        if temporal_train_loss is not None:
            self.log('temporal_train_loss', temporal_train_loss, on_step=True, on_epoch=True, sync_dist=True)
        if global_train_loss is not None:
            self.log('global_train_loss', global_train_loss, on_step=True, on_epoch=True, sync_dist=True)

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        temporal_val_loss, global_val_loss, temporal_output, global_output, temporal_ground_truth, global_ground_truth = self._process_batch(batch)
        
        # Calculate total validation loss
        val_loss = 0
        if temporal_val_loss is not None:
            val_loss = val_loss + temporal_val_loss
            self.log('val_temporal_loss', temporal_val_loss, on_epoch=True, sync_dist=True)
        if global_val_loss is not None:
            val_loss = val_loss + global_val_loss
            self.log('val_global_loss', global_val_loss, on_epoch=True, sync_dist=True)
            
        # Get center bias dataset for metrics
        if self.dataset == "salicon":
            center_bias_path = f"{SALICON_PATH}/center_bias.jpg"
            center_bias = torch.tensor(np.array(Image.open(center_bias_path).convert("L"))).float().to(self.device)
        elif self.dataset == "dhf1k":
            center_bias_path = f"{DHF1K_PATH}/center_bias.jpg"
            center_bias = torch.tensor(np.array(Image.open(center_bias_path).convert("L"))).float().to(self.device)
        elif self.dataset == "viewout":
            center_bias_path = f"{VIEWOUT_PATH}/center_bias.jpg"
            center_bias = torch.tensor(np.array(Image.open(center_bias_path).convert("L"))).float().to(self.device)
        
        # Calculate metrics
        metrics = {}
        if temporal_output is not None and temporal_ground_truth is not None:
            temporal_metrics = Metrics().get_metrics(temporal_output, temporal_ground_truth, center_bias_prior=center_bias)
            for key, value in temporal_metrics.items():
                self.log(f'val_temporal_{key}', value, on_epoch=True, sync_dist=True)
                metrics[f'temporal_{key}'] = value
                
        if global_output is not None and global_ground_truth is not None:
            global_metrics = Metrics().get_metrics(global_output, global_ground_truth, center_bias_prior=center_bias)
            for key, value in global_metrics.items():
                self.log(f'val_global_{key}', value, on_epoch=True, sync_dist=True)
                metrics[f'global_{key}'] = value
        
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True)
        return {'val_loss': val_loss, **metrics}
    
    def _process_batch_with_time(
        self, 
        batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        inputs, temporal_targets, global_targets, _ = batch
        batch_size = inputs.shape[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Forward pass through model
        temporal_output, global_output = self.model(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        batch_inference_time = end_time - start_time
        instance_inference_time = batch_inference_time / batch_size

        # Calculate losses using criterion
        temporal_loss = None
        global_loss = None
        if temporal_output is not None and temporal_targets is not None:
            temporal_loss = self.criterion(temporal_output, temporal_targets)
        if global_output is not None and global_targets is not None:
            global_loss = self.criterion(global_output, global_targets)
            
        return (
            temporal_loss,
            global_loss,
            temporal_output,
            global_output,
            temporal_targets,
            global_targets,
            instance_inference_time,
        )
    
    def test_step(self, batch, batch_idx):
        temporal_test_loss, global_test_loss, temporal_output, global_output, temporal_ground_truth, global_ground_truth, instance_inference_time = self._process_batch_with_time(batch)
        
        # Initialize metrics dictionary
        metrics = {}
        self.log('instance_inference_time', instance_inference_time, on_epoch=True, sync_dist=True)
        metrics['instance_inference_time'] = instance_inference_time

        # Calculate total validation loss
        test_loss = 0
        if temporal_test_loss is not None:
            test_loss = test_loss + temporal_test_loss
            self.log('test_temporal_loss', temporal_test_loss, on_epoch=True, sync_dist=True)
        if global_test_loss is not None:
            test_loss = test_loss + global_test_loss
            self.log('test_global_loss', global_test_loss, on_epoch=True, sync_dist=True)

        # Get center bias dataset for metrics
        if self.dataset == "salicon":
            center_bias_path = f"{SALICON_PATH}/center_bias.jpg"
            center_bias = torch.tensor(np.array(Image.open(center_bias_path).convert("L"))).float().to(self.device)
        elif self.dataset == "dhf1k":
            center_bias_path = f"{DHF1K_PATH}/center_bias.jpg"
            center_bias = torch.tensor(np.array(Image.open(center_bias_path).convert("L"))).float().to(self.device)

        # Calculate metrics
        if temporal_output is not None and temporal_ground_truth is not None:
            # Get global metrics for every temporal estimation
            temporal_metrics = Metrics().get_metrics(temporal_output, temporal_ground_truth, center_bias_prior=center_bias)
            for key, value in temporal_metrics.items():
                self.log(f'test_temporal_{key}', value, on_epoch=True, sync_dist=True)
                metrics[f'temporal_{key}'] = value
            # Get separate metrics for each temporal 
            for i in range(temporal_output.shape[1]):
                temporal_output_i = temporal_output[:, i, :, :]
                temporal_ground_truth_i = temporal_ground_truth[:, i, :, :]
                temporal_metrics_i = Metrics().get_metrics(temporal_output_i, temporal_ground_truth_i, center_bias_prior=center_bias)
                for key, value in temporal_metrics_i.items():
                    self.log(f'test_temporal_{i}_{key}', value, on_epoch=True, sync_dist=True)
                    metrics[f'temporal_{i}_{key}'] = value

        if global_output is not None and global_ground_truth is not None:
            global_metrics = Metrics().get_metrics(global_output, global_ground_truth, center_bias_prior=center_bias)
            for key, value in global_metrics.items():
                self.log(f'test_global_{key}', value, on_epoch=True, sync_dist=True)
                metrics[f'global_{key}'] = value

        self.log('test_loss', test_loss, on_epoch=True, sync_dist=True)
        return {'test_loss': test_loss, **metrics}
       
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input, _, _, sample_ids = batch
        temporal_output, global_output = self.model(input)
        return temporal_output, global_output, sample_ids

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': learning_rate_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            }
        }
