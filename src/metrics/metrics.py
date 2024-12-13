import torch
import torch.nn as nn
import torch.nn.functional as F


class Metrics:
    def __init__(self, eps: float = 1e-7):
        super(Metrics, self).__init__()
        self.eps = eps

    def _reshape_4d(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4:
            batch_size, sequence_length, height, width = pred.shape
            pred = pred.view(batch_size * sequence_length, height, width)
            target = target.view(batch_size * sequence_length, height, width)
        return pred, target

    def _normalize_map(self, saliency_map: torch.Tensor) -> torch.Tensor:
        normalized = saliency_map / (
            saliency_map.sum(dim=(-2, -1), keepdim=True) + self.eps
        )
        return normalized

    def kldiv(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        # Reshape as 2D tensors
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # Apply softmax to both predictions and targets
        pred = torch.log_softmax(pred, dim=1)
        target = torch.softmax(target, dim=1)

        # Calculate KL divergence
        kl = F.kl_div(
            pred,
            target,
            reduction="batchmean",
            log_target=False,
        )

        return kl

    def cc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        # Calculate correlation coefficient
        pred_mean = pred.mean(dim=(-2, -1), keepdim=True)
        target_mean = target.mean(dim=(-2, -1), keepdim=True)

        # Center the maps by subtracting their means
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Calculate correlation
        covariance = (pred_centered * target_centered).sum(dim=(-2, -1))
        pred_variance = torch.sqrt((pred_centered**2).sum(dim=(-2, -1)))
        target_variance = torch.sqrt((target_centered**2).sum(dim=(-2, -1)))
        correlation_coefficient = covariance / (
            pred_variance * target_variance + self.eps
        )

        return correlation_coefficient.mean()

    def nss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        # Normalize prediction to have zero mean and unit standard deviation
        pred_mean = pred.mean(dim=(-2, -1), keepdim=True)
        pred_std = pred.std(dim=(-2, -1), keepdim=True)
        pred_normalized = (pred - pred_mean) / (pred_std + self.eps)

        # Calculate sum of saliency map values at fixation locations
        pred_target_sum = (pred_normalized * target).sum(dim=(-2, -1))
        target_sum = target.sum(dim=(-2, -1))

        # Calculate NSS score
        nss_score = pred_target_sum / (target_sum + self.eps)

        return nss_score.mean()

    def sim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        # Normalize maps
        pred = self._normalize_map(pred)
        target = self._normalize_map(target)

        # Calculate similarity
        sim_score = torch.min(pred, target).sum(dim=(-2, -1))
        return sim_score.mean()

    def information_gain(
        self, pred: torch.Tensor, target: torch.Tensor, center_bias_prior: torch.Tensor
    ) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        center_bias_prior = center_bias_prior.unsqueeze(0).unsqueeze(0)
        center_bias_prior = F.interpolate(
            center_bias_prior,
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize predictions and center bias prior
        pred = self._normalize_map(pred)
        center_bias_prior = self._normalize_map(center_bias_prior)

        # Calculate log-likelihood
        pred_log_likelihood = torch.log2(pred + self.eps) * target
        prior_log_likelihood = torch.log2(center_bias_prior + self.eps) * target

        # Calculate information gain
        information_gain = (pred_log_likelihood - prior_log_likelihood).sum(
            dim=(-2, -1)
        ) / target.sum(dim=(-2, -1))

        return information_gain.mean()

    def get_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        center_bias_prior: torch.Tensor = None,
    ) -> dict:
        metrics = {
            "kldiv": self.kldiv(pred, target),
            "cc": self.cc(pred, target),
            "nss": self.nss(pred, target),
            "sim": self.sim(pred, target),
        }

        if center_bias_prior is not None:
            metrics["ig"] = self.information_gain(pred, target, center_bias_prior)

        return metrics
