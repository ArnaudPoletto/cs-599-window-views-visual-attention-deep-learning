import torch
import torch.nn as nn
import torch.nn.functional as F


class Metrics():
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

    def auc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Sort predictions and corresponding targets
        sorted_indices = torch.argsort(pred_flat, descending=True)
        sorted_target = target_flat[sorted_indices]

        # Calculate TPR and FPR
        tp = torch.cumsum(sorted_target, dim=0)
        fp = torch.cumsum(1 - sorted_target, dim=0)

        # Calculate rates
        total_positives = target_flat.sum()
        total_negatives = len(target_flat) - total_positives
        tpr = tp / (total_positives + self.eps)
        fpr = fp / (total_negatives + self.eps)

        # Calculate AUC using trapezoidal rule
        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        auc_score = torch.sum(width * height)

        return auc_score

    def nss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        # Normalize prediction to have zero mean and unit standard deviation
        pred_normalized = (pred - pred.mean(dim=(-2, -1), keepdim=True)) / (
            pred.std(dim=(-2, -1), keepdim=True) + self.eps
        )

        # Calculate NSS score
        nss_score = (pred_normalized * target).sum(dim=(-2, -1)) / (
            target.sum(dim=(-2, -1)) + self.eps
        )
        return nss_score.mean()

    def cc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._reshape_4d(pred, target)

        # Calculate correlation coefficient
        pred_mean = pred.mean(dim=(-2, -1), keepdim=True)
        target_mean = target.mean(dim=(-2, -1), keepdim=True)

        # Center the maps by subtracting their means
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Calculate correlation
        numerator = (pred_centered * target_centered).sum(dim=(-2, -1))
        denominator = torch.sqrt(
            (pred_centered**2).sum(dim=(-2, -1))
            * (target_centered**2).sum(dim=(-2, -1))
            + self.eps
        )

        return (numerator / denominator).mean()

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
        """Calculate Information Gain relative to center bias prior."""
        pred, target = self._reshape_4d(pred, target)
        
        center_bias_prior = center_bias_prior.unsqueeze(0).unsqueeze(0)
        center_bias_prior = F.interpolate(center_bias_prior, size=pred.shape[-2:], mode='bilinear', align_corners=False)

        # Normalize predictions and center bias prior
        pred = self._normalize_map(pred)
        center_bias_prior = self._normalize_map(center_bias_prior)

        # Calculate log-likelihood
        pred_ll = torch.log2(pred + self.eps) * target
        prior_ll = torch.log2(center_bias_prior + self.eps) * target

        # Calculate information gain
        ig = (pred_ll - prior_ll).sum(dim=(-2, -1)) / target.sum(dim=(-2, -1))
        return ig.mean()

    def get_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        center_bias_prior: torch.Tensor = None,
    ) -> dict:
        """Calculate all metrics at once."""
        metrics = {
            "auc": self.auc(pred, target),
            "nss": self.nss(pred, target),
            "cc": self.cc(pred, target),
            "sim": self.sim(pred, target),
        }

        if center_bias_prior is not None:
            metrics["ig"] = self.information_gain(pred, target, center_bias_prior)

        return metrics
