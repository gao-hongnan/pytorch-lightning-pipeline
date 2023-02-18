"""Probabilistic F1 score for binary classification."""
import torch
from torchmetrics import Metric
import numpy as np
from typing import Tuple


import torch
from torchmetrics import Metric


class BinaryPFBeta(Metric):
    def __init__(self, beta: float = 1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.beta = beta
        self.add_state("ctp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cfp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("y_true_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == 2:
            preds = preds[:, 1]
        preds = preds.clip(0, 1)
        self.y_true_count += target.sum()
        self.ctp += preds[target == 1].sum()
        self.cfp += preds[target == 0].sum()

    def compute(self):
        c_precision = self.ctp / (self.ctp + self.cfp)
        c_recall = self.ctp / self.y_true_count
        beta_squared = self.beta ** 2

        if c_precision > 0 and c_recall > 0:
            return (
                (1 + beta_squared)
                * (c_precision * c_recall)
                / (beta_squared * c_precision + c_recall)
            ).item()
        return 0.0

    def reset(self):
        self.ctp = torch.tensor(0)
        self.cfp = torch.tensor(0)
        self.y_true_count = torch.tensor(0)


def pfbeta_torch(preds: torch.Tensor, labels: torch.Tensor, beta: float = 1) -> float:
    """PyTorch implementation of the Probabilistic F-beta score for binary classification.
    Preds must be either a 1D tensor of probabilities or a 2D tensor of probs."""
    if preds.ndim == 2:
        preds = preds[:, 1]
    preds = preds.clip(0, 1)

    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()

    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count

    if c_precision > 0 and c_recall > 0:
        return (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        ).item()
    return 0.0


def optimize_thresholds(
    preds: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, float]:
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()

    f1_thresholded = []
    thresholds = np.linspace(0.001, 0.999, 999)
    for threshold in thresholds:
        predictions_thresholded = (preds > threshold).astype(int)
        f1 = pfbeta_torch(predictions_thresholded, labels)
        f1_thresholded.append(float(f1))
    max_f1 = np.max(f1_thresholded)
    best_threshold = thresholds[np.argmax(f1_thresholded)]
    return max_f1, best_threshold
