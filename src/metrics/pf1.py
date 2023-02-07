import torch
from torchmetrics import Metric
import numpy as np
from typing import Tuple


class BinaryProbF1(Metric):
    pass


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
