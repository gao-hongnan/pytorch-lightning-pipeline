"""Probabilistic F1 score for binary classification."""
import torch
from torchmetrics import Metric
import numpy as np
from typing import Tuple


import torch
from torchmetrics import Metric


import torch
from torchmetrics import Metric


class BinaryPFBeta(Metric):
    """
    PyTorch Metric that computes the Probabilistic F-beta score for binary classification.

    Args:
        beta: Float value of beta parameter in F-beta score. Default is 1.
        dist_sync_on_step: Synchronize metric state across processes at each forward pass. Default is False.

    Shape:
        - Preds: (N, ) or (N, C)
        - Labels: (N, ) where each value is either 0 or 1

    Example:
        >>> import torch
        >>> from torchmetrics import PFBeta
        >>> preds = torch.tensor([0.8, 0.6, 0.3, 0.2])
        >>> labels = torch.tensor([1, 0, 1, 0])
        >>> metric = PFBeta(beta=0.5)
        >>> metric(preds, labels)
        tensor(0.4732)

    Reference:
        [1] Powers, David M. "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness & correlation."
            Journal of machine learning technologies 2.1 (2011): 37-63.
    """

    def __init__(self, beta: float = 1, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.beta = beta
        self.add_state("ctp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cfp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("y_true_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Update the state variables for computing PFBeta.

        Args:
            preds: Predicted probabilities of shape (N, ) or (N, C).
            labels: Ground truth binary labels of shape (N, ).

        Returns:
            None
        """
        if preds.ndim == 2:
            preds = preds[:, 1]
        preds = preds.clip(0, 1)
        self.y_true_count += labels.sum()
        self.ctp += preds[labels == 1].sum()
        self.cfp += preds[labels == 0].sum()

    def compute(self) -> float:
        """
        Compute the PFBeta score.

        Returns:
            The computed PFBeta score as a float.
        """
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
        """
        Reset the state variables.

        Returns:
            None
        """
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
