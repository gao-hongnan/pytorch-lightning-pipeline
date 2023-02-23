"""Pooling layers.

Generalized Mean Pooling (GeM) layer: https://amaarora.github.io/2020/08/30/gempool.html
"""
import torch.nn.functional as F
import torch
from torch import nn


class GeM(nn.Module):
    def __init__(
        self, p: int = 3, eps: float = 1e-6, p_trainable: bool = False
    ) -> None:
        super().__init__()
        self.p_trainable = p_trainable
        if self.p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ret = self.gem(inputs)
        return ret

    def gem(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            inputs.clamp(min=self.eps).pow(self.p), (inputs.size(-2), inputs.size(-1))
        ).pow(1.0 / self.p)

    # def __repr__(self) -> str:
    #     return (
    #         self.__class__.__name__
    #         + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})"
    #     )
