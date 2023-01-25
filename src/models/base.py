"""Model Interface with config constructor signature."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from configs.base import Config


class Model(ABC, nn.Module):
    """Model Base Class."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def create_backbone(self) -> nn.Module:
        """Create the (backbone) of the model."""
        raise NotImplementedError("Please implement backbone for the model.")

    @abstractmethod
    def create_head(self) -> nn.Module:
        """Create the head of the model."""
        raise NotImplementedError("Please implement head for the model.")

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the backbone."""

    def forward_head(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the head."""
