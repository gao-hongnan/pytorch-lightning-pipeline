from __future__ import annotations

import timm
import torch
from torch import nn

from configs.base import Config
from src.models.base import Model


class TimmModel(Model):
    """Model class specific to timm models."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.model_name = config.model.model_name
        self.global_pool = config.model.global_pool
        self.num_classes = config.model.num_classes
        self.timm_kwargs = config.model.timm_kwargs
        print(f"Creating model: {self.model_name}")

        self.backbone = self.create_backbone()
        self.head = self.create_head()

        # hardcoded and call this after creating head since head needs to know
        # in_features in create_head() call.
        self.backbone.reset_classifier(num_classes=0, global_pool=self.global_pool)

    def create_backbone(self) -> nn.Module:
        """Create backbone model using timm's create_model function."""
        # TODO: pack all attributes to just one timm_kwargs
        return timm.create_model(**self.timm_kwargs)

    def create_head(self) -> nn.Module:
        in_features = self.backbone.get_classifier().in_features
        return nn.Linear(in_features, self.num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def forward_head(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        logits = self.forward_head(features)
        return logits
