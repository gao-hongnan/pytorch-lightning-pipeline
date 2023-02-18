"""Concrete implementation of the Model class for timm models."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import timm
import torch
import torchinfo
from rich.pretty import pretty_repr, pprint
from torch import nn
from torchinfo.model_statistics import ModelStatistics

from configs.base import Config
from src.models.base import Model
from src.models.pooling import GeM


class TimmModel(Model):
    """Model class specific to timm models."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.model_name = config.model.model_name
        self.global_pool = config.model.global_pool
        self.in_chans = config.model.in_chans
        self.num_classes = config.model.num_classes
        self.timm_kwargs = config.model.timm_kwargs
        print(f"Creating model: {self.model_name}")

        self.backbone = self.create_backbone()
        self.head = self.create_head()

        # hardcoded and call this after creating head since head needs to know
        # in_features in create_head() call.
        self.backbone.reset_classifier(num_classes=0, global_pool=self.global_pool)

        # run sanity check
        # self.run_sanity_check()

        # print model summary
        pprint(self)

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

    def run_sanity_check(self) -> None:
        """Post init sanity check."""
        inputs = torch.randn(4, 3, 224, 224)  # assume 3 channel images
        features = self.forward_features(inputs)
        logits = self.forward_head(features)
        print(f"Features shape: {features.shape}")
        print(f"Logits shape: {logits.shape}")

    def __str__(self):
        return pretty_repr(self)

    def model_summary(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Dict[str, Any]
    ) -> ModelStatistics:
        """Wrapper for torchinfo package to get the model summary."""
        if input_size is None:
            input_size = (1, self.in_chans, 224, 224)
        return torchinfo.summary(self, input_size=input_size, **kwargs)


class TimmModelWithGeM(TimmModel):
    """Model class specific to timm models."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # no super init because we don't want to call the parent's init
        # as run_sanity_check will fail in parent class with new attribute gem
        self.model_name = config.model.model_name
        self.global_pool = config.model.global_pool
        self.num_classes = config.model.num_classes
        self.timm_kwargs = config.model.timm_kwargs
        print(f"Creating model: {self.model_name}")

        self.backbone = self.create_backbone()
        self.head = self.create_head()

        # FIXME: hardcoded the global_pool to "" since GeM is used this is bad
        # because in config we have global_pool set to "avg" but it is not used.
        # Find out a way to fix this.
        self.backbone.reset_classifier(num_classes=0, global_pool="")

        self.gem = GeM(p_trainable=False)

        # run sanity check
        # self.run_sanity_check()

        # print model summary
        pprint(self)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        gem_features = self.gem(features)
        gem_features = gem_features[:, :, 0, 0]  # flatten
        return gem_features
