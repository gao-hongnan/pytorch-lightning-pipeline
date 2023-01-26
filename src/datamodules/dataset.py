from __future__ import annotations

from typing import Any, Dict, Optional, Union

import albumentations
import cv2
import pandas as pd
import torch
import torchvision

from configs.base import Config
from src.datamodules.base import AbstractDataset
from src.utils.types import TransformTypes


class ImageClassificationDataset(AbstractDataset):
    """A sample template for Image Classification Dataset."""

    def __init__(
        self,
        config: Config,
        df: Optional[pd.DataFrame] = None,
        transforms: TransformTypes = None,
        stage: str = "train",
        **kwargs: Dict[str, Any],
    ) -> None:
        """Constructor for the dataset class."""
        super().__init__(config, **kwargs)
        self.image_path = df[config.datamodule.dataset.image_path_col_name].values
        self.image_ids = df[config.datamodule.dataset.image_col_name].values
        self.targets = (
            df[config.datamodule.dataset.target_col_name].values
            if stage != "test"
            else None
        )
        self.df = df
        self.transforms = transforms
        self.stage = stage
        self.config = config

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def apply_image_transforms(
        self, image: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, albumentations.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(
            self.transforms, torchvision.transforms.Compose
        ):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # convert HWC to CHW
        return torch.tensor(image, dtype=dtype)

    # pylint: disable=no-self-use
    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target."""
        return torch.tensor(target, dtype=dtype)

    def __getitem__(
        self, index: int
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """Implements the getitem method."""
        image_path = self.image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        target = self.targets[index] if self.stage != "test" else torch.ones(1)
        target = self.apply_target_transforms(target)

        if self.stage in ["train", "valid", "evaluate", "debug"]:
            return image, target
        elif self.stage == "test":
            return image
        elif self.stage == "gradcam":
            # get image id as well to show on matplotlib image!
            # original image is needed to overlay the heatmap
            original_image = cv2.resize(
                cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
                (
                    self.config.datamodule.transforms.image_size,
                    self.config.datamodule.transforms.image_size,
                ),
            )
            return original_image, image, target, self.image_ids[index]
        else:
            raise ValueError(f"Invalid stage {self.stage}.")
