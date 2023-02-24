"""A sample template for Image Classification DataModule."""
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from configs.base import Config
from src.datamodules.dataset import ImageClassificationDataset


# pylint: disable=too-many-instance-attributes
class ImageClassificationDataModule(pl.LightningDataModule):
    """Data module for generic image classification dataset."""

    def __init__(
        self,
        config: Config,
        df_folds: pd.DataFrame,
        fold: int,
        test_df: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.df_folds = df_folds
        self.fold = fold  # instead of config.datamodule.fold for clarity
        self.test_df = test_df

        self.train_df: pd.DataFrame
        self.valid_df: pd.DataFrame
        self.oof_df: pd.DataFrame
        self.train_dataset: ImageClassificationDataset
        self.valid_dataset: ImageClassificationDataset
        self.gradcam_dataset: ImageClassificationDataset
        self.test_dataset: ImageClassificationDataset

    def prepare_data(self) -> None:
        """Prepare the data for training and validation.
        This method prepares state that needs to be set once per node (i.e. download data, etc.).

        DO NOT SET STATE HERE!
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign train/val datasets for use in dataloaders.
        This method is called on every GPU in distributed training.

        Note that we need to adhere to the allowed namings for the stages:
        - fit, evaluate, predict etc."""
        print(f"Stage: {stage}")
        print(f"Using Fold Number {self.fold}")
        # FIXME: now no matter what stage, train_df and valid_df will be run. Not good
        # if I use stage=="test".
        self.train_df = self.df_folds[self.df_folds["fold"] != self.fold].reset_index(
            drop=True
        )
        self.valid_df = self.df_folds[self.df_folds["fold"] == self.fold].reset_index(
            drop=True
        )
        self.oof_df = self.valid_df.copy()

        if self.config.datamodule.debug:
            num_debug_samples = self.config.datamodule.num_debug_samples
            print(f"Debug mode is on, using {num_debug_samples} images for training.")
            self.train_df = self.train_df.sample(num_debug_samples)
            self.valid_df = self.valid_df.sample(num_debug_samples)
            self.oof_df = self.valid_df.copy()

        if stage == "fit":
            train_transforms = self.config.datamodule.transforms.train_transforms
            valid_transforms = self.config.datamodule.transforms.valid_transforms

            self.train_dataset = ImageClassificationDataset(
                self.config,
                df=self.train_df,
                dataset_stage="train",
                transforms=train_transforms,
            )
            self.valid_dataset = ImageClassificationDataset(
                self.config,
                df=self.valid_df,
                dataset_stage="valid",
                transforms=valid_transforms,
            )
            self.gradcam_dataset = ImageClassificationDataset(
                self.config,
                df=self.valid_df,
                dataset_stage="gradcam",
                transforms=valid_transforms,
            )

        if stage == "test":
            test_transforms = self.config.datamodule.transforms.test_transforms
            self.test_dataset = ImageClassificationDataset(
                self.config,
                df=self.test_df,
                dataset_stage="test",
                transforms=test_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            **self.config.datamodule.dataloader.train_loader,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset, **self.config.datamodule.dataloader.valid_loader
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, **self.config.datamodule.dataloader.test_loader
        )

    def gradcam_dataloader(self) -> DataLoader:
        """Gradcam dataloader."""
        return DataLoader(
            self.gradcam_dataset, **self.config.datamodule.dataloader.gradcam_loader
        )
