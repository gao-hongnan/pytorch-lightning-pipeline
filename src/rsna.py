"""Concrete implementation of DataModule base class."""
# pylint: disable=all
from __future__ import annotations

import os
import sys

sys.path.insert(1, os.getcwd())

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torchvision.transforms as T
from configs.base import Config
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from rich.pretty import pprint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import *

TransformTypes = Optional[Union[A.Compose, T.Compose]]
BatchTensor = Tuple[torch.Tensor, torch.Tensor]
STEP_OUTPUT = Union[torch.Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]

# pylint: disable=invalid-name
class ImageClassificationDataset(Dataset):
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
        super().__init__(**kwargs)
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
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
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

        if self.stage in ["train", "valid", "debug"]:
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


# pylint: disable=too-many-instance-attributes
class RSNADataModule(pl.LightningDataModule):
    """Data module for generic image classification dataset."""

    def __init__(self, config: Config, df_folds: pd.DataFrame) -> None:
        super().__init__()
        self.config = config
        self.df_folds = df_folds
        self.fold = config.datamodule.fold

    def prepare_data(self) -> None:
        """Prepare the data for training and validation.
        This method prepares state that needs to be set once per node (i.e. download data, etc.).
        """
        print(f"Using Fold Number {self.fold}")
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

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders.
        This method is called on every GPU in distributed training."""
        if stage == "train":
            train_transforms = self.config.datamodule.transforms.train_transforms
            valid_transforms = self.config.datamodule.transforms.valid_transforms

            self.train_dataset = ImageClassificationDataset(
                self.config,
                df=self.train_df,
                stage="train",
                transforms=train_transforms,
            )
            self.valid_dataset = ImageClassificationDataset(
                self.config,
                df=self.valid_df,
                stage="valid",
                transforms=valid_transforms,
            )
            self.gradcam_dataset = ImageClassificationDataset(
                self.config,
                df=self.valid_df,
                stage="gradcam",
                transforms=valid_transforms,
            )

        if stage == "test":
            test_transforms = self.config.datamodule.transforms.test_transforms
            self.test_dataset = ImageClassificationDataset(
                self.config,
                df=self.test_df,
                stage="test",
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
        return DataLoader(
            self.gradcam_dataset, **self.config.datamodule.dataloader.gradcam_loader
        )


class RSNAUpsampleDataModule(RSNADataModule):
    def __init__(self, config: Config, df_folds: pd.DataFrame) -> None:
        super().__init__(config, df_folds)

    def prepare_data(self) -> None:
        """Prepare the data for training and validation.
        This method prepares state that needs to be set once per node (i.e. download data, etc.).
        """
        print(f"Using Fold Number {self.fold}")
        self.train_df = self.df_folds[self.df_folds["fold"] != self.fold].reset_index(
            drop=True
        )
        self.valid_df = self.df_folds[self.df_folds["fold"] == self.fold].reset_index(
            drop=True
        )
        self.oof_df = self.valid_df.copy()

        if self.config.datamodule.upsample:
            print("Upsampling the data")
            self.train_df = upsample_df(self.train_df, self.config)

        if self.config.datamodule.debug:
            num_debug_samples = self.config.datamodule.num_debug_samples
            print(f"Debug mode is on, using {num_debug_samples} images for training.")
            self.train_df = self.train_df.sample(num_debug_samples)
            self.valid_df = self.valid_df.sample(num_debug_samples)
            self.oof_df = self.valid_df.copy()


class TimmModel(nn.Module):
    """Model class specific to timm models."""

    def __init__(self, config: Config) -> None:
        super().__init__()
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


class RSNALightningModel(pl.LightningModule):
    """Lightning model class."""

    # TODO: add abstraction type hint
    # TODO: even though the constructor does not take in
    # optimizer, metrics etc, it is still not really violating
    # dependency inversion principle since the constructor
    # takes in config, which service locates the correct components.
    def __init__(self, config: Config, model: TimmModel) -> None:
        super().__init__()
        self.config = config
        self.config_dict = self.config.dict()
        self.model = model
        self.criterion = self._get_criterion()
        self.metrics = self._get_metrics()
        self.sigmoid_or_softmax = self._get_sigmoid_softmax()
        self.save_hyperparameters(ignore=["model", "config", "config_dict"])

    def _get_sigmoid_softmax(self) -> Union[nn.Sigmoid, nn.Softmax]:
        """Get the sigmoid or softmax function depending on loss function."""
        assert self.config.criterion.criterion in [
            "BCEWithLogitsLoss",
            "CrossEntropyLoss",
        ], "Criterion not supported"
        if self.config.criterion.criterion == "CrossEntropyLoss":
            return getattr(nn, "Softmax")(dim=1)
        return getattr(nn, "Sigmoid")()

    def _get_criterion(self) -> nn.Module:
        """Get loss function."""
        return getattr(nn, self.config.criterion.criterion)(
            **self.config.criterion.criterion_kwargs
        )

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = getattr(torch.optim, self.config.optimizer.optimizer)(
            self.model.parameters(), **self.config.optimizer.optimizer_kwargs
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.config.scheduler.scheduler)(
            optimizer, **self.config.scheduler.scheduler_kwargs
        )
        return [optimizer], [scheduler]

    def _get_metrics(self) -> nn.ModuleDict:
        """Get metrics."""
        metrics_collection = MetricCollection(self.config.metrics.metrics)
        return nn.ModuleDict(
            {
                "train_metrics": metrics_collection.clone(prefix="train_"),
                "valid_metrics": metrics_collection.clone(prefix="valid_"),
            }
        )

    def get_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get penultimate layer embeddings."""
        return self.model.forward_features(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward is needed in this module if you want to do self(inputs) instead
        of self.model(inputs)."""
        features = self.model.forward_features(inputs)
        logits = self.model.forward_head(features)
        return logits

    def training_step(self, batch: BatchTensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: BatchTensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "valid")

    def predict_step(self, batch: BatchTensor, batch_idx: int) -> torch.Tensor:
        """Predict step. Try-except block is to handle the case where
        I want to run inference on validation set, which has targets."""
        try:
            inputs, targets = batch
            logits = self(inputs)
            probs = self.sigmoid_or_softmax(logits)
            return probs, targets
        except ValueError:
            inputs = batch[0]
            logits = self(inputs)
            probs = self.sigmoid_or_softmax(logits)
            return probs

    def _shared_step(self, batch: BatchTensor, stage: str) -> torch.Tensor:
        """Shared step for train and validation step."""
        assert stage in ["train", "valid"], "stage must be either train or valid"

        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log(f"{stage}_loss", loss)

        probs = self.sigmoid_or_softmax(logits)

        pf1 = pfbeta_torch(probs, targets, beta=1)
        print(f"{stage}_pf1: {pf1}")
        self.log(f"{stage}_pf1", pf1)

        self.metrics[f"{stage}_metrics"](probs, targets)
        self.log_dict(
            self.metrics[f"{stage}_metrics"],
            on_step=True,  # whether to log on N steps
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "probs": probs, "targets": targets, "logits": logits}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """See source code for more info."""

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Good to use for logging validation metrics that are not
        calculated based on average. For example, if you want to log
        pf1, it is different from accumulating pf1 from each batch
        and then averaging it. Instead, you want to accumulate
        them and then calculate pf1 on the accumulated values."""

    def _shared_epoch_end(self, outputs: EPOCH_OUTPUT, stage: str) -> None:
        """Shared epoch end for train and validation epoch end."""
        # assert stage in ["train", "valid"], "stage must be either train or valid"
        # loss = torch.stack([x["loss"] for x in outputs]).mean()
        # self.log(f"{stage}_loss", loss)

        # probs = torch.cat([x["probs"] for x in outputs])
        # targets = torch.cat([x["targets"] for x in outputs])
        # logits = torch.cat([x["logits"] for x in outputs])

        # pf1 = pfbeta_torch(probs, targets, beta=1)
        # self.log(f"{stage}_pf1", pf1)

        # self.metrics[f"{stage}_metrics"](probs, targets)
        # self.log_dict(
        #     self.metrics[f"{stage}_metrics"],
        #     on_step=False, # whether to log on N steps
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )


config_dict = {
    "datamodule": {
        "dataset": {
            "root_dir": Path("./data"),
            "train_dir": Path("./data") / "train",
            "train_csv": Path("./data") / "train" / "train.csv",
            "test_dir": Path("./data") / "test",
            "test_csv": Path("./data") / "test" / "test.csv",
            "image_extension": "png",
            'image_col_name': 'patient_and_image_id',
            'image_path_col_name': 'image_path',
            'target_col_name': 'cancer',
            'group_by': 'patient_id',
            'stratify_by': 'cancer',
            'class_name_to_id': {'benign': 0, 'malignant': 1},
        },
        "resample": {
            "resample_strategy": "StratifiedGroupKFold",
            "resample_params": {"n_splits": 4, "shuffle": True, "random_state": 42},
        },
        "transforms": {
            "image_size": 64,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "inverse_mean": [-1, -1, -1],  # -mean/std
            "inverse_std": [2, 2, 2],  # 1/std
            "train_transforms": T.Compose(
                [
                    T.ToPILImage(),
                    T.RandomResizedCrop(64),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=45),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "valid_transforms": T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize(64),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "test_transforms": T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize(64),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
        },
        "dataloader": {
            'test_loader': {
                'batch_size': 32,
                'num_workers': 2,
                'pin_memory': True,
                'drop_last': False,
                'shuffle': False,
                'collate_fn': None,
            },
            'train_loader': {
                'batch_size': 32,
                'num_workers': 2,
                'pin_memory': True,
                'drop_last': False,
                'shuffle': True,
                'collate_fn': None,
            },
            'valid_loader': {
                'batch_size': 32,
                'num_workers': 2,
                'pin_memory': True,
                'drop_last': False,
                'shuffle': False,
                'collate_fn': None,
            },
            'gradcam_loader': {
                'batch_size': 8,
                'num_workers': 2,
                'pin_memory': True,
                'drop_last': False,
                'shuffle': False,
                'collate_fn': None,
            },
        },
        'debug': False,
        'num_debug_samples': 128,
        'fold': 1,
        "upsample": 10,
    },
    "model": {
        "model_name": "resnet18",
        "pretrained": True,
        "in_chans": 3,
        "num_classes": 2,
        "global_pool": "avg",
        "timm_kwargs": {
            "model_name": "resnet18",
            "pretrained": True,
            "in_chans": 3,
            "num_classes": 2,
            "global_pool": "avg",
        },
    },
    "metrics": {
        "metrics": {
            "accuracy": MulticlassAccuracy(num_classes=2, average="micro"),
            "multiclass_auroc": MulticlassAUROC(num_classes=2, average="macro"),
        }
    },
    "criterion": {
        "criterion": "CrossEntropyLoss",
        "criterion_kwargs": {
            "reduction": "mean",
            "weight": None,
            "label_smoothing": 0.0,
        },
    },
    "optimizer": {
        "optimizer": "AdamW",
        "optimizer_kwargs": {"lr": 3e-4, "weight_decay": 0.0},
    },
    "scheduler": {
        "scheduler": "CosineAnnealingLR",
        "scheduler_kwargs": {"T_max": 3, "eta_min": 1e-6},
    },
    'general': {
        'num_classes': 2,
        'device': "cpu",
        'project_name': 'rsna',
        'debug': False,
        'seed': 1992,
        'platform': 'local',
    },
    "trainer": {
        'accelerator': 'mps',
        'devices': 1,  # None
        #'fast_dev_run': 1,
        'log_every_n_steps': 1,
        'max_epochs': 3,
        'overfit_batches': 0.0,
        'logger': CSVLogger(save_dir="./logs"),
        'precision': 16,  # "bf16",
        'callbacks': None,
    },
}
