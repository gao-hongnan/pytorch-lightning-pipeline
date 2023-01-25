"""Concrete implementation of DataModule base class."""
# pylint: disable=all
from __future__ import annotations

import os
from pathlib import Path
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Any, Dict, Optional, Iterable, Union, List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, MISSING
from sklearn import model_selection
import torchvision.transforms as T
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, MulticlassAUROC, MulticlassAccuracy
import torch
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.csv_logs import CSVLogger

# pylint:disable=no-name-in-module
from pydantic import BaseModel, conint, validator


# pylint: disable=no-self-argument, no-self-use, too-few-public-methods


class Dataset(BaseModel):
    # filepath configs
    root_dir: Path
    train_dir: Path
    train_csv: Path
    test_dir: Path
    test_csv: Path
    url: Optional[str]
    blob_file: Optional[str]
    image_extension: Optional[str]

    # dataset configs
    image_col_name: str
    image_path_col_name: str
    target_col_name: str
    group_by: str
    stratify_by: str
    class_name_to_id: Dict[str, int]


class Resample(BaseModel):
    resample_strategy: str
    resample_params: Dict[str, Any]

    # @validator("resample_strategy")
    # def validate_resample_strategy(cls, resample_strategy: str) -> str:
    #     """Validates resample_strategy is in sklearn.model_selection."""
    #     try:
    #         _ = getattr(model_selection, resample_strategy)
    #     except AttributeError:
    #         raise ValueError(
    #             f"resample_strategy must be in {model_selection.__all__}"
    #         ) from BaseException


class Transforms(BaseModel):
    image_size: conint(ge=1)
    mean: Any  # List[float] but hydra gives OmegaConf.ListConfig
    std: Any
    inverse_mean: Any
    inverse_std: Any
    mixup: Optional[bool] = False
    mixup_params: Optional[Dict[str, Any]] = None
    train_transforms: Optional[T.Compose] = None
    valid_transforms: Optional[T.Compose] = None
    test_transforms: Optional[T.Compose] = None

    class Config:
        arbitrary_types_allowed = True  # allow T.Compose


class Dataloader(BaseModel):
    train_loader: Dict[str, Any]
    valid_loader: Dict[str, Any]
    test_loader: Optional[Dict[str, Any]]
    gradcam_loader: Dict[str, Any]


class DataModule(BaseModel):
    dataset: Dataset
    resample: Resample
    transforms: Transforms
    dataloader: Dataloader
    debug: bool = False
    num_debug_samples: int = 128
    fold: int = 1  # this is the validation fold
    upsample: int


class Model(BaseModel):
    model_name: str
    pretrained: bool
    in_chans: conint(ge=1)  # in_channels must be greater than or equal to 1
    num_classes: conint(ge=1)
    global_pool: str

    @validator("global_pool")
    def validate_global_pool(cls, global_pool: str) -> str:
        """Validates global_pool is in ["avg", "max"]."""
        if global_pool not in ["avg", "max"]:
            raise ValueError("global_pool must be avg or max")
        return global_pool


class TimmModel(Model):
    timm_kwargs: Dict[str, Any]


class Criterion(BaseModel):
    criterion: str
    criterion_kwargs: Dict[str, Any]


class Optimizer(BaseModel):
    optimizer: str
    optimizer_kwargs: Dict[str, Any]


class Scheduler(BaseModel):
    scheduler: str
    scheduler_kwargs: Dict[str, Any]


class Metrics(BaseModel):
    metrics: Dict[str, Metric]

    class Config:
        arbitrary_types_allowed = True


class Stores(BaseModel):
    project_name: str
    unique_id: str
    logs_dir: Path
    model_artifacts_dir: Path


class General(BaseModel):
    num_classes: int
    device: str  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name: str
    debug: bool
    stage: str  # "train", "test", "inference"
    seed: int
    platform: str  # "colab", "kaggle" or "local"
    unique_id: str


class Trainer(BaseModel):
    """All of pytorch-lightning.Trainer configs.
    See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    """

    accelerator: Optional[str] = None
    devices: Optional[int] = None
    fast_dev_run: Optional[Union[int, bool]] = None
    log_every_n_steps: int
    max_epochs: int
    overfit_batches: Optional[Union[int, float]] = 0.0
    logger: Optional[Union[Logger, Iterable[Logger], bool]]
    precision: Union[Literal[64, 32, 16], Literal["64", "32", "16", "bf16"]]
    callbacks: Optional[List[Callback]] = None

    class Config:
        arbitrary_types_allowed = True


class Config(BaseModel):
    datamodule: DataModule
    model: TimmModel
    criterion: Criterion
    optimizer: Optimizer
    scheduler: Scheduler
    metrics: Metrics
    trainer: Trainer
    stores: Stores
    general: General

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        """Creates Config object from a dictionary."""
        return cls(**config_dict)


if __name__ == "__main__":
    # TODO: interpolation
    # 1. num_classes
    # 2. num_epochs (for scheduler) and num_epochs (for training)
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
                    'num_workers': 0,
                    'pin_memory': True,
                    'drop_last': False,
                    'shuffle': False,
                    'collate_fn': None,
                },
                'train_loader': {
                    'batch_size': 32,
                    'num_workers': 0,
                    'pin_memory': True,
                    'drop_last': False,
                    'shuffle': True,
                    'collate_fn': None,
                },
                'valid_loader': {
                    'batch_size': 32,
                    'num_workers': 0,
                    'pin_memory': True,
                    'drop_last': False,
                    'shuffle': False,
                    'collate_fn': None,
                },
            },
            'debug': False,
            'num_debug_samples': 128,
            'fold': 1,
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
            'device': "gpu",
            'project_name': 'rsna',
            'debug': False,
            'seed': 1992,
            'platform': 'local',
        },
        "trainer": {
            'accelerator': 'cpu',
            'devices': None,
            'fast_dev_run': 1,
            'log_every_n_steps': 1,
            'max_epochs': 3,
            'overfit_batches': 0.0,
            'logger': CSVLogger(save_dir="./logs"),
            'precision': 16,
            'callbacks': None,
        },
    }

    config = Config.from_dict(config_dict)
    print(config)
    print(config.dict())