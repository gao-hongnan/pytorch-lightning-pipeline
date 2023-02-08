"""Controller for training pipeline."""
import logging
import warnings
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from configs.base import Config
from examples.image_classification.rsna_breast_cancer_detection.run import run

warnings.filterwarnings(action="ignore", category=UserWarning)

logger: logging.Logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

from configs.base import Config
from src.datamodules.datamodule import ImageClassificationDataModule
from src.models.lightning_module import ImageClassificationLightningModel
from src.models.model import TimmModel
from src.utils.general import GradCamWrapper, create_folds, preprocess, read_data_as_df

# pylint: disable=all
def run(config: Config) -> None:
    """Run the experiment."""

    pl.seed_everything(config.general.seed)

    train_file = config.datamodule.dataset.train_csv
    df = read_data_as_df(train_file)
    df["image_id"] = "dummy"

    df_folds = create_folds(df, config)
    print(df_folds.head())

    # dm = RSNADataModule(config, df_folds)
    dm = ImageClassificationDataModule(config, df_folds)
    dm.prepare_data()

    model = TimmModel(config)

    module = ImageClassificationLightningModel(config, model)
    trainer = pl.Trainer(**config.trainer.dict())

    if config.general.stage == "train":
        dm.setup(stage="train")
        # for OneCycleLR
        print(f"Dataloader length: {len(dm.train_dataloader())}")
        for batch in dm.train_dataloader():
            print(batch)
            break
        trainer.fit(module, datamodule=dm)
