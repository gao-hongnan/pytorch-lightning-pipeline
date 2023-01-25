"""Controller for training pipeline."""
import logging
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torchvision.transforms as T
from configs.base import Config
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from rich.pretty import pprint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

from utils import *
from configs.base import Config
from rsna import *

logger: logging.Logger = logging.getLogger(__name__)


def run(config: Config) -> None:
    """Run the experiment."""

    pl.seed_everything(config.general.seed)

    df = read_data_as_df(config)
    df = preprocess(df, config)
    df_folds = create_folds(df, config)
    print(df.head())

    # dm = RSNADataModule(config, df_folds)
    dm = RSNAUpsampleDataModule(config, df_folds)  # upsampled
    dm.prepare_data()

    model = TimmModel(config)
    inputs = torch.randn(4, 3, 224, 224)
    features = model.forward_features(inputs)
    logits = model.forward_head(features)
    print(features.shape)
    print(logits.shape)

    module = RSNALightningModel(config, model)
    print(config.trainer.dict())
    trainer = pl.Trainer(**config.trainer.dict())

    if config.general.stage == "train":
        dm.setup(stage="train")
        # print(dm.train_dataset[0][0].shape)
        trainer.fit(module, datamodule=dm)
    elif config.general.stage == "gradcam":
        dm.setup(stage="train")
        checkpoint = "/Users/reighns/gaohn/pytorch-lightning-hydra/rsna/logs/lightning_logs/version_7/checkpoints/epoch=2-step=12.ckpt"
        # module = module.load_from_checkpoint(checkpoint)
        module.load_state_dict(torch.load(checkpoint)["state_dict"])
        gradcam_loader = dm.gradcam_dataloader()

        originals, inputs, labels, image_ids = next(iter(gradcam_loader))
        originals = originals.cpu().detach().numpy() / 255.0

        # inputs = config.datamodule.transforms.valid_transforms(inputs)

        target_layers = [module.model.backbone.layer4[-1]]
        gradcam = GradCAM(model=module, target_layers=target_layers, use_cuda=False)

        heatmaps = gradcam(inputs, targets=None, eigen_smooth=False)

        print(inputs.shape, heatmaps.shape)

        plt.figure(figsize=(20, 20))

        for index, (original, input, heatmap) in enumerate(
            zip(originals, inputs, heatmaps)
        ):
            print(type(original), type(input), type(heatmap))
            overlay = show_cam_on_image(original, heatmap, use_rgb=False)
            plt.subplot(8, 4, index + 1)
            # plt.imshow(image)
            # plt.imshow(heatmap, cmap="jet", alpha=0.5)
            plt.imshow(overlay, cmap="gray")
            plt.axis("off")
        plt.show()
    elif config.general.stage == "inference":
        print("Inference mode")

        # checkpoint = "/Users/gaohn/gao/pytorch-lightning-hydra/rsna/logs/lightning_logs/version_1/checkpoints/epoch=2-step=12.ckpt"
        checkpoint = "/Users/reighns/gaohn/pytorch-lightning-hydra/rsna/logs/lightning_logs/version_7/checkpoints/epoch=2-step=12.ckpt"
        # module = module.load_from_checkpoint(checkpoint)
        module.load_state_dict(torch.load(checkpoint)["state_dict"])
        valid_loader = dm.val_dataloader()
        # predictions = trainer.predict(
        #     module, dataloaders=valid_loader, ckpt_path=checkpoint
        # )
        # predictions = trainer.predict(module, dataloaders=valid_loader)
        # print(predictions)


def hydra_to_pydantic(config: DictConfig) -> Config:
    """Converts Hydra config to Pydantic config."""
    OmegaConf.resolve(config)
    return Config(**config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry to training pipeline."""
    logger.info(f"Config representation:\n{OmegaConf.to_yaml(config)}")
    logger.info(f"Output dir: {HydraConfig.get().runtime.output_dir}")

    if not Path(HydraConfig.get().runtime.output_dir).exists():
        Path(HydraConfig.get().runtime.output_dir).mkdir(parents=True)

    transforms = instantiate(config.datamodule.transforms)
    config.datamodule.transforms = transforms

    metrics = instantiate(config.metrics.metrics)
    config.metrics.metrics = metrics

    # callbacks = instantiate(config.callbacks.callbacks)
    trainer = instantiate(config.trainer)
    config.trainer = trainer

    config = hydra_to_pydantic(config)
    pprint(config)

    run(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


# seed_all(42) not needed since we are using pl.seed_everything(42)


# train_pf1: 0.310546875
# valid_pf1: 0.236328125
# loss=1.07, v_num=, train_accuracy_step=0.312, train_multiclass_auroc_step=0.564

# train_pf1: 0.09375
# valid_pf1: 0.19433593750
