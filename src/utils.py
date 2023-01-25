import os
import random
from pathlib import Path
from typing import Union, List, Callable, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from configs.base import Config
from sklearn import model_selection
from torch import nn
from torchmetrics import Metric
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCamWrapper:
    def __init__(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,  # can be a batch of images
        target_categories: List[ClassifierOutputTarget],
        target_layers: List[torch.nn.Module],
        use_cuda: bool = False,
        gradcam_kwargs: Optional[Dict[str, Any]] = None,
        heatmap_kwargs: Optional[Dict[str, Any]] = None,
        showcam_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.input_tensor = input_tensor
        self.target_categories = target_categories
        self.target_layers = target_layers
        self.use_cuda = use_cuda
        self.gradcam_kwargs = gradcam_kwargs or {}
        self.heatmap_kwargs = heatmap_kwargs or {}
        self.showcam_kwargs = showcam_kwargs or {"use_rgb": True}

        self.gradcam = self._init_gradcam()

    def _init_gradcam(self) -> GradCAM:
        return GradCAM(
            self.model,
            self.target_layers,
            use_cuda=self.use_cuda,
            **self.gradcam_kwargs,
        )

    def _generate_heatmap(self) -> np.ndarray:
        heatmap = self.gradcam(
            self.input_tensor, self.target_categories, **self.heatmap_kwargs
        )
        return heatmap

    def _generate_overlay(self, heatmap: torch.Tensor) -> np.ndarray:
        return show_cam_on_image(self.input_tensor, heatmap, **self.showcam_kwargs)

    def display_single(self) -> None:
        """Displays the overlayed heatmap on the input image."""
        heatmap = self._generate_heatmap()
        heatmap = heatmap[0, :]
        overlay = self._generate_overlay(heatmap)
        image = (
            self.input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        )  # convert to numpy

        _fig, axes = plt.subplots(figsize=(20, 10), ncols=3)

        axes[0].imshow(image)
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].axis("off")

        plt.show()


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    # set PYTHONHASHSEED env var at fixed value
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id) -> None:
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def return_filepath(
    image_id: str,
    folder: Path,
    extension: str = "jpg",
) -> str:
    """Add a new column image_path to the train and test csv."""
    image_path = Path.joinpath(folder, f"{image_id}.{extension}").as_posix()
    return image_path


def get_sigmoid_softmax(config: Config) -> Union[nn.Sigmoid, nn.Softmax]:
    """Get the sigmoid or softmax function depending on loss function."""
    assert config.criterion.criterion in [
        "BCEWithLogitsLoss",
        "CrossEntropyLoss",
    ], "Criterion not supported"
    if config.criterion.criterion == "CrossEntropyLoss":
        return getattr(nn, "Softmax")(dim=1)
    return getattr(nn, "Sigmoid")()


class BinaryProbF1(Metric):
    pass


def pfbeta_torch(preds, labels, beta=1):
    if preds.ndim == 2:
        preds = preds[:, 1]
    preds = preds.clip(0, 1)

    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()

    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count

    if c_precision > 0 and c_recall > 0:
        return (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        ).item()
    else:
        return 0.0


def preprocess(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Preprocess data."""
    df[config.datamodule.dataset.image_col_name] = (
        df["patient_id"].astype(str) + "_" + df["image_id"].astype(str)
    )
    df[config.datamodule.dataset.image_path_col_name] = df[
        config.datamodule.dataset.image_col_name
    ].apply(
        lambda x: return_filepath(
            image_id=x,
            folder=config.datamodule.dataset.train_dir,
            extension=config.datamodule.dataset.image_extension,
        )
    )
    return df


def read_data_as_df(config: Config) -> pd.DataFrame:
    """Read data as a pandas dataframe."""
    df = pd.read_csv(config.datamodule.dataset.train_csv)
    return df


def create_folds(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Create a new column called "fold" and assign fold number to each row. Omit the use
    of train_test_split since the same result can be achieved by using
    (Stratified)KFold with n_splits=2."""
    cv = getattr(model_selection, config.datamodule.resample.resample_strategy)(
        **config.datamodule.resample.resample_params
    )

    group_by = config.datamodule.dataset.group_by
    stratify_by = config.datamodule.dataset.stratify_by
    stratify = df[stratify_by].values if stratify_by else None
    groups = df[group_by].values if group_by else None

    for _fold, (_train_idx, valid_idx) in enumerate(cv.split(df, stratify, groups)):
        df.loc[valid_idx, "fold"] = _fold + 1
    df["fold"] = df["fold"].astype(int)
    print(df.groupby(["fold", config.datamodule.dataset.target_col_name]).size())
    return df


def upsample_df(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    # Upsample cancer data
    # (from https://www.kaggle.com/code/awsaf49/rsna-bcd-efficientnet-tf-tpu-1vm-train)
    pos_df = df[df.cancer == 1].sample(frac=config.datamodule.upsample, replace=True)
    neg_df = df[df.cancer == 0]
    df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
    return df
