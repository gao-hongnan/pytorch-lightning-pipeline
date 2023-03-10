from typing import Optional, Iterable
from torch.utils.data import Sampler, Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn import model_selection

from src.datamodules.datamodule import ImageClassificationDataModule
from src.datamodules.dataset import ImageClassificationDataset
from src.utils.general import return_filepath
from configs.base import Config


def upsample_df(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    # Upsample cancer data
    # (from https://www.kaggle.com/code/awsaf49/rsna-bcd-efficientnet-tf-tpu-1vm-train)
    pos_df = df[df.cancer == 1].sample(frac=config.datamodule.upsample, replace=True)
    neg_df = df[df.cancer == 0]
    df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
    return df


def preprocess(
    df: pd.DataFrame, directory: str, extension: str, nested: bool, config: Config
) -> pd.DataFrame:
    """Preprocess data specific to RSNA.

    If nested, the images are stored as follows:
    - train_images
        - patient_id
            - image_id.png"""

    if nested:
        df[config.datamodule.dataset.image_col_name] = (
            df.patient_id.astype(str) + "/" + df.image_id.astype(str)
        )
    else:
        df[config.datamodule.dataset.image_col_name] = (
            df["patient_id"].astype(str) + "_" + df["image_id"].astype(str)
        )

    df[config.datamodule.dataset.image_path_col_name] = df[
        config.datamodule.dataset.image_col_name
    ].apply(
        lambda x: return_filepath(image_id=x, folder=directory, extension=extension)
    )
    return df


def create_folds(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Create a new column called "fold" and assign fold number to each row.
    Omit the use of train_test_split since the same result can be achieved by using
    (Stratified)KFold with n_splits=2.
    """
    cv = getattr(model_selection, config.datamodule.resample.resample_strategy)(
        **config.datamodule.resample.resample_params
    )

    # custom stratification, prevent data leakage
    num_bins = 5
    df["age_bin"] = pd.cut(df['age'].values.reshape(-1), bins=num_bins, labels=False)
    strat_cols = [
        'laterality',
        'view',
        'biopsy',
        'invasive',
        'BIRADS',
        'age_bin',
        'implant',
        'density',
        'machine_id',
        'difficult_negative_case',
        'cancer',
    ]

    df["stratify"] = ""
    for col in strat_cols:
        df['stratify'] += df[col].astype(str)

    group_by = config.datamodule.dataset.group_by
    stratify_by = config.datamodule.dataset.stratify_by
    stratify = df[stratify_by].values if stratify_by else None
    groups = df[group_by].values if group_by else None

    for _fold, (_train_idx, valid_idx) in enumerate(cv.split(df, stratify, groups)):
        df.loc[valid_idx, "fold"] = _fold + 1
    df["fold"] = df["fold"].astype(int)
    print(df.groupby(["fold", config.datamodule.dataset.target_col_name]).size())
    return df


class BalanceSampler(Sampler):
    """Ensures that each batch sees minority samples."""

    def __init__(self, dataset: Dataset, ratio: int = 8) -> None:
        self.r = ratio - 1
        self.dataset = dataset
        self.pos_index = np.where(dataset.df.cancer > 0)[0]
        self.neg_index = np.where(dataset.df.cancer == 0)[0]

        self.length = self.r * int(np.floor(len(self.neg_index) / self.r))

    def __iter__(self) -> Iterable[int]:
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[: self.length].reshape(-1, self.r)
        pos_index = np.random.choice(pos_index, self.length // self.r).reshape(-1, 1)

        index = np.concatenate([pos_index, neg_index], -1).reshape(-1)
        return iter(index)

    def __len__(self) -> int:
        return self.length


class RSNAUpsampleDataModule(ImageClassificationDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        """Assign train/val datasets for use in dataloaders.
        This method is called on every GPU in distributed training."""
        # FIXME: may need to change stage to have fit since lightning needs fit
        # for callbacks such as devicestatmonitor.
        print(f"Stage: {stage}")
        print(f"Using Fold Number {self.fold}")
        self.train_df = self.df_folds[self.df_folds["fold"] != self.fold].reset_index(
            drop=True
        )
        self.valid_df = self.df_folds[self.df_folds["fold"] == self.fold].reset_index(
            drop=True
        )
        self.oof_df = self.valid_df.copy()

        # upsample block
        if self.config.datamodule.upsample:
            print(f"Upsampling by {self.config.datamodule.upsample} times")
            self.train_df = upsample_df(self.train_df, self.config)

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


class RSNAUpsampleBalancedSamplerDataModule(RSNAUpsampleDataModule):
    """Upsample and Balanced Sampler DataModule."""

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        balance_sampler = BalanceSampler(self.train_dataset, ratio=8)
        self.config.datamodule.dataloader.train_loader.pop("shuffle", None)
        return DataLoader(
            self.train_dataset,
            sampler=balance_sampler,
            **self.config.datamodule.dataloader.train_loader,
        )

    def val_dataloader(self) -> DataLoader:
        """This is normal Sequential Sampler."""
        return DataLoader(
            self.valid_dataset, **self.config.datamodule.dataloader.valid_loader
        )
