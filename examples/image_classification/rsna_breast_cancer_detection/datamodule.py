from typing import Optional
from torch.utils.data import Sampler
import numpy as np

from src.datamodules.datamodule import ImageClassificationDataModule
from src.datamodules.dataset import ImageClassificationDataset
from src.utils.general import upsample_df


class BalanceSampler(Sampler):
    def __init__(self, dataset, ratio=8):
        self.r = ratio - 1
        self.dataset = dataset
        self.pos_index = np.where(dataset.df.cancer > 0)[0]
        self.neg_index = np.where(dataset.df.cancer == 0)[0]

        self.length = self.r * int(np.floor(len(self.neg_index) / self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[: self.length].reshape(-1, self.r)
        pos_index = np.random.choice(pos_index, self.length // self.r).reshape(-1, 1)

        index = np.concatenate([pos_index, neg_index], -1).reshape(-1)
        return iter(index)

    def __len__(self):
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
            print("Upsampling the data")
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
