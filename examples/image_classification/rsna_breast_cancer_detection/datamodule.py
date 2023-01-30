from src.datamodules.datamodule import ImageClassificationDataModule
from src.utils.general import upsample_df


class RSNAUpsampleDataModule(ImageClassificationDataModule):
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
