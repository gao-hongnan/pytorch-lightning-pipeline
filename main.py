"""Controller for training pipeline."""
import logging
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from rich.pretty import pprint
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

from configs.base import Config
from src.datamodules.datamodule import RSNAUpsampleDataModule
from src.models.lightning_module import RSNALightningModel
from src.models.model import TimmModel
from src.utils.general import GradCamWrapper, create_folds, preprocess, read_data_as_df

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

    elif config.general.stage == "evaluate":
        print("Evaluate mode")
        dm.setup(stage="evaluate")

        # checkpoint = "/Users/gaohn/gao/pytorch-lightning-hydra/rsna/logs/lightning_logs/version_1/checkpoints/epoch=2-step=12.ckpt"
        checkpoint = "/Users/gaohn/gao/pytorch-lightning-pipeline/outputs/rsna/20230125_151003/lightning_logs/version_0/checkpoints/epoch=2-step=12.ckpt"
        # module = module.load_from_checkpoint(checkpoint)
        module.load_state_dict(torch.load(checkpoint)["state_dict"])
        valid_loader = dm.val_dataloader()
        # predictions = trainer.predict(
        #     module, dataloaders=valid_loader, ckpt_path=checkpoint
        # )
        predictions = trainer.predict(module, dataloaders=valid_loader)
        print(predictions)


def hydra_to_pydantic(config: DictConfig) -> Config:
    """Converts Hydra config to Pydantic config."""
    # use to_container to resolve
    config = OmegaConf.to_object(config)  # = to_container(config, resolve=True)
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

    # callbacks = instantiate(config.trainer.callbacks)
    # print(callbacks)

    # callbacks = instantiate(config.callbacks.callbacks)
    trainer = instantiate(config.trainer)
    config.trainer = trainer

    config = hydra_to_pydantic(config)
    pprint(config)

    run(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


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
