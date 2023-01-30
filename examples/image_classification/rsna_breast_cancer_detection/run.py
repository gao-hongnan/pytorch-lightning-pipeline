"""Controller for training pipeline."""
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from configs.base import Config
from examples.image_classification.rsna_breast_cancer_detection.datamodule import (
    RSNAUpsampleDataModule,
)
from examples.image_classification.rsna_breast_cancer_detection.lightning_module import (
    RSNALightningModel,
)
from src.models.model import TimmModel
from src.utils.general import GradCamWrapper, create_folds, preprocess, read_data_as_df


# pylint: disable=all
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
    trainer = pl.Trainer(**config.trainer.dict())

    if config.general.stage == "train":
        dm.setup(stage="train")
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
