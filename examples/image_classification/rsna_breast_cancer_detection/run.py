"""Controller for training pipeline."""
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

from configs.base import Config
from examples.image_classification.rsna_breast_cancer_detection.datamodule import (
    RSNAUpsampleDataModule,
)
from examples.image_classification.rsna_breast_cancer_detection.lightning_module import (
    RSNALightningModel,
)
from src.models.model import TimmModel
from src.utils.general import GradCamWrapper, create_folds, preprocess, read_data_as_df

from src.metrics.pf1 import pfbeta_torch, optimize_thresholds
from src.inference import inference_all_folds

# pylint: disable=all
def run(config: Config) -> None:
    """Run the experiment."""

    pl.seed_everything(config.general.seed)

    train_file = config.datamodule.dataset.train_csv
    test_file = config.datamodule.dataset.test_csv
    df = read_data_as_df(train_file)
    df = preprocess(
        df,
        directory=config.datamodule.dataset.train_dir,
        extension=config.datamodule.dataset.image_extension,
        config=config,
    )

    df_folds = create_folds(df, config)
    print(df_folds.head())

    test_df = read_data_as_df(test_file)
    test_df = preprocess(
        test_df,
        directory=config.datamodule.dataset.test_dir,
        extension=config.datamodule.dataset.image_extension,
        config=config,
    )
    print(test_df.head())

    # dm = RSNADataModule(config, df_folds)
    dm = RSNAUpsampleDataModule(config, df_folds, test_df)
    dm.prepare_data()

    model = TimmModel(config)

    module = RSNALightningModel(config, model)
    trainer = pl.Trainer(**config.trainer.dict())

    if config.general.stage == "train":
        dm.setup(stage="train")
        # for OneCycleLR
        print(f"Dataloader length: {len(dm.train_dataloader())}")
        trainer.fit(module, datamodule=dm)

    elif config.general.stage == "evaluate":
        # python main.py --config-name rsna general.stage=evaluate
        print("Evaluate mode")
        dm.setup(stage="evaluate")

        target_checkpoints = [
            "artifacts/rsna/fold_1_epoch_5_targets.pt",
            "artifacts/rsna/fold_2_epoch_3_targets.pt",
            "artifacts/rsna/fold_3_epoch_4_targets.pt",
            "artifacts/rsna/fold_4_epoch_5_targets.pt",
        ]
        prob_checkpoints = [
            "artifacts/rsna/fold_1_epoch_5_probs.pt",
            "artifacts/rsna/fold_2_epoch_3_probs.pt",
            "artifacts/rsna/fold_3_epoch_4_probs.pt",
            "artifacts/rsna/fold_4_epoch_5_probs.pt",
        ]
        oof_targets, oof_probs = [], []
        for target_checkpoint, prob_checkpoint in zip(
            target_checkpoints, prob_checkpoints
        ):
            targets = torch.load(target_checkpoint, map_location=torch.device("cpu"))
            probs = torch.load(prob_checkpoint, map_location=torch.device("cpu"))
            oof_targets.append(targets)
            oof_probs.append(probs)
            for metric_name, metric in config.metrics.metrics.items():
                metric_value = metric(probs, targets)
                print(f"{metric_name}: {metric_value}")

        oof_targets = torch.cat(oof_targets)
        oof_probs = torch.cat(oof_probs)
        for metric_name, metric in config.metrics.metrics.items():
            metric_value = metric(oof_probs, oof_targets)
            print(f"OOF {metric_name}: {metric_value}")

        raw_pf1 = pfbeta_torch(oof_probs, oof_targets, beta=1.0)
        print(f"OOF raw_pf1: {raw_pf1}")

        binarized_pf1, threshold = optimize_thresholds(oof_probs, oof_targets)
        print(f"OOF binarized_pf1: {binarized_pf1} with threshold: {threshold}")

    elif config.general.stage == "test":
        # python main.py --config-name rsna general.stage=test model.model_name=tf_efficientnetv2_s datamodule.transforms.image_size=512 general.device=cpu
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()
        checkpoints = [
            "/kaggle/input/rsna-tf-efficientnetv2-s-size512/fold1_epoch5-valid_multiclass_auroc0.696480.ckpt",
            "/kaggle/input/rsna-tf-efficientnetv2-s-size512/fold2_epoch3-valid_multiclass_auroc0.691854.ckpt",
            "/kaggle/input/rsna-tf-efficientnetv2-s-size512/fold3_epoch4-valid_multiclass_auroc0.685808.ckpt",
            "/kaggle/input/rsna-tf-efficientnetv2-s-size512/fold4_epoch5-valid_multiclass_auroc0.676737.ckpt",
        ]

        predictions = inference_all_folds(module, checkpoints, test_loader, trainer)
        print(predictions)

    elif config.general.stage == "gradcam":
        dm.setup(stage="train")
        checkpoint = "artifacts/rsna/fold1_epoch=5-valid_multiclass_auroc=0.696480.ckpt"
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
