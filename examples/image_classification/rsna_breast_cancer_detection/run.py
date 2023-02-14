"""Controller for training pipeline."""
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

import pandas as pd
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
from src.models.model import TimmModel, TimmModelWithGeM
from src.utils.general import (
    GradCamWrapper,
    create_folds,
    preprocess,
    read_data_as_df,
    read_experiments_as_df_by_id,
    seed_all,
)
from rich import print

from src.metrics.pf1 import pfbeta_torch, optimize_thresholds
from src.inference import inference_all_folds

# pylint: disable=all
def run(config: Config) -> None:
    """Run the experiment."""
    pl.seed_everything(config.general.seed, workers=True)
    # seed_all(config.general.seed)

    fold = config.datamodule.fold

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
    dm = RSNAUpsampleDataModule(config, df_folds, fold, test_df)

    # model = TimmModel(config)
    model = TimmModelWithGeM(config)
    model.model_summary()

    module = RSNALightningModel(config, model)
    trainer = pl.Trainer(**config.trainer.dict())

    if config.general.dataset_stage == "train":
        dm.setup(stage="fit")
        train_dataloader = dm.train_dataloader()
        valid_dataloader = dm.val_dataloader()
        # for OneCycleLR
        print(f"Dataloader length: {len(train_dataloader)}")
        trainer.fit(module, train_dataloader, valid_dataloader)

    elif config.general.dataset_stage == "evaluate":
        # python main.py --config-name rsna general.stage=evaluate
        print("Evaluate mode")
        dm.setup(stage="evaluate")
        df_oof = df_folds.copy()

        experiment_df_path = config.general.experiment_df_path
        experiment_id = config.general.experiment_id
        experiment_df = read_experiments_as_df_by_id(experiment_df_path, experiment_id)

        target_checkpoints = experiment_df["oof_targets"].values
        prob_checkpoints = experiment_df["oof_probs"].values

        oof_targets, oof_probs = [], []
        for fold, (target_checkpoint, prob_checkpoint) in enumerate(
            zip(target_checkpoints, prob_checkpoints)
        ):
            fold = fold + 1
            targets = torch.load(target_checkpoint, map_location=torch.device("cpu"))
            probs = torch.load(prob_checkpoint, map_location=torch.device("cpu"))
            oof_targets.append(targets)
            oof_probs.append(probs)
            for metric_name, metric in config.metrics.metrics.items():
                metric_value = metric(probs, targets)
                print(f"{metric_name}: {metric_value}")

            df_oof.loc[df_oof["fold"] == fold, "oof_targets"] = targets.numpy()
            df_oof.loc[
                df_oof["fold"] == fold,
                [
                    f"class_{str(c)}_oof_probs"
                    for c in range(config.general.num_classes)
                ],
            ] = probs.numpy()

        oof_targets = torch.cat(oof_targets)
        oof_probs = torch.cat(oof_probs)
        for metric_name, metric in config.metrics.metrics.items():
            metric_value = metric(oof_probs, oof_targets)
            print(f"OOF {metric_name}: {metric_value}")

        raw_pf1 = pfbeta_torch(oof_probs, oof_targets, beta=1.0)
        print(f"OOF raw_pf1: {raw_pf1} without threshold")

        binarized_pf1, threshold = optimize_thresholds(oof_probs, oof_targets)
        print(f"OOF binarized_pf1: {binarized_pf1} with threshold: {threshold}")

        ### hardcoded only for rsna ###
        df_oof["prediction_id"] = (
            df_oof["patient_id"].astype(str) + "_" + df_oof["laterality"].astype(str)
        )
        df_oof = df_oof.groupby('prediction_id').max()  # .mean() #
        df_oof = df_oof.sort_index()
        df_oof = df_oof[df_oof["fold"] == 1]
        oof_probs = torch.from_numpy(
            df_oof[
                [f"class_{str(c)}_oof_probs" for c in range(config.general.num_classes)]
            ].values
        )
        oof_targets = torch.from_numpy(df_oof["oof_targets"].values)
        raw_pf1 = pfbeta_torch(oof_probs, oof_targets.flatten(), beta=1.0)
        print(f"OOF raw_pf1: {raw_pf1} without threshold")
        binarized_pf1, threshold = optimize_thresholds(oof_probs, oof_targets.flatten())
        print(f"OOF binarized_pf1: {binarized_pf1} with threshold: {threshold}")

    elif config.general.dataset_stage == "test":
        # python main.py --config-name rsna general.stage=test model.model_name=tf_efficientnetv2_s datamodule.transforms.image_size=512 general.device=mps
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()

        experiment_df_path = config.general.experiment_df_path
        experiment_id = config.general.experiment_id
        experiment_df = read_experiments_as_df_by_id(experiment_df_path, experiment_id)

        checkpoints = experiment_df["weight_paths"].values

        print(f"Checkpoints: {checkpoints}")

        predictions = inference_all_folds(module, checkpoints, test_loader, trainer)
        print(predictions)

    elif config.general.dataset_stage == "gradcam":
        dm.setup(stage="train")
        checkpoint = "artifacts/rsna/fold1_epoch=5-valid_multiclass_auroc=0.696480.ckpt"
        # module = module.load_from_checkpoint(checkpoint)
        module.load_state_dict(torch.load(checkpoint)["state_dict"])
        gradcam_loader = dm.gradcam_dataloader()

        originals, inputs, labels, image_ids = next(iter(gradcam_loader))
        originals = originals.cpu().detach().numpy() / 255.0

        # inputs = config.datamodule.transforms.valid_transforms(inputs)

        # target_layers = [module.model.backbone.layer4[-1]] # resnet
        target_layers = [model.backbone.conv_head]  # tf_efficientnetv2_s
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
