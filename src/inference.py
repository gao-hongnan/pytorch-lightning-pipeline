from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.models.lightning_module import ImageClassificationLightningModel


@torch.inference_mode(mode=True)
def inference_one_fold(
    model: ImageClassificationLightningModel,
    checkpoint: str,
    test_loader: DataLoader,
    trainer: pl.Trainer,
) -> np.ndarray:
    """Inference the model on one fold with PyTorch Lightning Interface.

    Args:
        model (ImageClassificationLightningModel): The model to be used for inference.
        checkpoint (str): The path to the checkpoint.
        test_loader (DataLoader): The dataloader for the test set.
        trainer (pl.Trainer): The trainer class from PyTorch Lightning.

    Returns:
        probs (np.ndarray): The predictions (probs) of the model.
    """
    # trainer.predict calls from predict_step in lightning_module.py
    prediction_dict = trainer.predict(
        model, dataloaders=test_loader, ckpt_path=checkpoint
    )
    probs = prediction_dict[0]["probs"].detach().cpu().numpy()
    return probs


@torch.inference_mode(mode=True)
def inference_all_folds(
    model: ImageClassificationLightningModel,
    checkpoints: List[str],
    test_loader: DataLoader,
    trainer: pl.Trainer,
) -> np.ndarray:
    """Inference the model on all K folds.

    Currently does not support weighted average.

    Args:
        checkpoints (List[str]): The list of paths to the checkpoints.

    Returns:
        mean_probs (np.ndarray): The mean of the predictions of all folds.
    """
    all_folds_probs = []
    for _fold_num, checkpoint in enumerate(checkpoints):
        print(f"Predicting fold {_fold_num}")
        probs = inference_one_fold(
            model=model, checkpoint=checkpoint, test_loader=test_loader, trainer=trainer
        )
        all_folds_probs.append(probs)
    mean_probs = np.mean(all_folds_probs, axis=0)
    return mean_probs
