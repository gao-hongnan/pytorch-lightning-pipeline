import collections
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm


@torch.inference_mode(mode=True)
def inference_one_fold(
    model,
    checkpoint: str,
    test_loader: DataLoader,
    trainer,
) -> np.ndarray:
    """Inference the model on one fold.

    Args:
        model (Model): The model to be used for inference.
            Note that pretrained should be set to False.
        state_dict (collections.OrderedDict): The state dict of the model.
        test_loader (DataLoader): The dataloader for the test set.

    Returns:
        test_probs (np.ndarray): The predictions of the model.
    """
    # from predict_step
    prediction_dict = trainer.predict(
        model, dataloaders=test_loader, ckpt_path=checkpoint
    )
    probs = prediction_dict[0]["probs"].detach().cpu().numpy()
    return probs


@torch.inference_mode(mode=True)
def inference_all_folds(
    model,
    checkpoints: List[str],
    test_loader: DataLoader,
    trainer,
) -> np.ndarray:
    """Inference the model on all K folds.

    Args:
        model (Model): The model to be used for inference.
            Note that pretrained should be set to False.
        state_dicts (List[collections.OrderedDict]): The state dicts of the models.
            Generally, K Fold means K state dicts.
        test_loader (DataLoader): The dataloader for the test set.

    Returns:
        mean_preds (np.ndarray): The mean of the predictions of all folds.
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
