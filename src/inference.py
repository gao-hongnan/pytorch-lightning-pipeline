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
    model: Model,
    state_dict: collections.OrderedDict,
    test_loader: DataLoader,
    config,
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
    device = config.device
    model.to(device)
    model.eval()

    model.load_state_dict(state_dict)

    current_fold_probs = []

    for batch in tqdm(test_loader, position=0, leave=True):
        images = batch.to(device, non_blocking=True)
        test_logits = model(images)
        test_probs = get_sigmoid_softmax(config)(test_logits).cpu().numpy()
        current_fold_probs.append(test_probs)
    current_fold_probs = np.concatenate(current_fold_probs, axis=0)
    return current_fold_probs


@torch.inference_mode(mode=True)
def inference_all_folds(
    model: Model,
    state_dicts: List[collections.OrderedDict],
    test_loader: DataLoader,
    config,
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
    for _fold_num, state in enumerate(state_dicts):
        current_fold_probs = inference_one_fold(model, state, test_loader, config)
        all_folds_probs.append(current_fold_probs)
    mean_preds = np.mean(all_folds_probs, axis=0)
    return mean_preds
