"""Inference script for PyTorch Lightning Trainer and normal PyTorch pipeline.

NOTE:
    There's a lot of repeated code here. Consider using a design
    pattern to sort it out. An adapter like from_lightning, from_pytorch
    etc might be helpful.
"""
from typing import List, Optional
from tqdm.auto import tqdm
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.models.base import Model

from src.utils.general import get_sigmoid_softmax


@torch.inference_mode(mode=True)
def inference_one_fold(
    model: Model,
    checkpoint: str,
    test_loader: DataLoader,
    device: str,
    criterion: str = "CrossEntropyLoss",
) -> np.ndarray:
    """Inference the model on one fold."""
    model.to(device)

    # print(model.state_dict())

    state_dict = torch.load(checkpoint, map_location=device)
    model_weights = state_dict["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    # model_state_dict = state_dict["state_dict"]
    model.load_state_dict(model_weights, strict=True)
    model.eval()
    # model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
    # print(model.state_dict())

    # model.load_state_dict(state_dict["state_dict"], strict=False)

    probs = []
    for batch in tqdm(test_loader, position=0, leave=True):
        images = batch.to(device, non_blocking=True)
        logits = model(images)
        prob = get_sigmoid_softmax(criterion)(logits)
        prob = torch.tensor(prob, dtype=torch.float)
        prob = prob.cpu().numpy()
        probs.append(prob)
    probs = np.concatenate(probs, axis=0)
    return probs


@torch.inference_mode(mode=True)
def inference_one_fold_with_pytorch_lightning(
    model: Model,
    checkpoint: str,
    test_loader: DataLoader,
    trainer: pl.Trainer,
) -> np.ndarray:
    """Inference the model on one fold with PyTorch Lightning Interface.

    Args:
        model (Model): The model to be used for inference.
        checkpoint (str): The path to the checkpoint.
        test_loader (DataLoader): The dataloader for the test set.
        trainer (pl.Trainer): The trainer class from PyTorch Lightning.

    Returns:
        probs (np.ndarray): The predictions (probs) of the model.
    """
    # trainer.predict calls from predict_step in lightning_module.py
    # model.eval()

    prediction_dicts = trainer.predict(
        model, dataloaders=test_loader, ckpt_path=checkpoint
    )
    probs = []
    for prediction_dict in prediction_dicts:
        prob = prediction_dict["probs"]
        prob = torch.tensor(prob, dtype=torch.float)
        prob = prob.cpu().numpy()
        probs.append(prob)
    probs = np.concatenate(probs)
    return probs


@torch.inference_mode(mode=True)
def inference_all_folds(
    model: Model,
    checkpoints: List[str],
    test_loader: DataLoader,
    trainer: Optional[pl.Trainer] = None,
    device: Optional[str] = None,
    criterion: Optional[str] = None,
    adapter: str = "pytorch_lightning",
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
        if adapter == "pytorch_lightning":
            probs = inference_one_fold_with_pytorch_lightning(
                model=model,
                checkpoint=checkpoint,
                test_loader=test_loader,
                trainer=trainer,
            )
        elif adapter == "pytorch":
            probs = inference_one_fold(
                model=model,
                checkpoint=checkpoint,
                test_loader=test_loader,
                device=device,
                criterion=criterion,
            )
        all_folds_probs.append(probs)
    mean_probs = np.mean(all_folds_probs, axis=0)
    return mean_probs


class Inferencer:
    """Adapter class for inference."""
