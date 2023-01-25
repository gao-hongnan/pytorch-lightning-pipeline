from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection

from configs.base import Config
from src.models.model import TimmModel
from src.utils.general import pfbeta_torch
from src.utils.types import BatchTensor, EpochOutput


class ImageClassificationLightningModel(pl.LightningModule):
    """Lightning model class."""

    # TODO: add abstraction type hint
    # TODO: even though the constructor does not take in
    # optimizer, metrics etc, it is still not really violating
    # dependency inversion principle since the constructor
    # takes in config, which service locates the correct components.
    def __init__(self, config: Config, model: TimmModel) -> None:
        super().__init__()
        self.config = config
        self.config_dict = self.config.dict()
        self.model = model
        self.criterion = self._get_criterion()
        self.metrics = self._get_metrics()
        self.sigmoid_or_softmax = self._get_sigmoid_softmax()
        self.save_hyperparameters(ignore=["model", "config", "config_dict"])

    def _get_sigmoid_softmax(self) -> Union[nn.Sigmoid, nn.Softmax]:
        """Get the sigmoid or softmax function depending on loss function."""
        assert self.config.criterion.criterion in [
            "BCEWithLogitsLoss",
            "CrossEntropyLoss",
        ], "Criterion not supported"
        if self.config.criterion.criterion == "CrossEntropyLoss":
            return getattr(nn, "Softmax")(dim=1)
        return getattr(nn, "Sigmoid")()

    def _get_criterion(self) -> nn.Module:
        """Get loss function."""
        return getattr(nn, self.config.criterion.criterion)(
            **self.config.criterion.criterion_kwargs
        )

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer],
        List[torch.optim.lr_scheduler._LRScheduler],  # pylint: disable=protected-access
    ]:
        optimizer = getattr(torch.optim, self.config.optimizer.optimizer)(
            self.model.parameters(), **self.config.optimizer.optimizer_kwargs
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.config.scheduler.scheduler)(
            optimizer, **self.config.scheduler.scheduler_kwargs
        )
        return [optimizer], [scheduler]

    def _get_metrics(self) -> nn.ModuleDict:
        """Get metrics."""
        metrics_collection = MetricCollection(self.config.metrics.metrics)
        return nn.ModuleDict(
            {
                "train_metrics": metrics_collection.clone(prefix="train_"),
                "valid_metrics": metrics_collection.clone(prefix="valid_"),
            }
        )

    def get_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get penultimate layer embeddings."""
        return self.model.forward_features(inputs)

    # pylint: disable=arguments-differ, unused-argument
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward is needed in this module if you want to do self(inputs) instead
        of self.model(inputs)."""
        features = self.model.forward_features(inputs)
        logits = self.model.forward_head(features)
        return logits

    # TODO: unsure why batch_idx is in the signature but unused in example
    def training_step(self, batch: BatchTensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: BatchTensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "valid")

    def predict_step(self, batch: BatchTensor, batch_idx: int) -> torch.Tensor:
        """Predict step. Try-except block is to handle the case where
        I want to run inference on validation set, which has targets.

        This is the step function that is called when you do `trainer.predict`
        and by calls `forward`.
        """
        try:
            inputs, targets = batch
            logits = self(inputs)
            probs = self.sigmoid_or_softmax(logits)
            return probs, targets
        except ValueError:
            inputs = batch[0]
            logits = self(inputs)
            probs = self.sigmoid_or_softmax(logits)
            return probs

    def _shared_step(self, batch: BatchTensor, stage: str) -> torch.Tensor:
        """Shared step for train and validation step."""
        assert stage in ["train", "valid"], "stage must be either train or valid"

        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log(f"{stage}_loss", loss)

        probs = self.sigmoid_or_softmax(logits)

        self.metrics[f"{stage}_metrics"](probs, targets)
        self.log_dict(
            self.metrics[f"{stage}_metrics"],
            on_step=True,  # whether to log on N steps
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "probs": probs, "targets": targets, "logits": logits}

    def training_epoch_end(self, outputs: EpochOutput) -> None:
        """See source code for more info."""

    def validation_epoch_end(self, outputs: EpochOutput) -> None:
        """Good to use for logging validation metrics that are not
        calculated based on average. For example, if you want to log
        pf1, it is different from accumulating pf1 from each batch
        and then averaging it. Instead, you want to accumulate
        them and then calculate pf1 on the accumulated values."""
        return self._shared_epoch_end(outputs, "valid")

    def _shared_epoch_end(self, outputs: EpochOutput, stage: str) -> None:
        """Shared epoch end for train and validation epoch end."""
        assert stage in ["train", "valid"], "stage must be either train or valid"
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{stage}_loss", loss)

        probs = torch.cat([x["probs"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])

        pf1 = pfbeta_torch(probs, targets, beta=1)
        self.log(f"{stage}_pf1", pf1)

        self.metrics[f"{stage}_metrics"](probs, targets)
        self.log_dict(
            self.metrics[f"{stage}_metrics"],
            on_step=False,  # whether to log on N steps
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "probs": probs, "targets": targets, "logits": logits}


class RSNALightningModel(ImageClassificationLightningModel):
    def _shared_step(self, batch: BatchTensor, stage: str) -> torch.Tensor:
        """Shared step for train and validation step."""
        assert stage in ["train", "valid"], "stage must be either train or valid"

        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log(f"{stage}_loss", loss)

        probs = self.sigmoid_or_softmax(logits)

        pf1 = pfbeta_torch(probs, targets, beta=1)
        print(f"{stage}_pf1: {pf1}")
        self.log(f"{stage}_pf1", pf1)

        self.metrics[f"{stage}_metrics"](probs, targets)
        self.log_dict(
            self.metrics[f"{stage}_metrics"],
            on_step=True,  # whether to log on N steps
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "probs": probs, "targets": targets, "logits": logits}
