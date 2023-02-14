"""Lightning module for RSNA Breast Cancer Detection."""
from typing import List, Union

import torch

from src.models.lightning_module import ImageClassificationLightningModel
from src.metrics.pf1 import pfbeta_torch
from src.utils.types import BatchTensor, EpochOutput, StepOutput


class RSNALightningModel(ImageClassificationLightningModel):
    """RSNA Lightning Module, added pf1."""

    def _shared_step(self, batch: BatchTensor, stage: str) -> StepOutput:
        """Shared step for train and validation step."""
        assert stage in ["train", "valid"], "stage must be either train or valid"

        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        self.log(f"{stage}_loss", loss)

        probs = self.sigmoid_or_softmax(logits)

        # newly added
        pf1 = pfbeta_torch(probs, targets, beta=1)
        # print(f"{stage}_pf1: {pf1}")
        self.log(f"{stage}_pf1", pf1)

        self.metrics[f"{stage}_metrics"](probs, targets)
        self.log_dict(
            self.metrics[f"{stage}_metrics"],
            on_step=True,  # whether to log on N steps
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "targets": targets, "logits": logits, "probs": probs}

    def _shared_epoch_end(
        self, outputs: Union[EpochOutput, List[EpochOutput]], stage: str
    ) -> None:
        """Shared epoch end for train and validation epoch end."""
        assert stage in ["train", "valid"], "stage must be either train or valid"
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{stage}_loss", loss)

        probs = torch.cat([x["probs"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])

        self.save_oof_predictions(probs, "probs")
        self.save_oof_predictions(logits, "logits")
        self.save_oof_predictions(targets, "targets")

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
