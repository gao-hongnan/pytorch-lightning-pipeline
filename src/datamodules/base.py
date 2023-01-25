"""Dataset Interface with config constructor signature."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.utils.data import Dataset

from configs.base import Config


class AbstractDataset(ABC, Dataset):
    """A sample template for PyTorch Dataset."""

    def __init__(self, config: Config) -> None:
        """Constructor for the dataset class."""
        super().__init__()
        self.config = config

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset."""

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """Implements the getitem method."""
