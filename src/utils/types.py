from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torchvision
import albumentations


TransformTypes = Optional[Union[albumentations.Compose, torchvision.transforms.Compose]]
BatchTensor = Tuple[torch.Tensor, torch.Tensor]

StepOutput = Union[torch.Tensor, Dict[str, Any]]
EpochOutput = List[StepOutput]
