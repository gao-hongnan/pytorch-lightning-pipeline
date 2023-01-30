"""Controller for training pipeline."""
import logging
import warnings
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from configs.base import Config
from examples.image_classification.rsna_breast_cancer_detection.run import run

warnings.filterwarnings(action="ignore", category=UserWarning)

logger: logging.Logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> Config:
    """Converts Hydra config to Pydantic config."""
    # use to_container to resolve
    config = OmegaConf.to_object(config)  # = to_container(config, resolve=True)
    return Config(**config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry to training pipeline."""
    # logger.info(f"Config representation:\n{OmegaConf.to_yaml(config)}")

    output_dir = HydraConfig.get().runtime.output_dir

    logger.info(f"Output dir: {output_dir}")

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    transforms = instantiate(config.datamodule.transforms)
    config.datamodule.transforms = transforms

    metrics = instantiate(config.metrics.metrics)
    config.metrics.metrics = metrics

    trainer = instantiate(config.trainer)
    config.trainer = trainer

    config = hydra_to_pydantic(config)
    # pretty print config
    pprint(config)

    run(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
