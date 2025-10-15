import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib as p

from ww_config.paths import *
from wakeword.dataset import DatasetBuilder
from log.log_handling import logger


@hydra.main(config_path=str(CONFIG_YAML_DIR), config_name="config", version_base=None)
def resolve_config(config: DictConfig):
  config_file = p.Path.joinpath(CONFIG_YAML_DIR, "config.yaml")
  logger.info(f"\nStarting the traing using this below configs {OmegaConf.to_yaml(cfg)}\
  \nconfig file used: {config_file}")

  return config


if __name__ == "__main__":
  config: DictConfig = resolve_config()
  dataset_builder = DatasetBuilder()
