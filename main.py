import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib as p

from ww_config.paths import *
from log.log_handling import logger
from handler.pipeline_handler import Handler


@hydra.main(config_path=str(CONFIG_YAML_DIR), config_name="config", version_base=None)
def resolve_config(config: DictConfig):
  config_file = p.Path.joinpath(CONFIG_YAML_DIR, "config.yaml")
  logger.info(f"\nStarting the traing using this below configs ############ \n{OmegaConf.to_yaml(config)}\
  ############\
  \nconfig file used: {config_file}")

  return config


if __name__ == "__main__":
  config: DictConfig = resolve_config()
  Handler.run(config)
