import hydra
from omegaconf import DictConfig, OmegaConf

from log.log_handling import logger
from ww_config.paths import *
from handler.pipeline_handler import Handler


@hydra.main(config_path=str(CONFIG_YAML_DIR), config_name="config", version_base=None)
def main(config: DictConfig):
  config_file = p.Path.joinpath(CONFIG_YAML_DIR, "config.yaml")
  logger.info(f"\nStarting the traing using this below configs ############ \n{OmegaConf.to_yaml(config)}\
  ############\
  \nconfig file used: {config_file}")


if __name__ == "__main__":
  Handler().run()
