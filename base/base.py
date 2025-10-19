from omegaconf import DictConfig, OmegaConf

from log.log_handling import logger
from ww_config.paths import *


class BaseClass():
  def __init__(self):
    self.logger = logger
    self.config: DictConfig = OmegaConf.load(CONFIG_FILE)
