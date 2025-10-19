from omegaconf import DictConfig
from wakeword.synthesize import Synthesizer


class Handler():
  def __init__(self, config: DictConfig):
    self.config = config

  def run(self):
    Synthesizer()
