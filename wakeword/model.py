from ww_config.config import *    # noqa
import librosa.display    # noqa
import torch

from schemas import ModelInitializerConfig
from ww_utils.patterns import Singleton


class ModelInitializer(metaclass=Singleton):
  def __init__(self, config: ModelInitializerConfig):
    self.config: ModelInitializerConfig = config

  def initialize(self):
    self.model = WakeWord(input_size=self.config.wakeword_config.input_size,
                          hidden_size=self.config.wakeword_config.hidden_size,
                          output_size=self.config.wakeword_config.output_size)
    self.dataloader = torch.utils.data.DataLoader(dataset=self.config.dataloader_config.data,
                                                  batch_size=self.config.dataloader_config.batch_size,
                                                  num_workers=self.config.dataloader_config.num_workers)
    self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                      lr=self.config.optimizer_config.lr)
    self.criterion = self.config.criterion_config.loss_function()


# ########## Model architecture
class WakeWord(torch.nn.Module):
  def __init__(self, input_size: int, hidden_size: int, output_size: int):
    super().__init__()
    self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)

  def forward(self, input_data):
    lstm_output, _ = self.lstm(input_data)
    output = torch.nn.functional.sigmoid(self.fc1(lstm_output[:, -1, :]))
    return output