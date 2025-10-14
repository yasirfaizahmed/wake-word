import librosa, numpy as np
import pathlib as p
from matplotlib import pyplot as plt
import soundfile as sf
from ww_config.config import *
import random
from scipy.signal import fftconvolve
import librosa.display
import os
import torch

from schemas import ModelInitializerConfig


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


class ModelInitializer():
  def __init__(self, config: ModelInitializerConfig):
    self.config: ModelInitializerConfig = config

  def build(self):
    self.model = WakeWord(input_size=self.config.wakeword_config.input_size,
                          hidden_size=self.config.wakeword_config.hidden_size,
                          output_size=self.config.wakeword_config.output_size)
    self.dataloader = torch.utils.data.DataLoader(dataset=self.config.dataloader_config.data,
                                                  batch_size=self.config.dataloader_config.batch_size,
                                                  num_workers=self.config.dataloader_config.num_workers)
    self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                      lr=self.config.optimizer_config.lr)
    self.criterion = self.config.criterion_config.loss_function()


def train():
  epochs = 25
  loss_per_epoch = []

  for epoch in range(epochs):
    train_loss_per_batch = 0
    model.train()
    for batch_count, (input_data, labels) in enumerate(dataloader):
      optimizer.zero_grad()
      outputs = model(input_data)

      loss = criterion(outputs, labels.view(-1, 1).float())
      loss.backward()
      optimizer.step()
      train_loss_per_batch += loss.item()
    loss_per_epoch.append(train_loss_per_batch/batch_count)
    print(f"Training loss per epoch: {train_loss_per_batch/batch_count}")


