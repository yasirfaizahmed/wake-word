import numpy as np
from huggingface_hub import hf_hub_download
import zipfile
import librosa
import torch
from tqdm import tqdm

from ww_config.paths import *
from ww_config.config import *


class DatasetBuilder():
  def __init__(self):
    self.setup_dataset_dir()
    self.noise_data: torch.Tensor = self.generate_noise()

  def setup_dataset_dir(self):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    WAKE_DIR.mkdir(parents=True, exist_ok=True)
    WAKE_CHOPPED.mkdir(parents=True, exist_ok=True)
    WAKE_AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    NON_WAKE_DIR.mkdir(parents=True, exist_ok=True)
    NON_WAKE_CHOPPED_DIR.mkdir(parents=True, exist_ok=True)

    NOISE_DATA_DIR.mkdir(parents=True, exist_ok=True)

  def pull_noise_from_HF(self) -> p.Path:
    # Example: download a single file from a repo
    file_path = hf_hub_download(repo_id=NOISE_HF_DATASET,   # repo name
                                filename=ZIP_FILE_NAME,   # File name in repo
                                repo_type="dataset")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
      zip_ref.extractall(NOISE_DATA_DIR)
    return NOISE_DATA_DIR

  def generate_noise(self) -> torch.Tensor:
    noise_data = []
    noise_data_dir: p.Path = self.pull_noise_from_HF()

    number_of_files = sum(1 for audio_file in noise_data_dir.iterdir() if audio_file.is_file())
    with tqdm(total=number_of_files, desc="Extracting MFCC features of noise samples") as pbar:
      for file in noise_data_dir.rglob("*.wav"):
        y, _ = librosa.load(file, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = mfcc.T

        # keeping the length of each sample same
        if len(mfcc) < MAX_LEN:
          pad = np.zeros((MAX_LEN - len(mfcc)), N_MFCC)
          mfcc = np.vstack([mfcc, pad])
        else:
          mfcc = mfcc[:MAX_LEN]

        noise_data.append(mfcc.T)

        pbar.update(1)

    return torch.tensor(noise_data, dtype=torch.float32)

  def generate_negative_wake_words(self) -> torch.Tensor:
    pass