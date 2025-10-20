import sys
import soundfile as sf
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import zipfile
import typing
import os
import uuid

sys.path.append("/home/xd/Documents/wake-word/vits")
sys.path.append("/home/xd/Documents/wake-word")

from vits import commons
from vits import utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from ww_config.config import *
from ww_config.paths import *
from base.base import BaseClass
from ww_utils.audio_utils import *

import logging
logging.getLogger("vits").setLevel(logging.CRITICAL)
logging.getLogger("vits.utils").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)


class Synthesizer(BaseClass):
  def __init__(self):
    super().__init__()

    SYNTHESIZED_DIR.mkdir(exist_ok=True)
    SYNTHESIZED_WW_DIR.mkdir(exist_ok=True)
    SYNTHESIZED_SIMILAR_SOUNDING_DIR.mkdir(exist_ok=True)

    self._pull_pretrained_model_from_HF()
    self._pull_noise_dataset_from_HF()

  def _pull_pretrained_model_from_HF(self):
    self.pre_trained_model_local_path = hf_hub_download(repo_id=self.config.synthesizer.pre_trained_model_hf_repo,   # repo name
                                                        filename=self.config.synthesizer.pre_trained_model,   # File name in repo
                                                        repo_type="model")
    self.logger.info(f"pulled pre trained voice model to path: {self.pre_trained_model_local_path}")

  def _get_text(self, text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

  def _pull_noise_dataset_from_HF(self) -> p.Path:
    # Example: download a single file from a repo
    file_path = hf_hub_download(repo_id=self.config.synthesizer.noise_dataset_hf_repo,   # repo name
                                filename=self.config.synthesizer.noise_dataset_file,   # File name in repo
                                repo_type="dataset")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
      zip_ref.extractall(NOISE_DATA_DIR)
    self.logger.info(f"Extracted all noise to path: {NOISE_DATA_DIR}")

  def infer(self, text, noise_scale, noise_scale_w) -> typing.Tuple:
    hps = utils.get_hparams_from_file("vits/configs/vctk_base.json")

    net_g = SynthesizerTrn(len(symbols),
                           hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           n_speakers=hps.data.n_speakers,
                           **hps.model).cuda()    # TODO: support cpu inferencing
    _ = net_g.eval()
    _ = utils.load_checkpoint(self.pre_trained_model_local_path, net_g, None)
    stn_tst = self._get_text(text, hps)

    with torch.no_grad():
      x_tst = stn_tst.cuda().unsqueeze(0)
      x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
      sid = torch.LongTensor([4]).cuda()
      audio = net_g.infer(x_tst,
                          x_tst_lengths,
                          sid=sid,
                          noise_scale=noise_scale,
                          noise_scale_w=noise_scale_w,
                          length_scale=self.config.synthesizer.vocal_duration)[0][0, 0].data.cpu().float().numpy()
    return audio, hps

  def synthesize_ww(self):
    # TODO: adjust the ranges to meet the requested sample count
    count = self.config.synthesizer.synthetic_wake_word_sample_count
    with tqdm(total=count, desc=f"Synthesizing {count} wakeword samples") as pbar:
      incrementer = 0.2 if self.config.synthesizer.synthetic_wake_word_sample_count == 25 else 0.1
      self.logger.info(f"Saving synthesized wakeword samples at {SYNTHESIZED_WW_DIR}")
      for noise_scale in np.arange(0.1, 1.0, incrementer):
        for noise_scale_w in np.arange(0.1, 1.0, incrementer):
          audio, hps = self.infer(text=self.config.wakeword.value, noise_scale=noise_scale, noise_scale_w=noise_scale_w)

          file_path = f"{SYNTHESIZED_WW_DIR}/vits_ww_{noise_scale:.2f}_{noise_scale_w:.2f}.wav"
          sf.write(file_path, audio, hps.data.sampling_rate)

          pbar.update(1)

  def synthesize_nww(self):
    count = len(self.config.wakeword.similar_sounding_wake_words) * self.config.wakeword.similar_sounding_wake_words_multiplier
    with tqdm(total=count,
              desc=f"Synthesizing {count} similar sounding wake word samples") as pbar:
      self.logger.info(f"Saving synthesized similar sounding wakeword samples at {SYNTHESIZED_SIMILAR_SOUNDING_DIR}")
      for similar_sounding_word in self.config.wakeword.similar_sounding_wake_words:
        incrementer = 0.3 if self.config.wakeword.similar_sounding_wake_words_multiplier == 9 else 0.1
        for noise_scale in np.arange(0.1, 1.0, incrementer):
          for noise_scale_w in np.arange(0.1, 1.0, incrementer):
            audio, hps = self.infer(text=similar_sounding_word, noise_scale=noise_scale, noise_scale_w=noise_scale_w)

            file_path = f"{SYNTHESIZED_SIMILAR_SOUNDING_DIR}/vits_ssww_{similar_sounding_word}_{noise_scale:.2f}_{noise_scale_w:.2f}.wav"
            sf.write(file_path, audio, hps.data.sampling_rate)

            pbar.update(1)

  def _add_background_noise(self, files: typing.List[p.Path], mode: typing.Literal["wake", "non_wake"]):
    noise_files = list(NOISE_DATA_DIR.glob(".wav"))
    randomly_picked_noise_files = random.sample(noise_files, len(files))
    with tqdm(total=len(files), desc="Adding background noise to files") as pbar:
      for file, noise_file in zip(files, randomly_picked_noise_files):
        file_sig, _ = librosa.load(file, sr=self.config.audio.sample_rate)
        noise_sig, _ = librosa.load(noise_file, sr=self.config.audio.sample_rate)

        target_len = int(self.config.audio.sample_rate * self.config.synthesizer.total_duration)

        file_sig = self._fix_length(file_sig, target_len)
        noise_sig = self._fix_length(noise_sig, target_len)

        # noise loudness — e.g., -10dB relative to main signal
        noise_ratio = 0.2  # smaller = quieter noise
        mixed = file_sig + noise_ratio * noise_sig

        # normalize final mix
        mixed = mixed / np.max(np.abs(mixed))

        if mode == "wake":
          out_path = WAKE_DIR.joinpath(f"{uuid.uuid4()}.wav")
          sf.write(out_path, mixed, self.config.audio.sample_rate)

        pbar.update(1)

  def _fix_length(self, signal, target_len):
    if len(signal) > target_len:
        return signal[:target_len]
    elif len(signal) < target_len:
        return np.pad(signal, (0, target_len - len(signal)))
    return signal

  def _augment(self):
    # load backgrounds into memory for speed
    bg_files = [os.path.join(NOISE_DATA_DIR, f) for f in os.listdir(NOISE_DATA_DIR) if f.endswith(".wav")]
    bg_sigs = []
    for p in bg_files:
      bg, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
      bg_sigs.append(bg)

    wake_files = [os.path.join(SYNTHESIZED_WW_DIR, f) for f in os.listdir(SYNTHESIZED_WW_DIR) if f.endswith(".wav")]

    file_count = 0
    for wf in wake_files:
      y, _ = librosa.load(wf, sr=SAMPLE_RATE, mono=True)
      # optional: trim silence
      y, _ = librosa.effects.trim(y, top_db=30)
      for i in range(N_AUG_PER_FILE):
        y_aug = y.copy()
        # order: small random transforms
        if random.random() < 0.6:
            y_aug = time_stretch(y_aug, 0.92, 1.08)
        if random.random() < 0.6:
            y_aug = pitch_shift(y_aug, -1.5, 1.5)
        if random.random() < 0.7:
            y_aug = time_shift(y_aug, max_shift_seconds=0.12)
        if random.random() < 0.8:
            y_aug = random_gain(y_aug, -6, 6)
        # if random.random() < 0.7 and len(bg_sigs) > 0:
        #     bg = random.choice(bg_sigs)
        #     snr = random.uniform(0, 20)  # 0 dB to 20 dB
        #     y_aug = add_background(y_aug, bg, snr_db=snr)
        if random.random() < 0.3:
            y_aug = add_reverb(y_aug, reverb_scale=random.uniform(0.05, 0.35))
        y_aug = random_filter(y_aug)
        # normalize to avoid clipping
        max_abs = np.max(np.abs(y_aug)) + 1e-9
        if max_abs > 1.0:
            y_aug = y_aug / max_abs * 0.99
        out_path = os.path.join(WAKE_AUGMENTED_DIR, f"aug_{file_count}.wav")
        sf.write(out_path, y_aug, SAMPLE_RATE)
        file_count += 1

    print(f"Created {file_count} augmented files in {WAKE_AUGMENTED_DIR}")


if __name__ == "__main__":
  Synthesizer().synthesize_nww()
