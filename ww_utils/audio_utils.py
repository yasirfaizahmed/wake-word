import numpy as np
import random
import librosa
from scipy.signal import fftconvolve

from ww_config.config import *


def random_gain(y, min_db=-6, max_db=6):
  db = random.uniform(min_db, max_db)
  gain = 10.0 ** (db / 20.0)
  return y * gain


def add_background(y, bg, snr_db):
  # adjust background to desired snr relative to signal RMS
  rms_signal = np.sqrt(np.mean(y**2)) + 1e-9
  rms_bg = np.sqrt(np.mean(bg**2)) + 1e-9
  # compute linear scaling for bg
  snr_linear = 10 ** (snr_db / 20.0)
  scale = (rms_signal / (snr_linear * rms_bg))
  bg_scaled = bg * scale
  # trim or pad bg to match length
  if len(bg_scaled) >= len(y):
    start = random.randint(0, len(bg_scaled) - len(y))
    bg_segment = bg_scaled[start:start + len(y)]
  else:
    # repeat bg to fill
    repeats = int(np.ceil(len(y) / len(bg_scaled)))
    bg_tiled = np.tile(bg_scaled, repeats)[:len(y)]
    bg_segment = bg_tiled
  return y + bg_segment


def time_stretch(y, low=0.9, high=1.1):
  rate = random.uniform(low, high)
  # librosa time_stretch expects magnitude STFT -> use effects.time_stretch on waveform via resampling workaround:
  try:
    y_stretched = librosa.effects.time_stretch(y, rate)
  except Exception:
    # fallback: resample (this will change pitch)
    # new_len = int(len(y) / rate)
    y_stretched = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE * rate))
    y_stretched = librosa.resample(y_stretched, orig_sr=int(SAMPLE_RATE * rate), target_sr=SAMPLE_RATE)
  return y_stretched


def pitch_shift(y, semitones_low=-2, semitones_high=2):
  n_steps = random.uniform(semitones_low, semitones_high)
  return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)


def time_shift(y, max_shift_seconds=0.15):
  max_shift = int(max_shift_seconds * SAMPLE_RATE)
  shift = random.randint(-max_shift, max_shift)
  if shift == 0:
    return y
  if shift > 0:
    return np.pad(y, (shift, 0), mode='constant')[:len(y)]
  else:
    return np.pad(y, (0, -shift), mode='constant')[-shift:len(y) - shift]


def add_reverb(y, reverb_scale=0.3):
  # simple exponential impulse response
  ir_len = int(0.03 * SAMPLE_RATE)  # 30 ms IR
  t = np.arange(ir_len) / SAMPLE_RATE
  decay = np.exp(-t * (1.0 + random.random() * 10.0))  # random decay rate
  ir = decay
  ir = ir / np.sum(np.abs(ir))
  y_rev = fftconvolve(y, ir)[:len(y)]
  return y * (1 - reverb_scale) + y_rev * reverb_scale


def random_filter(y):
  # light low-pass / high-pass via simple FFT mask
  # For simplicity use librosa.effects.preemphasis as a proxy
  if random.random() < 0.2:
    return librosa.effects.preemphasis(y)
  return y