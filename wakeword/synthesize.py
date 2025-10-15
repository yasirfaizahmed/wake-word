import soundfile as sf
import numpy as np
import torch

import commons    # noqa
import utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from ww_config.config import *


def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
      text_norm = commons.intersperse(text_norm, 0)
  text_norm = torch.LongTensor(text_norm)
  return text_norm


class Synthesizer():
  def __init__(self):
    pass

  def synthesize(self):
    hps = utils.get_hparams_from_file("vits/configs/vctk_base.json")

    net_g = SynthesizerTrn(len(symbols),
                           hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           n_speakers=hps.data.n_speakers,
                           **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint("/home/xd/Downloads/pretrained_vctk.pth", net_g, None)

    stn_tst = get_text(WAKE_WORD, hps)
    for noise_scale in np.arange(0.1, 1.0, 0.1):
      for noise_scale_w in np.arange(0.1, 1.0, 0.1):
        with torch.no_grad():
          x_tst = stn_tst.cuda().unsqueeze(0)
          x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
          sid = torch.LongTensor([4]).cuda()
          audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=2)[0][0, 0].data.cpu().float().numpy()
        sf.write(f"/home/xd/Documents/wake-word/.data/wake/vits_{noise_scale:.2f}_{noise_scale_w:.2f}.wav", audio, hps.data.sampling_rate)
        print(f"Saved file at /home/xd/Documents/wake-word/.data/wake/vits_{noise_scale:.2f}_{noise_scale_w:.2f}.wav")
