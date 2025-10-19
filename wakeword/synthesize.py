import sys
sys.path.append("/home/xd/Documents/wake-word/vits")
sys.path.append("/home/xd/Documents/wake-word")


import soundfile as sf
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from vits import commons
from vits import utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from ww_config.config import *
from ww_config.paths import *
from base.base import BaseClass


class Synthesizer(BaseClass):
  def __init__(self):
    super().__init__()
    self.pull_pretrained_model()

  def pull_pretrained_model(self):
    self.pre_trained_model_local_path = hf_hub_download(repo_id=self.config.synthesizer.pre_trained_model_hf_repo,   # repo name
                                                        filename=self.config.synthesizer.pre_trained_model,   # File name in repo
                                                        repo_type="model")
    self.logger.info(f"pulled pre trained voice model to path: {self.pre_trained_model_local_path}")

  def get_text(self, text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

  def synthesize_ww(self):
    SYNTHESIZED_DIR.mkdir(exist_ok=True)
    SYNTHESIZED_WW_DIR.mkdir(exist_ok=True)

    hps = utils.get_hparams_from_file("vits/configs/vctk_base.json")

    net_g = SynthesizerTrn(len(symbols),
                           hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           n_speakers=hps.data.n_speakers,
                           **hps.model).cuda()    # TODO: support cpu inferencing
    _ = net_g.eval()
    _ = utils.load_checkpoint(self.pre_trained_model_local_path, net_g, None)
    stn_tst = self.get_text(self.config.wakeword.value, hps)

    # TODO: adjust the ranges to meet the requested sample count
    with tqdm(total=self.config.synthesizer.wake_word_sample_count, desc="Synthesizing wakeword samples") as pbar:
      self.logger.info(f"Saving synthesized wakeword samples at {SYNTHESIZED_WW_DIR}")
      for noise_scale in np.arange(0.1, 1.0, 0.1):
        for noise_scale_w in np.arange(0.1, 1.0, 0.1):
          with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([4]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=2)[0][0, 0].data.cpu().float().numpy()

          file_path = f"{SYNTHESIZED_WW_DIR}/vits_{noise_scale:.2f}_{noise_scale_w:.2f}.wav"
          sf.write(file_path, audio, hps.data.sampling_rate)

          pbar.update(1)


if __name__ == "__main__":

  Synthesizer().synthesize_ww()
