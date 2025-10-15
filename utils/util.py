from ww_config.paths import *


# ######### Handy class to convert a dict to a object for config purpose
class ArgDict:
  def __init__(self, data: dict):
    for key, value in data.items():
      setattr(self, key, value)

  def __getattribute__(self, name):
    return getattr(self, name)


def create_dataset_dirs():
  # some dir creations
  if not RECORDINGS_PATH.exists():
    RECORDINGS_PATH.mkdir(parents=True, exist_ok=True)
  WAKE_DIR.mkdir(parents=True, exist_ok=True)
  WAKE_CHOPPED.mkdir(parents=True, exist_ok=True)
  WAKE_AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

  NON_WAKE_DIR.mkdir(parents=True, exist_ok=True)
  NON_WAKE_CHOPPED_DIR.mkdir(parents=True, exist_ok=True)