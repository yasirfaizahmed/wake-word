import pathlib as p


# ---------- CONFIG ----------
WAKE = "wake"
NON_WAKE = "non-wake"
WAKE_AUGMENTED = "wake-augmented"

PROJECT_PATH = p.Path(__file__).parent.parent
RECORDINGS_PATH = p.Path.joinpath(PROJECT_PATH, ".data")
WAKE_DIR = p.Path.joinpath(RECORDINGS_PATH, WAKE)
NON_WAKE_DIR = p.Path.joinpath(RECORDINGS_PATH, NON_WAKE)

SAMPLE_RATE = 16000   # per second
NUM_RECORDINGS = 3
HARDWARE = "0,0"
DURATION = 3 
SAMPLE_RATE = 44100
