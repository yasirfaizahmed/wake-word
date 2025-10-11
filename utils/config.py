import pathlib as p


# ---------- CONFIG ----------
WAKE = "wake"
NON_WAKE = "non-wake"
WAKE_AUGMENTED = "wake-augmented"

PROJECT_PATH = p.Path(__file__).parent.parent
RECORDINGS_PATH = p.Path.joinpath(PROJECT_PATH, ".data")
WAKE_DIR = p.Path.joinpath(RECORDINGS_PATH, WAKE)
WAKE_CHOPPED = p.Path.joinpath(RECORDINGS_PATH, "wake-chopped")
WAKE_AUGMENTED_DIR = p.Path.joinpath(RECORDINGS_PATH, WAKE_AUGMENTED)
NON_WAKE_DIR = p.Path.joinpath(RECORDINGS_PATH, NON_WAKE)
NON_WAKE_CHOPPED_DIR = p.Path.joinpath(RECORDINGS_PATH, "non-wake-chopped")

SAMPLE_RATE = 16000   # per second
NUM_RECORDINGS = 3
HARDWARE = "0,0"
DURATION = 2        # seconds
SAMPLE_RATE = 44100
N_MFCC = 20        # number of features for MFCC to capture
N_AUG_PER_FILE = 30     # how many augmented variants per original
# librosa default hop_len = 512, and with sr=16000 frame_rate = 16000/512 ~= 31
# and since our duration of each audio file is 2 seconds, hence MAX_LEN = 2 * 31 ~ 64
MAX_LEN = 64