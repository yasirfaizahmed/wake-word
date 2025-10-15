import pathlib as p


WAKE = "wake"
NON_WAKE = "non-wake"
WAKE_AUGMENTED = "wake-augmented"
NOISE = "noise"

PROJECT_PATH = p.Path(__file__).parent.parent
RECORDINGS_PATH = p.Path.joinpath(PROJECT_PATH, ".data")
WAKE_DIR = p.Path.joinpath(RECORDINGS_PATH, WAKE)
WAKE_CHOPPED = p.Path.joinpath(RECORDINGS_PATH, "wake-chopped")
WAKE_AUGMENTED_DIR = p.Path.joinpath(RECORDINGS_PATH, WAKE_AUGMENTED)
NON_WAKE_DIR = p.Path.joinpath(RECORDINGS_PATH, NON_WAKE)
NON_WAKE_CHOPPED_DIR = p.Path.joinpath(RECORDINGS_PATH, "non-wake-chopped")

NOISE_DATA_DIR = p.Path.joinpath(RECORDINGS_PATH, NOISE)