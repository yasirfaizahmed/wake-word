import pathlib as p

WAKE = "wake"
NON_WAKE = "non-wake"
WAKE_AUGMENTED = "wake-augmented"
NOISE = "noise"

PROJECT_PATH = p.Path(__file__).parent.parent
LOGS = p.Path(PROJECT_PATH, "_LOGs")
DATA_DIR = p.Path.joinpath(PROJECT_PATH, ".data")
WAKE_DIR = p.Path.joinpath(DATA_DIR, WAKE)
WAKE_CHOPPED = p.Path.joinpath(DATA_DIR, "wake-chopped")
WAKE_AUGMENTED_DIR = p.Path.joinpath(DATA_DIR, WAKE_AUGMENTED)
NON_WAKE_DIR = p.Path.joinpath(DATA_DIR, NON_WAKE)
NON_WAKE_CHOPPED_DIR = p.Path.joinpath(DATA_DIR, "non-wake-chopped")
NOISE_DATA_DIR = p.Path.joinpath(DATA_DIR, NOISE)

CONFIG_YAML_DIR = PROJECT_PATH
CONFIG_FILE = CONFIG_YAML_DIR.joinpath("config.yaml")

SYNTHESIZED_DIR = p.Path.joinpath(DATA_DIR, "synthesized")
SYNTHESIZED_WW_DIR = p.Path.joinpath(SYNTHESIZED_DIR, WAKE)
