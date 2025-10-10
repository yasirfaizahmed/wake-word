import invoke
import pathlib as p



PROJECT_PATH = p.Path(__file__).parent.parent
RECORDINGS_PATH = p.Path.joinpath(PROJECT_PATH, ".data")

NUM_RECORDINGS = 3
HARDWARE = "0,0"
DURATION = 3 
SAMPLE_RATE = 44100
FILE_PREFIX = "wake"
SUB_FOLDER = FILE_PREFIX


for num in range(NUM_RECORDINGS):
  invoke.run(command=f"arecord -D plughw:{HARDWARE} -f S16_LE -c 1 -r {SAMPLE_RATE} -d {DURATION} {RECORDINGS_PATH}/{SUB_FOLDER}/{FILE_PREFIX}{num}.wav")
