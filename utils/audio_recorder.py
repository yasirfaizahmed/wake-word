import invoke
from utils.config import *
import sys
import typing
import argparse
import pathlib as p
  

if __name__ == "__main__":
  if not RECORDINGS_PATH.exists():
    p.Path.mkdir(RECORDINGS_PATH)
    p.Path.mkdir(WAKE_DIR)
    p.Path.mkdir(NON_WAKE_DIR)

  parser = argparse.ArgumentParser(description="Wake, non-wake audio samples recorder")
  parser.add_argument("mode", help="The mode of recorder, either 'wake' or 'non-wake', default is 'wake'", default=WAKE)
  parser.add_argument("--count", type=int, help="Number of samples to record, default is 10", default=NUM_RECORDINGS)
  parser.add_argument("--duration", type=int, help="Duration of each audio sample, default is 4 seconds", default=DURATION)
  parser.add_argument("--sample-rate", type=int, help="Sample rate, default is 44100", default=SAMPLE_RATE)
  parser.add_argument("--path", type=str, help="Absolute path where the samples gets populated, default is '.'", default=WAKE_DIR)
  parser.add_argument("--hw", type=str, help="Audio card, mic device number, default is '0,0'", default=HARDWARE)
  
  args = parser.parse_args()

  for num in range(NUM_RECORDINGS):
    invoke.run(command=f"arecord -D plughw:{args.count} -f S16_LE -c 1 -r {args.sample_rate} -d {args.duration} {args.path}/{args.mode}/{args.mode}{num}.wav")
