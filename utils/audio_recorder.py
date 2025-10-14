import invoke
from ww_config.config import *
import argparse
import time


if __name__ == "__main__":
    # Ensure recording directories exist
    if not RECORDINGS_PATH.exists():
      RECORDINGS_PATH.mkdir(parents=True, exist_ok=True)
    WAKE_DIR.mkdir(parents=True, exist_ok=True)
    NON_WAKE_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Wake word / Non-wake word audio recorder")
    parser.add_argument("mode", choices=["wake", "non-wake"], help="Recording mode")
    parser.add_argument("--count", type=int, default=NUM_RECORDINGS, help="Number of samples to record")
    parser.add_argument("--duration", type=int, default=DURATION, help="Duration of each audio sample in seconds")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="Sample rate in Hz")
    parser.add_argument("--hw", type=str, default=HARDWARE, help="Audio card and mic device number, e.g. '0,0'")

    args = parser.parse_args()

    # Set target directory
    target_dir = WAKE_DIR if args.mode == "wake" else NON_WAKE_DIR

    print(f"\n Recording {args.count} samples in '{args.mode}' mode...")
    for num in range(args.count):
      filename = target_dir / f"{args.mode}_{num}.wav"
      cmd = f"arecord -D plughw:{args.hw} -f S16_LE -c 1 -r {args.sample_rate} -d {args.duration} {filename}"
      print(f" Recording: {filename}")
      invoke.run(cmd, echo=True)
      print("Sleeping for 5 seconds ...")
      time.sleep(5)
    
    print("\n Recording complete.")
