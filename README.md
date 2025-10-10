



##  Wake Word Audio Recorder

This Python script helps you record **wake word** and **non-wake word** audio samples using your system microphone.  
It uses `arecord` (Linux) under the hood to capture audio clips and save them in structured folders for dataset creation.

---

###  Requirements

#### 1. System Dependencies
Make sure you have:
- **Linux OS**
- **ALSA utilities** installed:

```bash
sudo apt update
sudo apt install alsa-utils
```

#### 2\. Python Dependencies

Install Python packages:

```bash
pip install invoke
```

Ensure your project contains a `utils/config.py` file with these constants defined:

```python
from pathlib import Path

RECORDINGS_PATH = Path("recordings")
WAKE_DIR = RECORDINGS_PATH / "wake"
NON_WAKE_DIR = RECORDINGS_PATH / "non-wake"

NUM_RECORDINGS = 10
DURATION = 4
SAMPLE_RATE = 44100
HARDWARE = "0,0"  # card,device — modify this after checking with `arecord -l`
```

---

###  Usage

#### 1\. List Available Audio Devices

Run this to check available mic inputs:

```bash
arecord -l
```

Find your card and device numbers (e.g. `card 0: PCH, device 0: ...`), then note them as `0,0` or `0,2`, etc.

---

#### 2\. Record Wake Word Samples

```bash
python recorder.py wake --count 10 --duration 3 --hw 0,0
```

This records 10 audio clips (each 3 seconds long) in:

```bash
recordings/wake/
```

---

#### 3\. Record Non-Wake Samples

```bash
python recorder.py non-wake --count 20 --duration 3 --hw 0,0
```

This records 20 clips in:

```bash
recordings/non-wake/
```

---

###  Command-Line Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `mode` | Either `wake` or `non-wake` | — |
| `--count` | Number of recordings to capture | `NUM_RECORDINGS` (from config) |
| `--duration` | Duration of each recording in seconds | `DURATION` |
| `--sample-rate` | Audio sample rate (Hz) | `SAMPLE_RATE` |
| `--hw` | Audio hardware ID (card,device) | `HARDWARE` |

---

###  Example Output

```bash
 Recording 10 samples in 'wake' mode...
 Recording: recordings/wake/wake_0.wav
 Recording: recordings/wake/wake_1.wav
...
 Recording complete.
```

---

###  Troubleshooting

-   **No sound in recordings?**  
    Test your mic manually:
    
    ```bash
    arecord -D plughw:0,0 -f S16_LE -c 1 -r 44100 -d 4 test.wav
    aplay test.wav
    ```
    
    If you hear nothing, try another hardware ID (`0,1`, `0,2`, etc.).
    
-   **Permissions issue:**  
    Run with `sudo` or check mic permissions:
    
    ```bash
    sudo chmod a+rw /dev/snd/*
    ```
    

---

###  Folder Structure

```
wake-word/
│
├── utils/
│   ├── config.py
|   └── audio_recorder.py
│
│
└── .data/
    ├── wake/
    │   ├── wake_0.wav
    │   └── wake_1.wav
    │
    └── non-wake/
        ├── non-wake_0.wav
        └── non-wake_1.wav
```

---

###  Tip for Better Data

-   Keep background noise **minimal and consistent**.
    
-   Use the **same mic and distance** for all recordings.
    
-   For non-wake samples, speak random phrases or silence for diversity.
    
-   You can augment data later using `librosa` (speed, pitch, noise).
    

---