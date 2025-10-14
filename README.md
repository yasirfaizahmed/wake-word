
[![wakatime](https://wakatime.com/badge/user/a9e00d41-03d8-4310-b678-7bcc046966dc/project/92fd1c78-5e08-4909-9fc4-539480f23b99.svg)](https://wakatime.com/badge/user/a9e00d41-03d8-4310-b678-7bcc046966dc/project/92fd1c78-5e08-4909-9fc4-539480f23b99)


###  Overview

**Wake-Word** is a lightweight end-to-end pipeline for creating **custom wake word detection models** designed for **edge devices** — optimized to work with **minimal labeled data**.

This repository provides tools for:

-    Recording & managing your dataset (wake / non-wake samples)
    
-    Preprocessing and feature extraction using **MFCCs**
    
-    Training a **compact LSTM-based model** from scratch
    
-    Model, optimizer, and loss initialization via **config-driven setup**
    
-    Exporting and testing models on low-power devices
    

---

###  Architecture

```bash
Wake-Word
│
├── .data/                 # Raw + processed data
│   ├── wake/
│   └── non-wake/
│
├── models/                # Trained and exported models
│
├── utils/
│   ├── config.py          # Constants and paths
│   ├── audio_recorder.py  # Recorder utility
│   ├── features.py        # MFCC extraction & augmentation
│   └── training.py        # Model training pipeline
│
├── schemas.py             # Pydantic-based config schemas
├── recorder.py            # CLI tool for dataset creation
└── train.py               # Main model training entrypoint
```

---

###  Installation

```bash
git clone --recurse-submodules https://github.com/yasirfaizahmed/wake-word.git
cd wake-word
pip install -r requirements.txt
```

#### System Dependencies

Make sure you have:

-   **Linux OS**
    
-   **ALSA utilities** for microphone input:
    

```bash
sudo apt update
sudo apt install alsa-utils
```

---

###  Configuration

Set up your constants in `utils/config.py`:

```python
from pathlib import Path

RECORDINGS_PATH = Path(".data")
WAKE_DIR = RECORDINGS_PATH / "wake"
NON_WAKE_DIR = RECORDINGS_PATH / "non-wake"

NUM_RECORDINGS = 10
DURATION = 4
SAMPLE_RATE = 44100
HARDWARE = "0,0"  # modify after checking `arecord -l`
```

---

###  Recording Dataset

The repository includes a **CLI audio recorder** to quickly build your dataset.

#### List Available Devices

```bash
arecord -l
```

#### Record Wake Word Samples

```bash
python recorder.py wake --count 10 --duration 3 --hw 0,0
```

Saved to:

```bash
.data/wake/
```

#### Record Non-Wake Samples

```bash
python recorder.py non-wake --count 20 --duration 3 --hw 0,0
```

Saved to:

```bash
.data/non-wake/
```

#### Command-Line Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `mode` | `wake` or `non-wake` | — |
| `--count` | Number of recordings | `NUM_RECORDINGS` |
| `--duration` | Duration per clip (s) | `DURATION` |
| `--sample-rate` | Audio sample rate (Hz) | `SAMPLE_RATE` |
| `--hw` | Mic hardware ID (card,device) | `HARDWARE` |

---

###  Model Creation Workflow

The training pipeline uses **Pydantic schemas** to define configurations:

```python
from schemas import ModelInitializerConfig, WakeWordConfig, DataLoaderConfig, OptimizerConfig, CriterionConfig

config = ModelInitializerConfig(
    wakeword_config=WakeWordConfig(input_size=20, hidden_size=64, output_size=1),
    dataloader_config=DataLoaderConfig(batch_size=32, num_workers=2),
    optimizer_config=OptimizerConfig(lr=0.001),
    criterion_config=CriterionConfig(loss_function="BCELoss")
)
```

The pipeline initializes:

-   **LSTM-based Wake Word model**
    
-   **Adam optimizer**
    
-   **BCELoss** criterion (auto-instantiated)
    
-   **DataLoader** for wake/non-wake samples
    

---

###  Training

After collecting data:

```bash
python train.py
```

This will:

-   Extract MFCC features
    
-   Split data into train/val sets
    
-   Train the LSTM classifier
    
-   Save model weights in `models/`
    

---

###  Troubleshooting

**No sound in recordings?**

```bash
arecord -D plughw:0,0 -f S16_LE -c 1 -r 44100 -d 4 test.wav
aplay test.wav
```

Try different device IDs (`0,1`, `0,2`, etc.) if no audio is captured.

**Permission error:**

```bash
sudo chmod a+rw /dev/snd/*
```

---

###  Tips for Better Data

-   Record in **consistent quiet environments**
    
-   Keep **same mic and distance**
    
-   Add **non-wake randomness** (silence, background talk, noise)
    
-   Augment using **Librosa** for pitch, speed, and noise variation
    

---

###  Roadmap

-    Automated MFCC augmentation
    
-    Quantization-aware training for edge deployment
    
-    ONNX export + microcontroller demo
    
-    Wake-word continuous detection loop
    

---