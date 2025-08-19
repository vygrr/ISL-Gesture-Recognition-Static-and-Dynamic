# Dynamic Gestures (INCLUDE-50)

Dynamic gestures are recognized from **short video sequences** using MediaPipe Holistic to obtain pose and hand
landmarks per frame. We prepare fixed-length sequences and train a BiLSTM-based classifier (optionally with attention).
All scripts used in our experiments are under `dynamic/include50/original/`.

> The directory `dynamic/include50/updated/` is a work-in-progress branch and is **not** part of the evaluated pipeline.

## Directory structure

```
dynamic/include50/
├── original/
│   ├── augment.py      # Optional: crop/pad frames, basic augmentation, write landmarks
│   ├── train.py        # Train BiLSTM classifier
│   ├── inference.py    # One-shot realtime/interactive inference
│   └── debug_*.py      # Diagnostics and visualization utilities
└── updated/            # WIP (ignored)
```

## Installation

```bash
pip install -r dynamic/requirements.txt
```

- Python 3.10 recommended.
- MediaPipe 0.10.x with `numpy<2`.
- PyTorch (CPU or CUDA; install the wheel matching your CUDA version if using GPU).

## Data policy

The **dynamic data footprint exceeds 100 MB**, so it is **excluded from the repository** via `.gitignore`.
This includes raw videos, extracted keypoints, large `.npz`/`.npy` arrays, checkpoints, and training runs.
Use the commands below to regenerate artifacts locally.

If you need reproducible, shareable artifacts without committing them, consider
- releasing checkpoints as GitHub **release assets**, or
- using an external storage backend (e.g., DVC, cloud object storage).

## Data preparation

1. **Augment/crop (optional) and extract landmarks**
   ```bash
   python dynamic/include50/original/augment.py      --root "D:/INCLUDE"      --out  "D:/INCLUDE_KEYPOINTS"      --left 10 --right 10 --top 0      --workers 4
   ```
   - Reads raw INCLUDE-50 videos from `--root` and writes processed keypoints under `--out`.
   - Basic cropping margins (`--left/--right/--top`) help center the signer for seated-use cases.
   - Parallelism via `--workers`.

2. **Sequence construction**
   - Landmarks from pose and both hands are concatenated per frame, normalized, and sampled to a fixed length `T`
     (default 200). Short clips are padded; long clips are trimmed.

## Training

```bash
python dynamic/include50/original/train.py   --data "D:/INCLUDE_KEYPOINTS"   --save "D:/RUNS_BiLSTM"   --epochs 120 --batch 32 --lr 1e-3   --workers 2 --seed 42 --amp
```
- Saves checkpoints, logs, and label encoders under `--save`.
- `--amp` enables mixed precision when a suitable GPU is available.

## Inference (camera)

```bash
python dynamic/include50/original/inference.py   --ckpt "D:/RUNS_BiLSTM/best.ckpt"   --width 1280 --height 720   --hol_complex 2   --hands_fallback
```

- **Windows camera backends**: the scripts default to MSMF. If a virtual camera (e.g., OBS or Camo) is not detected:
  - Ensure the virtual camera driver is installed.
  - Try `CAP_DSHOW` as a fallback and verify the device index.
  - If the feed appears only after opening another camera once, insert a short warm-up grab loop before reading frames.
  - Avoid “exclusive control” in other apps.

## Reproducibility

- Use `--seed` to fix all RNG sources where supported.
- Keep feature extraction parameters and sequence length consistent between training and inference.
