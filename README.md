# Indian Sign Language Recognition — Static & Dynamic Gestures

This repository contains our final-year project on automatic recognition of Indian Sign Language (ISL).
It includes two complementary tracks: **static** (single-frame handshapes) and **dynamic** (short sequences of full-body
pose and hands). Both tracks rely on MediaPipe landmark extraction and train lightweight neural models that can run in
real time on commodity hardware.

> Scope: Ignore `dynamic/include50/updated/` (work-in-progress). Use `static/` and `dynamic/include50/original/` for all experiments.

## Repository layout

```
.
├── static/                 # Static handshape pipeline (features, training, realtime inference)
├── dynamic/
│   └── include50/
│       ├── original/       # Dynamic gesture pipeline (augment/extract, training, inference)
│       └── updated/        # WIP — not part of the evaluated pipeline
└── (docs, configs, etc.)   # See sub-READMEs for details
```

## Quick start

1. **Create a Python 3.10 environment.**
2. Install requirements *per track*:
   ```bash
   # Static gestures
   pip install -r static/requirements.txt
   # Dynamic gestures
   pip install -r dynamic/requirements.txt
   ```
   > GPU users: install the correct PyTorch wheel for your CUDA version (e.g., `--index-url https://download.pytorch.org/whl/cu121`).

3. Follow the **`static/README.md`** and **`dynamic/README.md`** for data preparation, training, and inference.

### Environment notes

- Tested on Windows 10/11 and Python 3.10.
- MediaPipe 0.10.x requires **`numpy<2`**. OpenCV version is pinned for camera backend stability on Windows (MSMF).
- Paths in examples use Windows drive letters; all scripts also accept POSIX paths.

## Data & versioning policy

- **`static/data/` is tracked in Git.** It is small enough to keep under version control (encoders, small `.npz` files,
  and example checkpoints). These files enable out‑of‑the‑box reproduction of the static pipeline.
- **Dynamic data and artifacts are *not* tracked.** The dynamic pipeline generates >100 MB of feature arrays and
  checkpoints; those are excluded via `.gitignore`. Scripts recreate everything deterministically from raw videos.

Where possible, include a **tiny sample** (a few clips or a synthetic mini‑set) for CI/smoke tests without distributing the
full dataset. The dynamic README documents the expected folder layout.

## Method overview

- **Static track**: single-frame classification using hand landmarks (one or both hands). We compute normalized
  coordinates and train a small MLP. The model targets **alphabets** and **numerals**; a toggle in the
  inference script selects the active label set.
- **Dynamic track**: sequence classification using MediaPipe Holistic (pose + both hands). We construct fixed-length
  sequences (padding/trim) and train a BiLSTM-based classifier (optionally with attention).

## Reproducibility

- Each training script accepts a `--seed` flag to fix initialization and data shuffling.
- Model checkpoints and encoders are written to dedicated subfolders (see per-track READMEs).
- We keep all preprocessing steps in script form to enable end-to-end reruns on fresh machines.

## Team

- Member 1 — Dheeraj Devadas 
- Member 2 — Ayush Duduskar
- Member 3 — Aditya Gupta
- Member 4 — Gaurav Patil


