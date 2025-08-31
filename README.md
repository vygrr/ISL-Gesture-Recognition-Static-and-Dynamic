# Indian Sign Language Recognition — Static + Dynamic (Realtime)

A production‑ready ISL project with two branches that work together:

- **Static** (alphabets & numerals): lightweight **MLP** over 126‑D MediaPipe Hands landmarks.
- **Dynamic** (common words): **CTR‑GCN** (plus LSTM/BiLSTM‑Attention/RelPos options) over pose+hands keypoints.
- **Unified realtime app**: `inference.py` (repo root) fuses static+dynamic predictions and can use **Gemini** to stitch tokens into short, grammatical sentences (adds only function words; no new content).

> **Note:** Large datasets/checkpoints are excluded. A helper script to download prepared dynamic keypoints/checkpoints will be added (see **Data & Downloads**).

---

## Install (dependencies)

### Option A — Root runtime (realtime app + Gemini)
At repo root:
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
This installs only what the **root realtime app** needs.

### Option B — Per‑module
If you work inside modules directly:
```bash
# Dynamic module (includes pandas for logs, training CSVs)
pip install -r dynamic/requirements.txt

# Static module (includes joblib for encoder I/O)
pip install -r static/requirements.txt
```

**Python:** 3.10+ recommended. **CUDA:** optional but recommended for dynamic training/inference.

---

## What’s inside
```
Major Project VII/
├─ inference.py                 # Unified realtime (Static+Dynamic) + Gemini sentence formation
├─ gemini_client.py             # Minimal Gemini client
├─ dynamic/
│  ├─ augment.py                # Split → augment → keypoints (pose+hands) with RESUME/verify
│  ├─ train.py                  # CTR‑GCN training (normalize_body/use_bones/use_vel, bi-hand options)
│  ├─ train_alt.py              # LSTM / BiLSTM+Attention / RelPos Transformer training
│  ├─ eval.py                   # Evaluate on val/test; strict ckpt params; macro‑F1/acc/loss
│  ├─ inference.py              # Realtime tester for trained dynamic models
│  ├─ debug_draw.py             # Visualize/annotate sequences, export MP4s
│  ├─ debug_metadata.py         # Inspect dataset stats, label maps, splits
│  └─ debug_frequency.py        # Class‑frequency helper for Top‑K selection
└─ static/
   ├─ load.py                   # Build 126‑D features (MP Hands) → alphabets/numerals .npz
   ├─ train.py                  # Train MLPs and save encoders/models
   ├─ inference.py              # Webcam inference for static only
   ├─ accuracy.py               # Quick test‑set accuracy & report
   └─ collage.py                # Dataset collage utilities
```

---

## Data & Downloads
This repo does **not** include dynamic data (raw videos or extracted keypoints) or large checkpoints.

- **Dynamic keypoints (coming soon):** a helper script (e.g., `tools/download_dynamic_data.py`) will download prepared **augmented keypoints** and example **CTR‑GCN checkpoints** for quick tests. The `dynamic/README.md` documents the expected directory layout so you can prepare your own in the meantime.
- **Static data:** generate 126‑D `.npz` feature files using `static/load.py` from your labeled images.

---

## Quickstart

### 1) Static (alphabets & numerals)
```bash
pip install -r static/requirements.txt
python static/train.py

# Run webcam demo
python static/inference.py
# Expected files:
#   static/data/model/{alphabets.pth,numerals.pth}
#   static/data/encoder/{alphabets.pkl,numerals.pkl}
```

### 2) Dynamic (words)
Prepare augmented keypoints and train/evaluate models — see **dynamic/README.md**.
```bash
pip install -r dynamic/requirements.txt

# Example: Realtime test of a trained CTR‑GCN
python dynamic/inference.py   --data dynamic/data/top_100/aug_keypoints   --ckpt dynamic/data/top_100/ctr_gcn/ckpt_best.pt   --live_draw
```

### 3) Unified realtime (static+dynamic + Gemini)
```bash
pip install -r requirements.txt
python inference.py --use_gemini --gemini_key $GEMINI_API_KEY
# Tips:
#   --mode {auto,manual}   windowing
#   --flip/--no-flip       mirror for left/right dominant signers
#   --default_dynamic      start in dynamic mode (else static)
```

---

## Folder conventions (dynamic)
After augmentation you should have:
```
dynamic/data/<subset>/
├─ aug_keypoints/
│  ├─ label_to_id.json
│  ├─ index_train.csv, index_val.csv, [index_test.csv]
│  ├─ train/<label_id>/*.npz
│  └─ val/<label_id>/*.npz
└─ ctr_gcn/
   ├─ ckpt_best.pt, ckpt_last.pt, params.json, log.csv
   └─ ... (other runs allowed)
```

`<subset>` is typically `include_50`, `include` (full), or `top_<K>` (e.g., `top_100`).

---

## Troubleshooting
- **MediaPipe on Windows:** use prebuilt `mediapipe` wheels and update GPU drivers.
- **Model mismatch:** `eval.py` and `dynamic/inference.py` rebuild features strictly from `params.json`/checkpoint to avoid silent errors.
- **Left‑handed users:** prefer `--flip` at inference (CTR‑GCN trained on right‑handers by default).

