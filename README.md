# Indian Sign Language Recognition — Static + Dynamic (Realtime)

A production‑ready ISL recognition project with two branches that work together:

- **Static** (alphabets & numerals): lightweight **MLP** over 126‑D MediaPipe hand landmarks.
- **Dynamic** (top‑K/common words): **CTR‑GCN** and alternative sequence models over pose+hands keypoints.
- **Unified realtime app**: `inference.py` at repo root combines static & dynamic predictions and optionally
  uses **Gemini** to stitch tokens into short, grammatical sentences (adds only function words; no new content).

> **Note:** This repo intentionally excludes heavy datasets and checkpoints. A small helper script to download the
> dynamic keypoints/data will be added (see **Data & Downloads**).

---

## What’s inside
```
Major Project VII/
├─ inference.py                 # Unified realtime (Static+Dynamic) + Gemini sentence formation
├─ gemini_client.py             # Minimal client with guardrails for sentence formation
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

## Environment
- Python **3.10+**
- GPU optional but recommended for dynamic training/inference
- Suggested packages (install via pip):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install mediapipe opencv-python numpy pandas scikit-learn tqdm joblib requests wordfreq
  ```

> Exact versions aren’t hard‑pinned here. For deterministic runs, create a virtualenv and freeze with `pip freeze > requirements.txt` once working.

---

## Data & Downloads (important)
This repo does **not** include dynamic data (raw videos or extracted keypoints) or large checkpoints.

- **Dynamic keypoints (to be added):** In the next update, a helper script (e.g. `tools/download_dynamic_data.py`) will download the
  prepared **augmented keypoints** and example **CTR‑GCN checkpoints** for quick testing. The README in `dynamic/` already documents
  the expected directory layout so you can prepare them yourself if preferred.
- **Static data:** You can generate the 126‑D `.npz` feature files using `static/load.py` from your labeled images.

---

## Quickstart

### 1) Static (alphabets & numerals)
```bash
# Train
python static/train.py

# Run webcam demo
python static/inference.py
# Expected files:
#   static/data/model/{alphabets.pth,numerals.pth}
#   static/data/encoder/{alphabets.pkl,numerals.pkl}
```

### 2) Dynamic (words)
Prepare augmented keypoints and train/evaluate models — see full details in **dynamic/README.md**.
```bash
# Example: Realtime test of a trained CTR‑GCN
python dynamic/inference.py   --data dynamic/data/top_100/aug_keypoints   --ckpt dynamic/data/top_100/ctr_gcn/ckpt_best.pt   --live_draw
```

### 3) Unified realtime (static+dynamic + Gemini)
```bash
# Default looks for dynamic assets in dynamic/data/top_100/{aug_keypoints,ctr_gcn}
python inference.py --use_gemini --gemini_key $GEMINI_API_KEY
# Tips:
#   --mode {auto,manual}   windowing
#   --flip/--no-flip       mirror for left/right dominant signers
#   --default_dynamic      start in dynamic mode (else static)
```

---

## Folder conventions (dynamic)
The dynamic pipeline expects this shape **after** augmentation:
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
- **MediaPipe errors on Windows:** install `mediapipe` prebuilt wheels and ensure a modern GPU driver.
- **OpenCV high CPU usage:** the code sets low thread counts; ensure other apps aren’t grabbing the camera.
- **Model mismatch:** `eval.py` and `dynamic/inference.py` strictly read `params.json` or `ckpt['params']` to rebuild the feature config.
- **Left‑handed users:** prefer `--flip` at inference (the CTR‑GCN was trained on right‑handers by default).
