# Static Module (Alphabets & Numerals) — MLP over 126‑D Hands

Frame‑level classification for alphabets and numerals using MediaPipe Hands landmarks.

---

## Install
```bash
pip install -r static/requirements.txt
```

---

## 1) Build features
`load.py` crawls your labeled image folders, extracts **Left+Right hand** landmarks (63+63=126), and writes:
```
static/data/load/alphabets_data.npz
static/data/load/numerals_data.npz
```

Run:
```bash
python static/load.py --root /path/to/your/static_dataset_root
```

---

## 2) Train
```bash
python static/train.py
# Saves:
#   static/data/model/{alphabets.pth,numerals.pth}
#   static/data/encoder/{alphabets.pkl,numerals.pkl}
```

Use `accuracy.py` for a quick test accuracy & classification report.

---

## 3) Webcam inference (static only)
```bash
python static/inference.py
# Press 'm' to toggle Alphabet ↔ Numeral; 'q' to quit
```

**Notes**
- Some alphabets are single‑hand only (e.g., C, I, L, O, U, V); numerals use one hand.
- Training applies horizontal‑flip augmentation so either hand can be recognized.
