# Static Gestures

Static gestures are recognized from **single frames** using hand landmarks. The pipeline extracts normalized features,
trains a compact MLP classifier, and offers a realtime inference script with camera support on Windows (MSMF backend).

## Directory structure

```
static/
├── data/
│   ├── encoder/     # Label encoders (saved during training)
│   ├── load/        # Preprocessed feature files (.npz)
│   └── model/       # Trained weights/checkpoints
├── load.py          # Build dataset (.npz) from raw landmarks/images
├── train.py         # Train MLP classifier
├── inference.py     # Realtime webcam inference (toggle alphabets/numerals)
├── debug.py         # Utilities/visualization
└── accuracy.py      # Evaluation helpers
```

## Installation

```bash
pip install -r static/requirements.txt
```

- Python 3.10 recommended.
- `mediapipe==0.10.21` with `numpy<2`.
- `opencv-python==4.11.0.86` for stable MSMF backend on Windows.

## Data policy

The **`static/data/` directory is tracked in Git**. It contains small artifacts such as:
- `.npz` feature files for quick starts,
- trained checkpoints for reference,
- `encoder` objects used by inference.

You can always regenerate these artifacts using the commands below if you prefer a clean run.

## Data preparation

```bash
python static/load.py --data_dir "PATH_TO_RAW" --out static/data/load
```
- Output: `.npz` files in `static/data/load/`.
- Labels: **Alphabets** and **Numerals** are maintained as separate label sets.
  Some alphabet classes are single-hand (e.g., C, I, L, O, U, V); others use two hands.

## Training

```bash
python static/train.py   --data static/data/load   --save_model static/data/model   --save_encoder static/data/encoder   --epochs 300 --batch 64 --lr 1e-3 --seed 42
```

Artifacts:
- **Weights** → `static/data/model/`
- **Label encoders** → `static/data/encoder/`

## Realtime inference

```bash
python static/inference.py
```

During inference:
- Press **`m`** to switch between **alphabets** and **numerals**.
- Camera tips (Windows): the script uses MSMF. If the webcam does not initialize, retry with the default camera once,
  then switch index; or try `cv2.CAP_DSHOW` as a fallback.
