#!/usr/bin/env python3
"""
inference.py (MLP-only)

- Opens webcam
- Loads Alphabet & Numeral PyTorch MLP models + LabelEncoders
- Uses MediaPipe Hands to extract 126-dim features per frame
- Press 'm' to switch Alphabet ↔ Numeral
- Shows current domain, prediction & confidence
- Press 'q' to quit

Expected files:
  data/model/alphabets.pth, data/encoder/alphabets.pkl
  data/model/numerals.pth,  data/encoder/numerals.pkl
"""

import cv2
import numpy as np
import torch
import joblib
import mediapipe as mp
from pathlib import Path
import sys

# ------------------ Config ------------------
HERE = Path(__file__).resolve().parent
MODEL_DIR   = HERE / "data" / "model"
ENCODER_DIR = HERE / "data" / "encoder"

FILES = {
    "Alphabet": {
        "model":   MODEL_DIR / "alphabets_model.pth",
        "encoder": ENCODER_DIR / "alphabets_le.pkl",
    },
    "Numeral": {
        "model":   MODEL_DIR / "numerals_model.pth",
        "encoder": ENCODER_DIR / "numerals_le.pkl",
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# ------------------ Model ------------------
class MLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),

            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ MediaPipe ------------------
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------ Load models ------------------
def ensure_files():
    missing = []
    for dom in FILES:
        for k, p in FILES[dom].items():
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("ERROR: Missing required files:")
        for m in missing:
            print("  -", m)
        sys.exit(1)

def load_all():
    bank = {}
    for dom, paths in FILES.items():
        le = joblib.load(paths["encoder"])
        model = MLP(input_dim=126, num_classes=len(le.classes_)).to(DEVICE)
        state = torch.load(paths["model"], map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        bank[dom] = {"model": model, "encoder": le}
    return bank

# ------------------ Feature extraction ------------------
def extract_feature_vector(frame):
    """
    Returns:
      vec (np.ndarray, 126) + landmarks for drawing; or (None, None) if no hands.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None

    feats = {"Left": [0.0]*63, "Right": [0.0]*63}
    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        side = handed.classification[0].label  # "Left" or "Right"
        coords = [c for p in lm.landmark for c in (p.x, p.y, p.z)]
        feats[side] = coords

    vec = np.array(feats["Left"] + feats["Right"], dtype=np.float32)
    return vec, res.multi_hand_landmarks

# ------------------ Main loop ------------------
if __name__ == "__main__":
    ensure_files()
    models = load_all()
    print("Loaded:", ", ".join(models.keys()))
    print("Press 'm' to switch Alphabet/Numeral, 'q' to quit.")

    current_domain = "Alphabet"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        hands.close()
        sys.exit(1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        vec, lms = extract_feature_vector(frame)
        label_text = "No hands detected"

        if vec is not None:
            inp = torch.from_numpy(vec).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = models[current_domain]["model"](inp)
                probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            label = models[current_domain]["encoder"].inverse_transform([pred])[0]
            label_text = f"{label} ({conf*100:.1f}%)"

            for lm in lms:
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

        # UI
        cv2.rectangle(frame, (0,0), (540,110), (0,0,0), -1)
        cv2.putText(frame, f"Domain: {current_domain}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,0), 2)
        cv2.putText(frame, f"Pred  : {label_text}", (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,0), 2)
        cv2.putText(frame, "Press 'm' to switch  |  'q' to quit", (10,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("ISL Real-Time Prediction (MLP)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            current_domain = "Numeral" if current_domain == "Alphabet" else "Alphabet"
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
