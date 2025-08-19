#!/usr/bin/env python3
"""
load.py

Discovers "English Alphabet" & "Numerals" folders inside a provided dataset root,
extracts MediaPipe hand landmarks (Left 63 + Right 63 = 126 features),
applies horizontal‐flip augmentation (to simulate left-handed signing),
filters samples by required hand-count, and writes:

  data/load/alphabets_data.npz  (X: [n,126], y: [n,])
  data/load/numerals_data.npz   (X: [m,126], y: [m,])

Rules:
- Numerals require 1 hand.
- Alphabets: C, I, L, O, U, V require 1 hand; all others (including J, H, Y) require 2.
- Labels (folder names) are kept as-is (e.g., E1, E2, 9a, 9b).

Usage:
  python load.py --data_dir "PATH_TO_DATASET_ROOT"

Where PATH_TO_DATASET_ROOT is the folder that directly contains
"English Alphabet" and "Numerals" subfolders (or they appear in subtrees).
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

# Single-hand only alphabet letters
SINGLE_HAND_ALPHABETS = {"C", "I", "L", "O", "U", "V"}

# MediaPipe Hands (static images)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_landmarks(img):
    """
    Returns (126-dim feature vector, hand_count) or (None,0) if no hands.
    Feature layout: Left(63) then Right(63). Missing hand → zeros.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, 0

    feats = {"Left": [0.0]*63, "Right": [0.0]*63}
    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        side = handed.classification[0].label  # "Left" or "Right"
        coords = [c for p in lm.landmark for c in (p.x, p.y, p.z)]
        feats[side] = coords

    vec = np.array(feats["Left"] + feats["Right"], dtype=np.float32)
    return vec, len(res.multi_hand_landmarks)

def find_class_dirs(root: Path):
    """
    Walk `root` to collect all “English Alphabet” and “Numerals” directories.
    Returns: (alpha_dirs, num_dirs) as lists of Paths.
    """
    alpha_dirs, num_dirs = [], []
    for dp, dns, _ in os.walk(root):
        for d in dns:
            if d == "English Alphabet":
                alpha_dirs.append(Path(dp) / d)
            elif d == "Numerals":
                num_dirs.append(Path(dp) / d)
    return alpha_dirs, num_dirs

def load_data(class_dirs, is_alpha=True):
    """
    For each class directory:
      - decide required hands (numerals=1; alphabets=1 for SINGLE_HAND_ALPHABETS else 2)
      - iterate images, extract landmarks; keep only samples matching req. hand count
      - horizontally flip the image and repeat (left-hand augmentation)
      - keep label exactly as folder name (e.g., E1, E2, 9a, 9b, J, H, Y)
    Returns: X (n×126), y (n,)
    """
    X, y = [], []
    kind = "Alphabets" if is_alpha else "Numerals"
    print(f"\nLoading {kind} with flip augmentation...")

    for base in class_dirs:
        print(" •", base)
        for lbl in sorted(os.listdir(base)):
            folder = Path(base) / lbl
            if not folder.is_dir():
                continue

            # required hands
            req = 1 if (not is_alpha or lbl in SINGLE_HAND_ALPHABETS) else 2

            images = [f for f in os.listdir(folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            kept = 0
            for imgf in tqdm(images, desc=f"   {lbl}", unit="img"):
                path = str(folder / imgf)
                img = cv2.imread(path)
                if img is None:
                    continue

                # original
                feats, cnt = extract_landmarks(img)
                if feats is not None and cnt == req:
                    X.append(feats); y.append(lbl); kept += 1

                # flipped
                img_flipped = cv2.flip(img, 1)
                feats_f, cnt_f = extract_landmarks(img_flipped)
                if feats_f is not None and cnt_f == req:
                    X.append(feats_f); y.append(lbl); kept += 1

            print(f"    → {kept} samples (incl. flips) for '{lbl}'")

    if X:
        return np.vstack(X), np.array(y, dtype=object)
    else:
        return np.empty((0,126), dtype=np.float32), np.array([], dtype=object)

def get_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True,
                   help="Dataset root (folder containing 'English Alphabet' and 'Numerals', possibly nested).")
    p.add_argument("--out_dir", type=Path, default=here / "data" / "load",
                   help="Where to save the .npz files (default: static/data/load).")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    alpha_dirs, num_dirs = find_class_dirs(args.data_dir)

    X_a, y_a = load_data(alpha_dirs, is_alpha=True)
    X_n, y_n = load_data(num_dirs,  is_alpha=False)

    np.savez_compressed(args.out_dir / "alphabets_data.npz", X=X_a, y=y_a)
    np.savez_compressed(args.out_dir / "numerals_data.npz",  X=X_n, y=y_n)

    print("\n✅ Saved:")
    print("  ", args.out_dir / "alphabets_data.npz")
    print("  ", args.out_dir / "numerals_data.npz")

    hands.close()
