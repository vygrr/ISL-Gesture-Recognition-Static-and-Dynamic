# augment.py — INCLUDE dynamic: splits → augmentation → keypoints (hands+pose)
# - GitHub-friendly defaults: reads split CSVs from ./data and auto-picks output:
#     include50 → ./data/include_50/aug_keypoints
#     all       → ./data/include/aug_keypoints
#     topk      → ./data/top_<K>/aug_keypoints  (K = --top_k, default 100; e.g., ./data/top_100/aug_keypoints)
# - Mirrors your original heuristics:
#     1) Torso crop: full width, cut slightly below hip line (hip-based, EMA-stabilized).
#     2) Gentle idle-trim: keep largest active/present block with padding (train only, opt-in).
# - Saves: label_to_id.json, index_{train,val,test}.csv, and {split}/{label_id}/*.npz (x:(200,258) float16, y:int16).
# - Top-K mode: defaults ranking CSV to ./data/real_world_ranking.csv; override via --ranking_csv.

from __future__ import annotations
import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

import mediapipe as mp_solutions

# ----------------- Constants -----------------
SEQ_LEN = 200            # pad/trim target frames
CROP_PX = 50             # fixed px for side/top crops
DTYPE   = np.float16     # compact storage

# -------- Known faulty videos (exclude from ALL splits) --------
FAULTY_EXCLUDE_ANY_SPLIT: List[Tuple[str, str]] = [
    ("40. I",            "MVI_0001"),
    ("40. I",            "MVI_0002"),
    ("40. I",            "MVI_0003"),
    ("40. Paint",        "MVI_4928"),
    ("34. Pen",          "MVI_4908"),
    ("61. Father",       "MVI_3912"),
    ("87. Hot",          "MVI_5138"),
    ("37. Hat",          "MVI_3835"),
    ("37. Hat",          "MVI_4197"),
    ("11. Car",          "MVI_3118"),
    ("1. Dog",           "MVI_3002"),
    ("19. House",        "MVI_3439"),
    ("16. Train Ticket", "MVI_4193"),
    ("42. T-Shirt",      "MVI_4004"),
]

def _match_faulty(label: str, video_path: str) -> bool:
    l = str(label).lower()
    vp = str(video_path).lower()
    for lbl_sub, mvi in FAULTY_EXCLUDE_ANY_SPLIT:
        if lbl_sub.lower() in l and mvi.lower() in vp:
            return True
    return False

def filter_faulty_any_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    bad_idx = []
    for idx, row in df.iterrows():
        if _match_faulty(row["label"], row["video_path"]):
            bad_idx.append(idx)
    if bad_idx:
        print(f"[INFO] Excluding {len(bad_idx)} known-faulty items from {split_name} split.")
        df = df.drop(index=bad_idx).reset_index(drop=True)
    else:
        print(f"[INFO] No known-faulty items found in {split_name} split.")
    return df

# ----------------- MediaPipe: one Holistic per worker -----------------
_HOLISTIC = None
def init_worker():
    global _HOLISTIC
    try: cv2.setNumThreads(1)
    except Exception: pass
    if _HOLISTIC is None:
        _HOLISTIC = mp_solutions.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.67,
            min_tracking_confidence=0.67,
        )

def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def coerce_include50(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda v: str(v).strip().lower() in {"1", "true", "t", "yes", "y"})

def build_label_map(df_list: List[pd.DataFrame]) -> Dict[str, int]:
    labels = sorted(set(pd.concat(df_list)["label"].astype(str).tolist()))
    return {lbl: i for i, lbl in enumerate(labels)}

# ----------------- Label normalization & TOP-K selection -----------------
_label_leading_id_re = re.compile(r"^\s*\d+\s*[\.\)\-:]*\s*")

def normalize_label_text(s: str) -> str:
    """
    Normalize a label/word for fuzzy matching:
    - strip leading numbering like "12. " / "12)" / "12-"
    - lowercase; collapse spaces; treat _ and - as spaces
    """
    s = str(s)
    s = _label_leading_id_re.sub("", s).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def topk_label_set_from_ranking(ranking_csv: Path, top_k: int, dataset_labels: List[str]) -> Tuple[Set[str], Dict[str, str], List[str]]:
    """Return (keep_set of dataset labels, mapping rank_label->dataset_label, misses)."""
    df = pd.read_csv(ranking_csv)
    if df.empty:
        raise ValueError(f"Ranking CSV is empty: {ranking_csv}")

    # pick label and freq columns heuristically
    cols_lower = {c: c.lower() for c in df.columns}
    label_col = next((c for c in df.columns if any(k in cols_lower[c] for k in ["label","word","phrase","name"])), df.columns[0])
    freq_col  = next((c for c in df.columns if any(k in cols_lower[c] for k in ["freq","count","weight","score"])), None)

    if freq_col:
        df = df.sort_values(freq_col, ascending=False)
    top = df[label_col].astype(str).head(int(top_k)).tolist()

    ds_labels = list(set(map(str, dataset_labels)))
    norm_to_ds = {normalize_label_text(lab): lab for lab in ds_labels}

    keep_set: Set[str] = set()
    mapping: Dict[str, str] = {}
    misses: List[str] = []

    for raw in top:
        n = normalize_label_text(raw)
        hit = norm_to_ds.get(n)
        if hit:
            keep_set.add(hit); mapping[raw] = hit; continue

        # single soft match (contains-each-other)
        candidates = [lab for lab in ds_labels
                      if n in normalize_label_text(lab) or normalize_label_text(lab) in n]
        candidates = list(dict.fromkeys(candidates))  # unique, preserve order
        if len(candidates) == 1:
            keep_set.add(candidates[0]); mapping[raw] = candidates[0]
        else:
            misses.append(raw)
    return keep_set, mapping, misses

# ----------------- Old fixed-margin crops (left/right/top) -----------------
def crop_frame_fixed(frame: np.ndarray, crop_kind: str) -> Optional[np.ndarray]:
    if frame is None: return None
    h, w = frame.shape[:2]
    if w <= CROP_PX or h <= CROP_PX: return None
    if crop_kind == "orig":       return frame
    if crop_kind == "left":       return frame[:, CROP_PX:w, :]
    if crop_kind == "right":      return frame[:, 0:w - CROP_PX, :]
    if crop_kind == "top":        return frame[CROP_PX:h, :, :]
    if crop_kind == "top_left":   return frame[CROP_PX:h, CROP_PX:w, :]
    if crop_kind == "top_right":  return frame[CROP_PX:h, 0:w - CROP_PX, :]
    return None

# ----------------- Torso crop (hip-based abdomen cut) -----------------
def abdomen_cut_from_pose(pose_lm, H: int, offset_frac: float = 0.08, px: int = 10) -> int:
    """Compute a horizontal cut slightly BELOW hips: hip_y + offset_frac*torso + px."""
    POSE_L = {"NOSE":0,"LEFT_HIP":23,"RIGHT_HIP":24}
    pts = pose_lm.landmark
    hip_y  = 0.5*(pts[POSE_L["LEFT_HIP"]].y + pts[POSE_L["RIGHT_HIP"]].y) * H
    nose_y = pts[POSE_L["NOSE"]].y * H
    torso  = max(10.0, hip_y - nose_y)
    y_cut  = int(min(H, hip_y + offset_frac*torso + px))
    return max(40, y_cut)

def compute_torso_cut_for_video(video_path: str, max_probe_frames: int = 40, offset_frac: float = 0.08, px: int = 10) -> Optional[int]:
    """Quickly scan the first N frames to lock a stable y_cut (abdomen line)."""
    global _HOLISTIC
    if _HOLISTIC is None:
        init_worker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    ycut_ema = None
    found = False
    for _ in range(max_probe_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if H is None:
            H = frame.shape[0]
        res = _HOLISTIC.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            y_raw = abdomen_cut_from_pose(res.pose_landmarks, H, offset_frac=offset_frac, px=px)
            ycut_ema = y_raw if ycut_ema is None else 0.9*ycut_ema + 0.1*y_raw
            found = True
    cap.release()
    if found:
        return int(round(ycut_ema))
    return None

def crop_frame_torso(frame: np.ndarray, y_cut: int) -> Optional[np.ndarray]:
    """Keep full width and everything ABOVE y_cut (remove bottom)."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    y = max(40, min(h, int(y_cut)))
    return frame[0:y, 0:w, :]

# ----------------- Gentle idle-trim (activity ∨ presence) -----------------
def extract_LH_RH_from_vecs(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given (T,258) features, slice out LH and RH as (T,21,3)."""
    if vecs.ndim != 2 or vecs.shape[1] < 258:
        raise ValueError("Bad vecs shape")
    pose_dim = 33*4
    LH = vecs[:, pose_dim:pose_dim+21*3].reshape(-1, 21, 3)
    RH = vecs[:, pose_dim+21*3:pose_dim+42*3].reshape(-1, 21, 3)
    return LH, RH

def hand_activity(LH: np.ndarray, RH: np.ndarray) -> np.ndarray:
    hands = np.concatenate([LH, RH], axis=1)  # (T,42,3)
    diff  = np.diff(hands, axis=0)
    v     = np.sqrt((diff**2).sum(axis=2)).sum(axis=1)
    return np.concatenate([[0.0], v]).astype(np.float32)

def hand_presence_ratio(LH: np.ndarray, RH: np.ndarray) -> np.ndarray:
    lhp = (LH!=0).any(axis=2).sum(axis=1)
    rhp = (RH!=0).any(axis=2).sum(axis=1)
    return ((lhp + rhp) / 42.0).astype(np.float32)

def gentle_trim_vecs(vecs: np.ndarray,
                     prepad: int = 8, postpad: int = 8,
                     run: int = 8, q: float = 0.35,
                     min_keep_frac: float = 0.20) -> Tuple[np.ndarray, Tuple[int,int]]:
    """Lenient trim on (T,258) based on EMA(activity) ∨ presence; keeps largest block + padding."""
    T = vecs.shape[0]
    LH, RH = extract_LH_RH_from_vecs(vecs)
    act = hand_activity(LH, RH)
    prs = hand_presence_ratio(LH, RH)

    # EMA smooth activity
    act_s = np.zeros_like(act, dtype=np.float32)
    prev = None
    for i, a in enumerate(act):
        prev = a if prev is None else 0.9*prev + 0.1*a
        act_s[i] = prev

    # Binary score: high when activity or hands are present
    # Threshold at quantile q of non-zero activity
    nz = act_s[act_s > 0]
    thr = (np.quantile(nz, q) if nz.size > 0 else 0.0)
    score = ((act_s >= thr) | (prs > 0.10)).astype(np.uint8)

    # Longest run of ones (≥ run)
    best, best_len = (0, 0), 0
    s = None
    for i, v in enumerate(score):
        if v and s is None: s = i
        if (not v or i==len(score)-1) and s is not None:
            e = i if not v else i
            if (e - s + 1) >= run and (e - s + 1) > best_len:
                best_len, best = (e-s+1), (s, e)
            s = None
    i1, i2 = best
    i1 = max(0, i1 - prepad)
    i2 = min(T-1, i2 + postpad)
    if best_len == 0 or (i2 - i1 + 1) < int(min_keep_frac*T):
        return vecs, (0, T-1)
    return vecs[i1:i2+1], (i1, i2)

# ----------------- Core extraction per video -----------------
def extract_keypoints_from_video_worker(video_path: str,
                                        crop_kind: str,
                                        seq_len: int = SEQ_LEN,
                                        torso_offset_frac: float = 0.08,
                                        torso_px: int = 10,
                                        trim_idle: bool = False) -> np.ndarray:
    """Extract (seq_len, 258) feature array from a video using the requested crop & trim."""
    global _HOLISTIC
    if _HOLISTIC is None:
        init_worker()

    # Special handling for 'torso' crop: compute y_cut once (scan a few frames)
    y_cut: Optional[int] = None
    if crop_kind == "torso":
        y_cut = compute_torso_cut_for_video(video_path, offset_frac=torso_offset_frac, px=torso_px)
        # Fallback if no pose detected in probe frames
        if y_cut is None:
            y_cut = 300  # sensible default; will clamp to frame height at runtime

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    feats: List[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Apply crop
        if crop_kind == "torso":
            frame = crop_frame_torso(frame, y_cut)
        else:
            frame = crop_frame_fixed(frame, crop_kind)
        if frame is None:
            continue

        # Run holistic and assemble (pose 33*(x,y,z,v)=132, LH 63, RH 63 → 258)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _HOLISTIC.process(rgb)

        vec: List[float] = []
        # Pose
        if results.pose_landmarks and results.pose_landmarks.landmark:
            for lm in results.pose_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            vec.extend([0.0] * (33 * 4))
        # Left hand
        if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
            for lm in results.left_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * (21 * 3))
        # Right hand
        if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
            for lm in results.right_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * (21 * 3))

        v = np.asarray(vec, dtype=DTYPE)
        if v.shape[0] != 258:
            if v.shape[0] > 258:
                v = v[:258].astype(DTYPE)
            else:
                v = np.pad(v, (0, 258 - v.shape[0]), mode="constant").astype(DTYPE)
        feats.append(v)

    cap.release()

    if len(feats) == 0:
        arr = np.zeros((seq_len, 258), dtype=DTYPE)
        return arr

    arr = np.stack(feats, axis=0)  # (T, 258)

    # Optional gentle idle-trim
    if trim_idle:
        arr, _ = gentle_trim_vecs(arr, prepad=8, postpad=8, run=8, q=0.35, min_keep_frac=0.20)

    # Pad/clip to seq_len
    if arr.shape[0] >= seq_len:
        arr = arr[:seq_len]
    else:
        pad = np.zeros((seq_len - arr.shape[0], 258), dtype=DTYPE)
        arr = np.concatenate([arr, pad], axis=0)

    return arr.astype(DTYPE)

# ----------------- Augmentation kinds -----------------
def get_augment_kinds_for_split(split: str, left: int, right: int, top: int, torso: int) -> List[str]:
    """Return the list of crop kinds to run for this split."""
    if split == "train":
        kinds = ["orig"]
        kinds += ["left"] * max(0, left)
        kinds += ["right"] * max(0, right)
        kinds += ["top"] * max(0, top)
        kinds += ["torso"] * max(0, torso)  # seated-style crop
        return kinds
    # For val/test, don't artificially augment
    return ["orig"]

def build_output_sample_path(out_root: Path, split: str, label_id: int, base_stem: str, suffix: str) -> Path:
    sub = out_root / split / f"{label_id:03d}"
    safe_mkdir(sub)
    return sub / f"{base_stem}__{suffix}.npz"

# ----------------- Tasks & parallel runner -----------------
def build_tasks_for_split(
    split: str,
    df: pd.DataFrame,
    root: Path,
    out_root: Path,
    label_to_id: Dict[str, int],
    left: int,
    right: int,
    top: int,
    torso: int,
    trim_idle_flag: bool,
    torso_frac: float,
    torso_px: int,
) -> List[dict]:
    kinds = get_augment_kinds_for_split(split, left, right, top, torso)
    tasks: List[dict] = []
    # Enumerate kinds per video (stable suffix ordering)
    for row in df.itertuples(index=False):
        label = str(getattr(row, "label"))
        vid_rel = str(getattr(row, "video_path")).replace("\\", "/")
        vid_path = root / vid_rel
        if not vid_path.exists():
            continue
        label_id = label_to_id[label]
        base_stem = Path(vid_rel).with_suffix("").name
        for k_idx, kind in enumerate(kinds):
            suffix = kind if kind == "orig" else f"{kind}_{k_idx}"
            out_p = build_output_sample_path(out_root, split, label_id, base_stem, suffix)
            tasks.append({
                "split": split,
                "label": label,
                "label_id": label_id,
                "video_rel": vid_rel,
                "video_path": str(vid_path),
                "crop_kind": kind,
                "out_path": str(out_p),
                "trim_idle": bool(trim_idle_flag),
                "torso_frac": float(torso_frac),
                "torso_px": int(torso_px),
            })
    return tasks

def worker_run(task: dict) -> dict:
    """Run one task in a worker process. Returns a small dict describing success or error."""
    try:
        arr = extract_keypoints_from_video_worker(
            task["video_path"],
            task["crop_kind"],
            seq_len=SEQ_LEN,
            torso_offset_frac=float(task.get("torso_frac", 0.08)),
            torso_px=int(task.get("torso_px", 10)),
            trim_idle=bool(task.get("trim_idle", False)),
        )
        np.savez_compressed(task["out_path"], x=arr, y=np.int16(task["label_id"]))
        return {
            "ok": True,
            "npz_path": task["out_path"],
            "label": task["label"],
            "label_id": task["label_id"],
            "video_path": task["video_rel"],
            "crop_kind": task["crop_kind"],
        }
    except Exception as e:
        return {"ok": False, "error": f"{e}", "task": task}

def process_split_parallel(
    split: str,
    df: pd.DataFrame,
    root: Path,
    out_root: Path,
    label_to_id: Dict[str, int],
    left: int,
    right: int,
    top: int,
    torso: int,
    workers: int,
    chunksize: int,
    trim_idle_flag: bool,
    torso_frac: float,
    torso_px: int,
):
    tasks = build_tasks_for_split(split, df, root, out_root, label_to_id, left, right, top, torso, trim_idle_flag, torso_frac, torso_px)
    total = len(tasks)
    print(f"[{split}] videos: {len(df)} | tasks (video×crop): {total} | workers: {workers}")

    index_rows = []
    errors = 0

    if total == 0:
        print(f"[{split}] No tasks to run.")
        pd.DataFrame(index_rows).to_csv(out_root / f"index_{split}.csv", index=False)
        return

    with mp.get_context("spawn").Pool(processes=workers, initializer=init_worker) as pool:
        for res in tqdm(pool.imap_unordered(worker_run, tasks, chunksize=chunksize),
                        total=total, desc=f"{split}: tasks", unit="task"):
            if res.get("ok"):
                index_rows.append({
                    "npz_path": res["npz_path"],
                    "label": res["label"],
                    "label_id": res["label_id"],
                    "video_path": res["video_path"],
                    "crop_kind": res["crop_kind"],
                })
            else:
                errors += 1
                t = res.get("task", {})
                tqdm.write(f"[ERR] {split} | {t.get('video_rel')} | {t.get('crop_kind')}: {res.get('error')}")

    idx_df = pd.DataFrame(index_rows)
    idx_path = out_root / f"index_{split}.csv"
    idx_df.to_csv(idx_path, index=False)
    print(f"[{split}] wrote {len(idx_df)} items → {idx_path} | errors: {errors}")

# ---------------------- Defaults & main ----------------------
def default_out_dir(subset: str, top_k: int) -> Path:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    if subset == "include50":
        return data_dir / "include_50" / "aug_keypoints"
    if subset == "all":
        return data_dir / "include" / "aug_keypoints"
    if subset == "topk":
        return data_dir / f"top_{int(top_k)}" / "aug_keypoints"
    return data_dir / "include_50" / "aug_keypoints"

def default_ranking_csv() -> Path:
    return Path(__file__).resolve().parent / "data" / "real_world_ranking.csv"

def read_splits(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_csv = root / "include_train.csv"
    val_csv   = root / "include_val.csv"
    test_csv  = root / "include_test.csv"
    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)

def main():
    ap = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    ap.add_argument("--root", type=str, default=str(script_dir / "data"),
                    help="Dataset root containing include_{train,val,test}.csv and raw videos (default: ./data)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default auto-selected by --subset)")
    ap.add_argument("--subset", choices=["include50","all","topk"], default="include50",
                    help="Subset: include50 (include_50==True), all (no filter), topk (Top-K labels from ranking CSV)")
    ap.add_argument("--ranking_csv", type=str, default=None,
                    help="Ranking CSV for --subset topk (default: ./data/real_world_ranking.csv)")
    ap.add_argument("--top_k", type=int, default=100, help="K for --subset topk (default: 100)")
    # Augmentation counts for TRAIN
    ap.add_argument("--left", type=int, default=10, help="Train: # left crops per video")
    ap.add_argument("--right", type=int, default=10, help="Train: # right crops per video")
    ap.add_argument("--top", type=int, default=0, help="Train: # top crops per video")
    ap.add_argument("--torso", type=int, default=1, help="Train: # torso (abdomen-cut) crops per video")
    # Feature-extraction options
    ap.add_argument("--trim_idle", action="store_true", help="Enable gentle idle-trim (train split only)")
    ap.add_argument("--torso_frac", type=float, default=0.08, help="Abdomen cut offset (fraction of torso)")
    ap.add_argument("--torso_px", type=int, default=10, help="Abdomen cut offset (pixels)")
    # Faulty-filter toggle
    ap.add_argument("--no_faulty_filter", action="store_true", help="Disable filtering of known faulty videos")
    # Runtime
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    ap.add_argument("--chunksize", type=int, default=2, help="Pool.imap_unordered chunksize")
    args = ap.parse_args()

    root = Path(args.root)

    # Decide default output dir if not provided
    if args.out is None:
        out_root = default_out_dir(args.subset, int(args.top_k))
        print(f"[INFO] --out not provided; using default based on --subset → {out_root}")
    else:
        out_root = Path(args.out)
    safe_mkdir(out_root)

    # Read splits
    train_df, val_df, test_df = read_splits(root)

    # Apply subset filter
    subset = args.subset
    if subset == "include50":
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if "include_50" not in df.columns:
                raise ValueError(f"CSV '{name}' must contain 'include_50' for subset=include50.")
        train_df = train_df[coerce_include50(train_df["include_50"])].copy()
        val_df   = val_df[coerce_include50(val_df["include_50"])].copy()
        test_df  = test_df[coerce_include50(test_df["include_50"])].copy()

    elif subset == "topk":
        rk_path = Path(args.ranking_csv) if args.ranking_csv else default_ranking_csv()
        if args.ranking_csv is None:
            print(f"[INFO] --ranking_csv not provided; using default → {rk_path}")
        if not rk_path.exists():
            raise FileNotFoundError(f"Ranking CSV not found: {rk_path}")
        # Build Top-K keep-set
        all_ds_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]], axis=0).astype(str).tolist()
        keep_set, mapping, misses = topk_label_set_from_ranking(rk_path, int(args.top_k), all_ds_labels)
        print(f"[INFO] Top-K mapping: matched={len(mapping)}  misses={len(misses)}  keep_labels={len(keep_set)}")
        if misses:
            preview = ", ".join(misses[:10]); more = "" if len(misses)<=10 else f" (+{len(misses)-10} more)"
            print(f"[WARN] Could not map some ranking labels: {preview}{more}")
        train_df = train_df[train_df["label"].astype(str).isin(keep_set)].copy()
        val_df   = val_df[val_df["label"].astype(str).isin(keep_set)].copy()
        test_df  = test_df[test_df["label"].astype(str).isin(keep_set)].copy()
    # else: 'all' → no filter

    # Optionally exclude known faulty videos
    if not args.no_faulty_filter:
        train_df = filter_faulty_any_split(train_df, split_name="train")
        val_df   = filter_faulty_any_split(val_df,   split_name="val")
        test_df  = filter_faulty_any_split(test_df,  split_name="test")

    # Label map across remaining data
    label_to_id = build_label_map([train_df, val_df, test_df])
    (out_root / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] classes={len(label_to_id)} → saved {out_root/'label_to_id.json'}")

    # Process splits
    print("\n=== START: TRAIN ===")
    process_split_parallel("train", train_df, root, out_root, label_to_id,
                           left=max(0, args.left), right=max(0, args.right), top=max(0, args.top), torso=max(0, args.torso),
                           workers=max(1, args.workers), chunksize=max(1, args.chunksize),
                           trim_idle_flag=bool(args.trim_idle), torso_frac=float(args.torso_frac), torso_px=int(args.torso_px))

    print("\n=== START: VAL ===")
    process_split_parallel("val", val_df, root, out_root, label_to_id,
                           left=0, right=0, top=0, torso=0,
                           workers=max(1, args.workers), chunksize=max(1, args.chunksize),
                           trim_idle_flag=False, torso_frac=float(args.torso_frac), torso_px=int(args.torso_px))

    print("\n=== START: TEST ===")
    process_split_parallel("test", test_df, root, out_root, label_to_id,
                           left=0, right=0, top=0, torso=0,
                           workers=max(1, args.workers), chunksize=max(1, args.chunksize),
                           trim_idle_flag=False, torso_frac=float(args.torso_frac), torso_px=int(args.torso_px))

    print("\n[DONE] All splits processed.")

if __name__ == "__main__":
    # Windows needs spawn guard for multiprocessing
    main()
