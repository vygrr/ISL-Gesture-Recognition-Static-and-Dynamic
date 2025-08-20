#!/usr/bin/env python3
"""
train.py — CTR-GCN on augmented keypoints (pose+hands) with optional bi-handed training.

Data expectation (from augment.py):
  <data_root>/
    label_to_id.json
    index_train.csv
    index_val.csv
    index_test.csv     (optional)
    train/<label_id>/*.npz, val/<label_id>/*.npz, ...

Auto save-dir selection:
  If --save is NOT given and --data points to one of:
    dynamic/data/include_50/aug_keypoints
    dynamic/data/include/aug_keypoints
    dynamic/data/top_<K>/aug_keypoints
  then outputs go to:
    dynamic/data/<subset>/(ctr_gcn | ctr_gcn_bihand)
  Otherwise, pass --save explicitly.

Checkpoints & logs:
  - ckpt_best.pt    (best by val macro-F1; tie-break by lower val loss)
  - ckpt_last.pt    (last epoch for resume)
  - log.csv         (per-epoch metrics)
  - params.json     (run config)

Bi-handed training:
  --bihand            : enable random left/right flip+swap (no canonicalisation)
  --bihand_p          : per-sample flip prob (during TRAIN batches)
  --bihand_ramp_epoch : linearly ramp flip prob 0→bihand_p over the first E epochs
  Validation when --bihand is ON is forced to "both":
    evaluate original and deterministically flipped inputs, average probabilities,
    and compute metrics on the averaged prediction.
"""

from __future__ import annotations
import argparse, json, math, os, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------- Constants ----------------------
SEQ_LEN  = 200
FEAT_DIM = 258
V_POSE, V_LHAND, V_RHAND = 33, 21, 21
V_ALL = V_POSE + V_LHAND + V_RHAND  # 75

POSE_L = {
    "NOSE":0, "LEFT_EYE_INNER":1, "LEFT_EYE":2, "LEFT_EYE_OUTER":3,
    "RIGHT_EYE_INNER":4, "RIGHT_EYE":5, "RIGHT_EYE_OUTER":6,
    "LEFT_EAR":7, "RIGHT_EAR":8,
    "MOUTH_LEFT":9, "MOUTH_RIGHT":10,
    "LEFT_SHOULDER":11, "RIGHT_SHOULDER":12,
    "LEFT_ELBOW":13, "RIGHT_ELBOW":14,
    "LEFT_WRIST":15, "RIGHT_WRIST":16,
    "LEFT_PINKY":17, "RIGHT_PINKY":18,
    "LEFT_INDEX":19, "RIGHT_INDEX":20,
    "LEFT_THUMB":21, "RIGHT_THUMB":22,
    "LEFT_HIP":23, "RIGHT_HIP":24,
    "LEFT_KNEE":25, "RIGHT_KNEE":26,
    "LEFT_ANKLE":27, "RIGHT_ANKLE":28,
    "LEFT_HEEL":29, "RIGHT_HEEL":30,
    "LEFT_FOOT_INDEX":31, "RIGHT_FOOT_INDEX":32,
}
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
POSE_EDGES = [
    (POSE_L["LEFT_SHOULDER"],  POSE_L["RIGHT_SHOULDER"]),
    (POSE_L["LEFT_SHOULDER"],  POSE_L["LEFT_ELBOW"]),
    (POSE_L["LEFT_ELBOW"],     POSE_L["LEFT_WRIST"]),
    (POSE_L["RIGHT_SHOULDER"], POSE_L["RIGHT_ELBOW"]),
    (POSE_L["RIGHT_ELBOW"],    POSE_L["RIGHT_WRIST"]),
    (POSE_L["LEFT_SHOULDER"],  POSE_L["LEFT_HIP"]),
    (POSE_L["RIGHT_SHOULDER"], POSE_L["RIGHT_HIP"]),
    (POSE_L["LEFT_HIP"],       POSE_L["RIGHT_HIP"]),
    (POSE_L["LEFT_HIP"],       POSE_L["LEFT_KNEE"]),
    (POSE_L["LEFT_KNEE"],      POSE_L["LEFT_ANKLE"]),
    (POSE_L["RIGHT_HIP"],      POSE_L["RIGHT_KNEE"]),
    (POSE_L["RIGHT_KNEE"],     POSE_L["RIGHT_ANKLE"]),
    (POSE_L["NOSE"],           POSE_L["LEFT_SHOULDER"]),
    (POSE_L["NOSE"],           POSE_L["RIGHT_SHOULDER"]),
]
POSE_WRIST_L = POSE_L["LEFT_WRIST"]
POSE_WRIST_R = POSE_L["RIGHT_WRIST"]
LHAND_ROOT   = V_POSE + 0
RHAND_ROOT   = V_POSE + V_LHAND + 0

POSE_SWAP_PAIRS = [
    (POSE_L["LEFT_EYE_INNER"], POSE_L["RIGHT_EYE_INNER"]),
    (POSE_L["LEFT_EYE"],       POSE_L["RIGHT_EYE"]),
    (POSE_L["LEFT_EYE_OUTER"], POSE_L["RIGHT_EYE_OUTER"]),
    (POSE_L["LEFT_EAR"],       POSE_L["RIGHT_EAR"]),
    (POSE_L["MOUTH_LEFT"],     POSE_L["MOUTH_RIGHT"]),
    (POSE_L["LEFT_SHOULDER"],  POSE_L["RIGHT_SHOULDER"]),
    (POSE_L["LEFT_ELBOW"],     POSE_L["RIGHT_ELBOW"]),
    (POSE_L["LEFT_WRIST"],     POSE_L["RIGHT_WRIST"]),
    (POSE_L["LEFT_PINKY"],     POSE_L["RIGHT_PINKY"]),
    (POSE_L["LEFT_INDEX"],     POSE_L["RIGHT_INDEX"]),
    (POSE_L["LEFT_THUMB"],     POSE_L["RIGHT_THUMB"]),
    (POSE_L["LEFT_HIP"],       POSE_L["RIGHT_HIP"]),
    (POSE_L["LEFT_KNEE"],      POSE_L["RIGHT_KNEE"]),
    (POSE_L["LEFT_ANKLE"],     POSE_L["RIGHT_ANKLE"]),
    (POSE_L["LEFT_HEEL"],      POSE_L["RIGHT_HEEL"]),
    (POSE_L["LEFT_FOOT_INDEX"],POSE_L["RIGHT_FOOT_INDEX"]),
]

# ---------------------- IO utils ----------------------
def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def auto_save_dir(data_root: Path, model_folder: str) -> Optional[Path]:
    """
    If data_root ends with one of:
      .../data/include_50/aug_keypoints
      .../data/include/aug_keypoints
      .../data/top_<K>/aug_keypoints
    return dynamic/data/<subset>/<model_folder>, else None.
    """
    if data_root.name != "aug_keypoints":
        return None
    subset = data_root.parent.name  # include_50 | include | top_<K>
    if subset.startswith("include") or subset.startswith("top_"):
        return data_root.parent.parent / subset / model_folder
    return None

def _infer_split_from_csv(csv_path: Path) -> str:
    name = csv_path.name.lower()
    if "train" in name: return "train"
    if "val"   in name: return "val"
    if "test"  in name: return "test"
    return ""

def _parse_from_raw_path(raw: str):
    """
    Extract (split, label_dir, filename) hints from a stored path, even if absolute.
    """
    p = Path(raw)
    filename = p.name
    label_dir = p.parent.name if p.parent.name.isdigit() else ""
    split_dir = ""
    if len(p.parents) >= 2:
        cand = p.parents[1].name.lower()
        if cand in ("train","val","test"):
            split_dir = cand
    return split_dir, label_dir, filename

def load_index(csv_path: Path, base: Path) -> Tuple[List[str], List[int]]:
    """
    Robustly resolves npz file paths recorded in index_*.csv, even if those
    paths were saved on a different machine.

    Strategy per row:
      1) Use path as-is (absolute) if it exists.
      2) If relative, resolve under `base`.
      3) If missing, try `base / filename`.
      4) If missing, reconstruct as `base / <split>/<label_dir>/<filename>`
         where <split>, <label_dir> are extracted from the saved path.
      5) If still missing, reconstruct as `base / <split_from_csv>/<label_from_col>/<filename>`
         using the CSV being read (train/val/test) and the numeric label_id column
         (zero-padded to 3 digits, and unpadded fallback).
      6) Final fallback: `base / <split_from_csv>/<filename>` (rare layouts).
    """
    if not csv_path.exists():
        return [], []

    df = pd.read_csv(csv_path)
    need = {"npz_path", "label_id"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {need}")

    split_from_csv = _infer_split_from_csv(csv_path)
    paths, labels = [], []

    for _, r in df.iterrows():
        raw = str(r["npz_path"]).strip()
        label_id = int(r["label_id"])
        split_from_raw, label_from_raw, filename = _parse_from_raw_path(raw)
        label_03 = f"{label_id:03d}"

        candidates: List[Path] = []

        p_raw = Path(raw)
        if p_raw.is_absolute():
            candidates.append(p_raw)

        candidates.append((base / raw).resolve())      # relative under base
        candidates.append((base / filename).resolve()) # file directly under base

        if split_from_raw and label_from_raw:
            candidates.append((base / split_from_raw / label_from_raw / filename).resolve())

        if split_from_csv:
            candidates.append((base / split_from_csv / label_03 / filename).resolve())
            candidates.append((base / split_from_csv / str(label_id) / filename).resolve())

        if split_from_raw:
            candidates.append((base / split_from_raw / label_03 / filename).resolve())
            candidates.append((base / split_from_raw / str(label_id) / filename).resolve())

        if split_from_csv:
            candidates.append((base / split_from_csv / filename).resolve())

        chosen = None
        for c in candidates:
            try:
                if c.exists():
                    chosen = c
                    break
            except OSError:
                continue

        if chosen is None:
            tqdm.write(f"[WARN] missing sample (skip): {raw}")
            continue

        paths.append(str(chosen))
        labels.append(label_id)

    return paths, labels

# ---------------------- Dataset ----------------------
class NpzSeq(Dataset):
    def __init__(self, paths: List[str], labels: List[int]):
        assert len(paths) == len(labels)
        self.paths, self.labels = paths, labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        with np.load(self.paths[idx]) as z:
            x = z["x"].astype(np.float32)     # (T,258)
            y = int(z.get("y", self.labels[idx]))
        valid = np.any(x != 0, axis=1)
        length = int(valid.nonzero()[0].max()) + 1 if valid.any() else x.shape[0]
        length = max(1, min(int(length), x.shape[0]))
        return x, y, length

def collate(batch):
    xs, ys, ls = zip(*batch)
    X = torch.from_numpy(np.stack(xs, 0))    # (B,T,258)
    Y = torch.tensor(ys, dtype=torch.long)
    L = torch.tensor(ls, dtype=torch.long)
    return X, Y, L

# ---------------------- Feature helpers ----------------------
def slice_to_joints_xyz(x_258: np.ndarray) -> np.ndarray:
    T = x_258.shape[0]
    pose  = x_258[:, :33*4].reshape(T, 33, 4)[..., :3]
    lhand = x_258[:, 33*4 : 33*4 + 21*3].reshape(T, 21, 3)
    rhand = x_258[:, 33*4 + 21*3 : ].reshape(T, 21, 3)
    return np.concatenate([pose, lhand, rhand], axis=1)  # (T,75,3)

def body_center_scale(joints: np.ndarray, eps=1e-6) -> np.ndarray:
    j = joints.copy()
    lh, rh = POSE_L["LEFT_HIP"], POSE_L["RIGHT_HIP"]
    ls, rs = POSE_L["LEFT_SHOULDER"], POSE_L["RIGHT_SHOULDER"]
    for t in range(j.shape[0]):
        midhip = 0.5*(j[t, lh] + j[t, rh]); j[t] -= midhip
        sd = float(np.linalg.norm(j[t, ls, :2] - j[t, rs, :2]))
        j[t] /= (sd if sd > eps else 1.0)
    return j

def bone_vectors(joints: np.ndarray) -> np.ndarray:
    V = V_ALL
    P = np.full((V,), -1, dtype=np.int32)
    # pose parents
    for a,b in POSE_EDGES:
        if P[b] == -1 and a != b: P[b] = a
        if P[a] == -1 and b != a: P[a] = b
    # left hand parents (relative to left wrist root)
    for (u,v) in HAND_EDGES:
        gu, gv = V_POSE+u, V_POSE+v
        if P[gv] == -1: P[gv] = gu
    # right hand parents (relative to right wrist root)
    for (u,v) in HAND_EDGES:
        gu, gv = V_POSE+V_LHAND+u, V_POSE+V_LHAND+v
        if P[gv] == -1: P[gv] = gu
    P[LHAND_ROOT] = POSE_WRIST_L
    P[RHAND_ROOT] = POSE_WRIST_R
    bones = np.zeros_like(joints, dtype=np.float32)
    for v in range(V):
        p = int(P[v])
        bones[:, v, :] = joints[:, v, :] - (joints[:, p, :] if p >= 0 else 0.0)
    return bones

def flip_lr_and_swap(j: np.ndarray) -> np.ndarray:
    out = j.copy()
    out[..., 0] *= -1.0
    # swap pose LR pairs
    for a, b in POSE_SWAP_PAIRS:
        tmp = out[:, a].copy(); out[:, a] = out[:, b]; out[:, b] = tmp
    # swap hands
    L = out[:, V_POSE:V_POSE+V_LHAND].copy()
    R = out[:, V_POSE+V_LHAND:V_POSE+V_LHAND+V_RHAND].copy()
    out[:, V_POSE:V_POSE+V_LHAND] = R
    out[:, V_POSE+V_LHAND:V_POSE+V_LHAND+V_RHAND] = L
    return out

# ---------------------- CTR-GCN ----------------------
class CTRGCNBlock(nn.Module):
    def __init__(self, C_in, C_out, V=V_ALL, kernel_t=9, stride=1, dropout=0.3):
        super().__init__()
        self.theta = nn.Conv2d(C_in, C_out//4, 1)
        self.phi   = nn.Conv2d(C_in, C_out//4, 1)
        self.g     = nn.Conv2d(C_in, C_out,   1)

        A = np.eye(V, dtype=np.float32)
        for (a,b) in POSE_EDGES: A[a,b]=A[b,a]=1
        for (u,v) in HAND_EDGES:
            gu, gv = V_POSE+u, V_POSE+v; A[gu,gv]=A[gv,gu]=1
            gu, gv = V_POSE+V_LHAND+u, V_POSE+V_LHAND+v; A[gu,gv]=A[gv,gu]=1
        # connect wrists to hand roots
        A[POSE_WRIST_L, LHAND_ROOT] = A[LHAND_ROOT, POSE_WRIST_L] = 1
        A[POSE_WRIST_R, RHAND_ROOT] = A[RHAND_ROOT, POSE_WRIST_R] = 1
        D = np.sum(A, 1, keepdims=True) + 1e-6
        A = A / np.sqrt(D @ D.T)
        self.register_buffer("A_base", torch.from_numpy(A))
        self.A_learn = nn.Parameter(torch.zeros(V, V))
        nn.init.uniform_(self.A_learn, -0.01, 0.01)

        pad = (kernel_t - 1)//2
        self.tconv = nn.Conv2d(C_out, C_out, kernel_size=(kernel_t,1), stride=(stride,1), padding=(pad,0))
        self.bn = nn.BatchNorm2d(C_out)
        self.drop = nn.Dropout(dropout)
        self.res = None
        if C_in != C_out or stride != 1:
            self.res = nn.Sequential(nn.Conv2d(C_in, C_out, 1, stride=(stride,1)), nn.BatchNorm2d(C_out))

    def forward(self, x):  # (N,C,T,V)
        q = self.theta(x).mean(2, keepdim=True)
        k = self.phi(x).mean(2, keepdim=True)
        attn = torch.einsum("nctv,nctw->nvw", q, k) / math.sqrt(q.shape[1] + 1e-6)
        A_dyn = torch.softmax(attn, dim=-1)

        gx = self.g(x)
        y = torch.einsum("nctv,vw->nctw", gx, self.A_base + self.A_learn) \
          + torch.einsum("nctv,nvw->nctw", gx, A_dyn)
        y = self.tconv(y)
        r = x if self.res is None else self.res(x)
        return self.drop(F.relu(self.bn(y) + r, inplace=True))

class CTRGCN(nn.Module):
    def __init__(self, num_classes, c_in=9, channels=(64,128,256), blocks=(2,2,2), kernel_t=9, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c_in, channels[0], 1), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        layers = []
        cprev = channels[0]
        for i, (c, n) in enumerate(zip(channels, blocks)):
            for j in range(n):
                stride = 2 if (i>0 and j==0) else 1
                layers.append(CTRGCNBlock(cprev, c, V=V_ALL, kernel_t=kernel_t, stride=stride, dropout=dropout))
                cprev = c
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(cprev, num_classes)
    def forward(self, x):               # x: (N,C,T,V)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ---------------------- Loss / metrics ----------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.05): super().__init__(); self.eps = eps
    def forward(self, logits, target):
        n = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        one = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1.0)
        soft = (1-self.eps)*one + self.eps/n
        return -(soft * logp).sum(1).mean()

def _macro_f1_from_conf(conf: np.ndarray) -> float:
    C = conf.shape[0]; f1s=[]
    for c in range(C):
        tp = conf[c,c]; fp = conf[:,c].sum()-tp; fn = conf[c,:].sum()-tp
        supp = conf[c,:].sum()
        if supp == 0: continue
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

@torch.no_grad()
def evaluate(model, loader, feat_conf: dict, device, ls_eps=0.05):
    """
    If feat_conf['val_bihand_both'] = True:
        For each batch:
          - build original features
          - build deterministically flipped features
          - average probabilities, compute metrics; loss is the mean of the two CE losses.
    Otherwise (default):
        Evaluate on original features only.
    """
    model.eval()
    crit = LabelSmoothingCE(ls_eps)
    loss_sum, correct, total = 0.0, 0, 0
    num_classes = feat_conf["num_classes"]
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    use_both = bool(feat_conf.get("val_bihand_both", False))

    for X, Y, L in loader:
        X, Y = X.to(device), Y.to(device)

        # Build ORIGINAL input
        conf_o = dict(feat_conf); conf_o["phase"]="val"; conf_o["do_bihand"]=False; conf_o["force_flip"]=False
        xb_o = build_ctrgcn_input(X, L, conf_o)

        if use_both:
            # Build FLIPPED input deterministically (no randomness)
            conf_f = dict(conf_o); conf_f["force_flip"]=True
            xb_f = build_ctrgcn_input(X, L, conf_f)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda" and feat_conf["amp"])):
                logits_o = model(xb_o["x"])
                logits_f = model(xb_f["x"])
                loss = 0.5*(crit(logits_o, Y) + crit(logits_f, Y))
                prob = 0.5*(torch.softmax(logits_o, dim=1) + torch.softmax(logits_f, dim=1))
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda" and feat_conf["amp"])):
                logits_o = model(xb_o["x"])
                loss = crit(logits_o, Y)
                prob = torch.softmax(logits_o, dim=1)

        loss_sum += float(loss.item()) * Y.size(0)
        pred = prob.argmax(1)
        correct += (pred == Y).sum().item()
        total   += Y.numel()
        for t, p in zip(Y.cpu().numpy(), pred.cpu().numpy()):
            conf[int(t), int(p)] += 1

    return {"loss": loss_sum / max(1,total),
            "acc":  correct / max(1,total),
            "macro_f1": _macro_f1_from_conf(conf)}

def better_by_f1_then_loss(curr, best):
    if best is None: return True
    if curr["macro_f1"] > best["macro_f1"] + 1e-12: return True
    if abs(curr["macro_f1"] - best["macro_f1"]) <= 1e-12 and curr["loss"] < best["loss"] - 1e-12: return True
    return False

# ---------------------- Batch feature builder ----------------------
def build_ctrgcn_input(Xb: torch.Tensor, Lb: torch.Tensor, cfg: dict) -> Dict[str, torch.Tensor]:
    """
    Xb: (B,T,258). Returns { x: (B,C,T,V) } where C in {3,6,9}.
    Steps: slice→(optional center/scale)→(optional bi-hand flip)→concat (bones/vel)→permute.

    Flipping rules:
      • TRAIN: if cfg["do_bihand"] and random()<cfg["bihand_p"], flip.
      • VAL  : if cfg["force_flip"] is True, always flip; otherwise never flip.
    """
    B, T, _ = Xb.shape; device = Xb.device
    joints_list = []
    for i in range(B):
        j = slice_to_joints_xyz(Xb[i].cpu().numpy())
        if cfg["normalize_body"]:
            j = body_center_scale(j)
        do_flip = False
        if cfg.get("phase") == "train" and cfg.get("do_bihand", False):
            p = float(cfg.get("bihand_p", 0.0))
            do_flip = (random.random() < p)
        elif cfg.get("phase") != "train" and cfg.get("force_flip", False):
            do_flip = True
        if do_flip:
            j = flip_lr_and_swap(j)
        joints_list.append(j)
    joints = torch.from_numpy(np.stack(joints_list, 0)).to(device)  # (B,T,75,3)

    feats = [joints]
    if cfg["use_bones"]:
        bones_list = [bone_vectors(joints[i].cpu().numpy()) for i in range(B)]
        bones = torch.from_numpy(np.stack(bones_list, 0)).to(device)
        feats.append(bones)
    if cfg["use_vel"]:
        vel = torch.zeros_like(joints)
        vel[:, 1:] = joints[:, 1:] - joints[:, :-1]
        feats.append(vel)

    x = torch.cat(feats, dim=-1).permute(0,3,1,2).contiguous()  # (B,C,T,V)
    assert x.shape[1] in (3,6,9), f"Unexpected C_in={x.shape[1]} (expected 3/6/9)"
    return {"x": x}

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder that contains label_to_id.json & index_*.csv (augment output root)")
    ap.add_argument("--save", type=str, default="", help="Output folder (auto-selected if --data is under data/include_*/top_*/aug_keypoints)")
    ap.add_argument("--bihand", action="store_true", help="Enable bi-handed training (random left/right flip+swap). Avoids canonicalisation.")
    ap.add_argument("--bihand_p", type=float, default=0.5, help="Flip probability during TRAIN when --bihand")
    ap.add_argument("--bihand_ramp_epoch", type=int, default=0,
                    help="Linearly ramp flip prob from 0→bihand_p over the first E epochs (0 disables ramp).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--kernel_t", type=int, default=9)
    ap.add_argument("--normalize_body", action="store_true")
    ap.add_argument("--use_bones", action="store_true")
    ap.add_argument("--use_vel", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ls_eps", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="", help="Path to resume, or 'auto' to pick ckpt_last.pt if present")
    args = ap.parse_args()

    # Repro
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data).resolve()
    # Decide save root
    model_folder = "ctr_gcn_bihand" if args.bihand else "ctr_gcn"
    if args.save:
        save_root = Path(args.save).resolve()
    else:
        auto = auto_save_dir(data_root, model_folder)
        if auto is None:
            raise SystemExit("Could not auto-select save folder (data not under data/include_*/top_*/aug_keypoints). Please pass --save.")
        save_root = auto
    safe_mkdir(save_root)
    last_path = save_root / "ckpt_last.pt"
    best_path = save_root / "ckpt_best.pt"
    log_csv   = save_root / "log.csv"

    # Labels
    l2id = json.loads((data_root / "label_to_id.json").read_text(encoding="utf-8"))
    num_classes = len(l2id)

    # Splits
    tr_paths, tr_labels = load_index(data_root / "index_train.csv", data_root)
    va_paths, va_labels = load_index(data_root / "index_val.csv",   data_root)
    if not tr_paths or not va_paths:
        raise SystemExit("No training/validation data found in --data. Run augment first.")

    train_loader = DataLoader(NpzSeq(tr_paths, tr_labels), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=collate)
    val_loader   = DataLoader(NpzSeq(va_paths, va_labels), batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=collate)

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type.upper()}" + (f": {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    c_in = 3 + (3 if args.use_bones else 0) + (3 if args.use_vel else 0)
    model = CTRGCN(num_classes, c_in=c_in, channels=(64,128,256), blocks=(2,2,2), kernel_t=args.kernel_t, dropout=args.dropout).to(device)

    # Optim & sched
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type=="cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9,0.999))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup = max(100, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # Save params
    params = {
        "model": "ctr_gcn" + ("_bihand" if args.bihand else ""),
        "num_classes": num_classes, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "wd": args.wd, "dropout": args.dropout, "kernel_t": args.kernel_t,
        "normalize_body": bool(args.normalize_body),
        "use_bones": bool(args.use_bones), "use_vel": bool(args.use_vel),
        "bihand": bool(args.bihand), "bihand_p": float(args.bihand_p),
        "bihand_ramp_epoch": int(args.bihand_ramp_epoch),
        "amp": bool(args.amp), "ls_eps": float(args.ls_eps), "grad_clip": float(args.grad_clip),
        "data_root": str(data_root), "save_root": str(save_root)
    }
    (save_root / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    # (Optional) resume
    start_epoch = 1
    if args.resume:
        rpath: Optional[Path] = None
        if args.resume.lower() == "auto":
            if last_path.exists(): rpath = last_path
        else:
            rp = Path(args.resume)
            if rp.exists(): rpath = rp
        if rpath is not None:
            print(f"[RESUME] Loading {rpath}")
            ck = torch.load(rpath, map_location="cpu")
            model.load_state_dict(ck["model_state"], strict=True)
            if "opt_state" in ck:
                try: optim.load_state_dict(ck["opt_state"])
                except Exception: pass
            if "sched_state" in ck:
                try: sched.load_state_dict(ck["sched_state"])
                except Exception: pass
            if "epoch" in ck:
                start_epoch = int(ck["epoch"]) + 1
            print(f"[RESUME] start_epoch={start_epoch}")

    # Shared feature config
    base_conf = {
        "normalize_body": bool(args.normalize_body),
        "use_bones": bool(args.use_bones),
        "use_vel": bool(args.use_vel),
        "num_classes": int(num_classes),
        "amp": bool(args.amp),
        # training-time flip flags filled per-epoch below
        "do_bihand": bool(args.bihand),
        "bihand_p": float(args.bihand_p),
        # eval-time flip control
        "force_flip": False,
        "phase": "train",
        "val_bihand_both": False,  # set True when --bihand is on
    }

    best: Optional[Dict[str, float]] = None
    logs = []

    for epoch in range(start_epoch, args.epochs + 1):
        # Compute current flip probability with ramp (if enabled)
        if args.bihand and args.bihand_ramp_epoch > 0:
            if epoch <= args.bihand_ramp_epoch:
                p_curr = args.bihand_p * (epoch / float(max(1, args.bihand_ramp_epoch)))
            else:
                p_curr = args.bihand_p
        else:
            p_curr = args.bihand_p

        # ---- Train ----
        model.train()
        crit = LabelSmoothingCE(args.ls_eps)
        loss_sum, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"train {epoch:03d}", leave=False)
        for X, Y, L in pbar:
            X, Y, L = X.to(device), Y.to(device), L.to(device)
            conf = dict(base_conf)
            conf["phase"] = "train"
            conf["bihand_p"] = float(p_curr)  # ramped probability
            optim.zero_grad(set_to_none=True)
            xb = build_ctrgcn_input(X, L, conf)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type=="cuda")):
                logits = model(xb["x"])
                loss = crit(logits, Y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim); scaler.update(); sched.step()
            loss_sum += float(loss.item()) * Y.size(0)
            correct  += (logits.argmax(1) == Y).sum().item()
            total    += Y.numel()
        tr_loss = loss_sum / max(1,total)
        tr_acc  = correct / max(1,total)

        # ---- Val ----
        conf = dict(base_conf)
        conf["phase"] = "val"
        conf["do_bihand"] = False
        conf["force_flip"] = False
        conf["val_bihand_both"] = bool(args.bihand)  # enforce BOTH when bihand training is enabled
        va = evaluate(model, val_loader, conf, device, ls_eps=args.ls_eps)

        logs.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va["loss"], "val_acc": va["acc"], "val_f1": va["macro_f1"],
                     "bihand_p_curr": float(p_curr) if args.bihand else 0.0})
        pd.DataFrame(logs).to_csv(log_csv, index=False)
        print(f"[{epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va['loss']:.4f} acc={va['acc']:.4f} f1={va['macro_f1']:.4f}"
              + (f" | flip_p={p_curr:.3f}" if args.bihand else ""))

        # Save last
        ck = {"model_state": model.state_dict(), "opt_state": optim.state_dict(),
              "sched_state": sched.state_dict(), "epoch": epoch, "params": params}
        torch.save(ck, last_path)

        # Save best (macro-F1 then loss)
        if better_by_f1_then_loss(va, best):
            best = {"macro_f1": va["macro_f1"], "loss": va["loss"]}
            torch.save(ck, best_path)
            print(f"[BEST] f1={va['macro_f1']:.6f} loss={va['loss']:.6f} → {best_path}")

    print(f"[DONE] Best checkpoint at: {best_path}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import cv2; cv2.setNumThreads(1)
    except Exception:
        pass
    main()
