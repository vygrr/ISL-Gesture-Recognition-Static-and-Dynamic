#!/usr/bin/env python3
"""
eval.py — Evaluate CTR-GCN / LSTM / BiLSTM+Attn / RelPos Transformer on augmented keypoint NPZs.

Usage:
  python eval.py --data dynamic/data/include_50/aug_keypoints \
                 --ckpt dynamic/data/include_50/ctr_gcn/ckpt_best.pt \
                 --amp

Outputs (next to the checkpoint by default or in --out_dir):
  per_class_val.csv
  per_class_test.csv        (only if index_test.csv exists)
  metrics_summary.csv

Notes:
  • Model details (architecture & feature flags) are read from params next to the checkpoint,
    or from ckpt['params'] if present.
  • Batch size defaults to the value saved in params.json; you can override with --batch.
  • Robustly resolves sample paths in index_*.csv even if they were saved as old absolute paths.
  • If params['model'] is CTR-GCN and params['bihand'] is True, evaluation runs “both”
    (original + deterministically flipped) and averages probabilities.
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------- Constants / topology ----------------------
SEQ_LEN  = 200
FEAT_DIM = 258  # 33*4 (pose) + 21*3 (lhand) + 21*3 (rhand)
DTYPE    = np.float32

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
V_POSE, V_LHAND, V_RHAND = 33, 21, 21
V_ALL = V_POSE + V_LHAND + V_RHAND  # 75

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

# ---------------------- Robust CSV path loader ----------------------
def _infer_split_from_csv(csv_path: Path) -> str:
    n = csv_path.name.lower()
    if "train" in n: return "train"
    if "val"   in n: return "val"
    if "test"  in n: return "test"
    return ""

def _parse_from_raw_path(raw: str):
    p = Path(raw)
    filename = p.name
    label_dir = p.parent.name if p.parent.name.isdigit() else ""
    split_dir = ""
    if len(p.parents) >= 2:
        cand = p.parents[1].name.lower()
        if cand in ("train", "val", "test"): split_dir = cand
    return split_dir, label_dir, filename

def load_index(csv_path: Path, base: Path) -> Tuple[List[str], List[int]]:
    """
    Robustly resolves npz file paths recorded in index_*.csv, even if those paths
    were saved on a different machine.

    Heuristic per row:
      1) Use path as-is if absolute & exists.
      2) If relative, resolve under `base`.
      3) Try `base / filename`.
      4) Rebuild as `base / <split>/<label_dir>/<filename>` from saved path hints.
      5) Rebuild as `base / <split_from_csv>/<label_id or 03d>/<filename>`.
      6) Fallback `base / <split_from_csv>/<filename>`.
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

        candidates.append((base / raw).resolve())      # as relative
        candidates.append((base / filename).resolve())

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
class NpzSeqDataset(Dataset):
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

def collate_pad(batch):
    xs, ys, ls = zip(*batch)
    X = torch.from_numpy(np.stack(xs, 0))    # (B,T,258)
    Y = torch.tensor(ys, dtype=torch.long)
    L = torch.tensor(ls, dtype=torch.long)
    return X, Y, L

# ---------------------- Feature transforms (match train.py) ----------------------
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
    for a,b in POSE_EDGES:
        if P[b] == -1 and a != b: P[b] = a
        if P[a] == -1 and b != a: P[a] = b
    for (u,v) in HAND_EDGES:
        gu, gv = V_POSE+u, V_POSE+v
        if P[gv] == -1: P[gv] = gu
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
    for a, b in POSE_SWAP_PAIRS:
        tmp = out[:, a].copy(); out[:, a] = out[:, b]; out[:, b] = tmp
    L = out[:, V_POSE:V_POSE+V_LHAND].copy()
    R = out[:, V_POSE+V_LHAND:V_POSE+V_LHAND+V_RHAND].copy()
    out[:, V_POSE:V_POSE+V_LHAND] = R
    out[:, V_POSE+V_LHAND:V_POSE+V_LHAND+V_RHAND] = L
    return out

def build_batch_inputs(Xb: torch.Tensor, Lb: torch.Tensor, cfg: Dict, device) -> Dict[str, torch.Tensor]:
    """
    Returns:
      For CTR-GCN: {'x': (B,C,T,V)}
      For alt models: {'x': (B,T,D), 'mask': (B,T) bool}
    """
    B, T, _ = Xb.shape
    model = cfg["model"]
    if model == "ctr_gcn":
        joints_list = []
        for i in range(B):
            j = slice_to_joints_xyz(Xb[i].cpu().numpy())
            if cfg["normalize_body"]:
                j = body_center_scale(j)
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
        x = torch.cat(feats, dim=-1).permute(0,3,1,2).contiguous()      # (B,C,T,V)
        return {"x": x}
    else:
        X = Xb.to(device)
        mask = (torch.arange(T, device=device)[None, :] >= Lb[:, None].to(device))
        return {"x": X, "mask": mask}

# ---------------------- Models (mirror training code) ----------------------
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

class LSTMHead(nn.Module):
    def __init__(self, feat_dim, hidden, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden, 128)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, num_classes)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = h[-1]
        z = torch.tanh(self.fc1(h))
        z = self.drop(z)
        return self.out(z)

class BiLSTMAttn(nn.Module):
    def __init__(self, num_classes, hidden=128, layers=2, dropout=0.3, feat_dim=FEAT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=(dropout if layers>1 else 0.0))
        self.attn_W = nn.Linear(2*hidden, 2*hidden)
        self.attn_v = nn.Linear(2*hidden, 1, bias=False)
        self.fc = nn.Linear(2*hidden, 128)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, num_classes)
        nn.init.xavier_uniform_(self.fc.weight);  nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,2H)
        B, Tmax, D = H.shape
        device = H.device
        pad_mask = (torch.arange(Tmax, device=device)[None, :] >= lengths.to(device)[:, None])
        scores = self.attn_v(torch.tanh(self.attn_W(H))).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(pad_mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=1)
        ctx = (alpha.unsqueeze(-1) * H).sum(1)                         # (B,2H)
        z = torch.tanh(self.fc(ctx))
        z = self.drop(z)
        return self.out(z)

class RelPosMHA(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.2, max_len=512):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead; self.dk = d_model // nhead; self.max_len = max_len
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rel_bias = nn.Embedding(2*max_len-1, nhead)
    def forward(self, x, key_padding_mask=None):  # x: (B,T,D)
        B, T, D = x.shape
        q = self.q(x).view(B,T,self.nhead,self.dk).transpose(1,2)
        k = self.k(x).view(B,T,self.nhead,self.dk).transpose(1,2)
        v = self.v(x).view(B,T,self.nhead,self.dk).transpose(1,2)
        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.dk)
        pos = torch.arange(T, device=x.device)
        rel = torch.clamp(pos[None,:]-pos[:,None] + (self.max_len-1), 0, 2*self.max_len-2)
        bias = self.rel_bias(rel).permute(2,0,1).unsqueeze(0)
        attn = attn + bias
        if key_padding_mask is not None:
            m = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(m, float('-inf'))
        w = torch.softmax(attn, dim=-1)
        w = self.drop(w)
        out = torch.matmul(w, v).transpose(1,2).contiguous().view(B,T,D)
        return self.o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.2, max_len=512, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = RelPosMHA(d_model, nhead, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(nn.Linear(d_model, d_model*ffn_mult), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model*ffn_mult, d_model), nn.Dropout(dropout))
    def forward(self, x, key_padding_mask=None):
        x = x + self.mha(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x

class RelPosTransformer(nn.Module):
    def __init__(self, num_classes, d_in=FEAT_DIM, d_model=256, layers=6, nhead=8, dropout=0.2, max_len=512):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, dropout, max_len) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x, key_padding_mask=None):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        x = x.mean(1)
        return self.fc(x)

# ---------------------- Params & model builder ----------------------
def load_params_from_ckpt_and_sidecar(ckpt_path: Path) -> Dict:
    """
    Return params dict from ckpt['params'] (if present) or params.json next to the ckpt.
    """
    params = {}
    try:
        ck = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ck, dict) and "params" in ck and isinstance(ck["params"], dict):
            params = dict(ck["params"])
    except Exception:
        pass
    if not params:
        side = ckpt_path.parent / "params.json"
        if side.exists():
            try:
                params = json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                pass
    if not params:
        raise SystemExit("Could not find run params (neither in ckpt['params'] nor params.json next to --ckpt).")
    return params

def _normalize_model_name(name: str) -> str:
    n = (name or "").lower().strip()
    return {
        "ctr-gcn": "ctr_gcn",
        "ctr_gcn_bihand": "ctr_gcn",
        "bilstm+att": "bilstm_att",
        "bilstm_attn": "bilstm_att",
        "relpos_transformer": "relpos",
    }.get(n, n)

def build_model_from_params(params: Dict):
    model_name = _normalize_model_name(params.get("model", ""))
    num_classes = int(params["num_classes"])
    dropout = float(params.get("dropout", 0.3))
    if model_name == "ctr_gcn":
        c_in = 3 + (3 if params.get("use_bones", False) else 0) + (3 if params.get("use_vel", False) else 0)
        model = CTRGCN(num_classes, c_in=c_in, channels=(64,128,256), blocks=(2,2,2),
                       kernel_t=int(params.get("kernel_t", 9)), dropout=dropout)
    elif model_name == "lstm":
        model = LSTMHead(FEAT_DIM, hidden=int(params.get("hidden", 128)),
                         num_classes=num_classes, dropout=dropout)
    elif model_name == "bilstm_att":
        model = BiLSTMAttn(num_classes=num_classes, hidden=int(params.get("hidden", 128)),
                           layers=int(params.get("layers", 2)), dropout=dropout, feat_dim=FEAT_DIM)
    elif model_name == "relpos":
        model = RelPosTransformer(num_classes=num_classes, d_in=FEAT_DIM,
                                  d_model=int(params.get("d_model", 256)),
                                  layers=int(params.get("tr_layers", params.get("layers", 6))),
                                  nhead=int(params.get("nhead", 8)),
                                  dropout=float(params.get("dropout", 0.2)),
                                  max_len=max(SEQ_LEN, 512))
    else:
        raise SystemExit(f"Unknown model in params.json: '{params.get('model','')}' (normalized='{model_name}').")
    return model, model_name

# ---------------------- Metrics ----------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1

    total = conf.sum()
    acc = (np.trace(conf) / total) if total > 0 else 0.0

    per_class = []
    f1s = []
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        supp = conf[c, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        per_class.append((supp, tp / supp if supp > 0 else 0.0, prec, rec, f1))
        if supp > 0: f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return acc, macro_f1, conf, per_class

# ---------------------- Eval loop ----------------------
@torch.no_grad()
def run_split(split_name: str, csv_path: Path, data_root: Path, loader_args: Dict,
              device, model, model_name: str, params: Dict,
              id_to_label: Optional[Dict[int,str]], out_dir: Path):
    if not csv_path.exists():
        print(f"[{split_name}] CSV missing → skip: {csv_path}")
        return None

    paths, labels = load_index(csv_path, data_root)
    if len(paths) == 0:
        print(f"[{split_name}] No items resolved from {csv_path.name} → skip.")
        return None

    ds = NpzSeqDataset(paths, labels)
    loader = DataLoader(ds, batch_size=loader_args["batch"], shuffle=False,
                        num_workers=loader_args["workers"], pin_memory=True,
                        collate_fn=collate_pad, drop_last=False)
    use_amp = loader_args["amp"] and (device.type == "cuda")
    autocast = torch.amp.autocast if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast

    # Feature config
    feat_cfg = {
        "model": model_name,
        "normalize_body": bool(params.get("normalize_body", False)),
        "use_bones": bool(params.get("use_bones", False)) if model_name == "ctr_gcn" else False,
        "use_vel": bool(params.get("use_vel", False)) if model_name == "ctr_gcn" else False,
    }
    bihand_both = (model_name == "ctr_gcn") and bool(params.get("bihand", False))

    y_true, y_pred = [], []
    model.eval()

    for X, Y, L in tqdm(loader, desc=f"eval:{split_name}", leave=False):
        X, Y, L = X.to(device), Y.to(device), L.to(device)

        if model_name == "ctr_gcn" and bihand_both:
            # ORIGINAL
            batch_o = build_batch_inputs(X, L, feat_cfg, device)
            # FLIPPED (deterministic)
            B, T, _ = X.shape
            joints_list = []
            for i in range(B):
                j = slice_to_joints_xyz(X[i].cpu().numpy())
                if feat_cfg["normalize_body"]: j = body_center_scale(j)
                j = flip_lr_and_swap(j)
                joints_list.append(j)
            joints_f = torch.from_numpy(np.stack(joints_list, 0)).to(device)
            feats_f = [joints_f]
            if feat_cfg["use_bones"]:
                bones_list = [bone_vectors(joints_f[i].cpu().numpy()) for i in range(B)]
                bones = torch.from_numpy(np.stack(bones_list, 0)).to(device); feats_f.append(bones)
            if feat_cfg["use_vel"]:
                vel = torch.zeros_like(joints_f); vel[:,1:] = joints_f[:,1:] - joints_f[:,:-1]; feats_f.append(vel)
            x_f = torch.cat(feats_f, dim=-1).permute(0,3,1,2).contiguous()

            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits_o = model(batch_o["x"])
                logits_f = model(x_f)
                prob = 0.5*(torch.softmax(logits_o, dim=1) + torch.softmax(logits_f, dim=1))
            pred = prob.argmax(1)
        else:
            batch = build_batch_inputs(X, L, feat_cfg, device)
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if model_name == "ctr_gcn":
                    logits = model(batch["x"])
                elif model_name in ("lstm","bilstm_att"):
                    logits = model(batch["x"], L)
                else:  # relpos
                    logits = model(batch["x"], key_padding_mask=batch["mask"])
            pred = logits.argmax(1)

        y_true.append(Y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)
    num_classes = int(params["num_classes"])

    acc, macro_f1, conf, per_class = compute_metrics(y_true, y_pred, num_classes)
    print(f"[{split_name}] Acc={acc:.4f}  Macro-F1={macro_f1:.4f}  N={len(y_true)}")

    # Per-class CSV
    if id_to_label is None:
        id_to_label = {i: str(i) for i in range(num_classes)}
    rows = []
    for cid in range(num_classes):
        supp, acc_c, prec, rec, f1 = per_class[cid]
        rows.append({
            "class_id": cid,
            "class_label": id_to_label.get(cid, str(cid)),
            "support": int(supp),
            "acc": float(acc_c),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })
    df_pc = pd.DataFrame(rows).sort_values("class_id")
    out_pc = out_dir / f"per_class_{split_name}.csv"
    df_pc.to_csv(out_pc, index=False)
    print(f"[OUT] {split_name}: per-class metrics → {out_pc}")

    return {"split": split_name, "acc": acc, "macro_f1": macro_f1, "n": len(y_true)}

# ---------------------- Checkpoint resolver ----------------------
def resolve_ckpt(ckpt_arg: Path) -> Path:
    if ckpt_arg.is_file():
        return ckpt_arg
    if ckpt_arg.is_dir():
        for name in ("ckpt_best.pt", "ckpt_last.pt"):
            p = ckpt_arg / name
            if p.exists(): return p
    raise FileNotFoundError(f"Could not resolve checkpoint from: {ckpt_arg}")

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Augmented data folder with label_to_id.json and index_{val,test}.csv")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint file OR a run directory containing it")
    ap.add_argument("--out_dir", type=str, default=None, help="Output folder (default: alongside checkpoint)")
    ap.add_argument("--batch", type=int, default=None, help="Override batch size (default: params.json['batch'])")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data).resolve()
    ckpt_path = resolve_ckpt(Path(args.ckpt).resolve())

    # Load params first to pick default batch size
    params = load_params_from_ckpt_and_sidecar(ckpt_path)
    default_batch = int(params.get("batch", 128))
    batch_size = args.batch if args.batch is not None else default_batch

    out_dir = Path(args.out_dir).resolve() if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Labels (readable class names)
    l2id_path = data_root / "label_to_id.json"
    if l2id_path.exists():
        l2id = json.loads(l2id_path.read_text(encoding="utf-8"))
        id_to_label = {int(v): k for k, v in l2id.items()}
    else:
        id_to_label = None

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type.upper()}" + (f": {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    model, model_name = build_model_from_params(params)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model_state"], strict=False)
    model = model.to(device).eval()
    print(f"[Model] {model_name} | normalize_body={params.get('normalize_body', False)} "
          f"use_bones={params.get('use_bones', False)} use_vel={params.get('use_vel', False)} "
          f"| bihand={params.get('bihand', False)} | batch={batch_size}")

    results = []
    # Evaluate VAL
    results.append(run_split("val",  data_root / "index_val.csv",  data_root,
                             {"batch": batch_size, "workers": args.workers, "amp": args.amp},
                             device, model, model_name, params, id_to_label, out_dir))
    # Evaluate TEST (if present)
    results.append(run_split("test", data_root / "index_test.csv", data_root,
                             {"batch": batch_size, "workers": args.workers, "amp": args.amp},
                             device, model, model_name, params, id_to_label, out_dir))
    results = [r for r in results if r is not None]

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "metrics_summary.csv", index=False)
        print(f"[OUT] Summary → {out_dir / 'metrics_summary.csv'}")
    else:
        print("[INFO] No splits evaluated (missing CSVs or no samples).")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import cv2; cv2.setNumThreads(1)
    except Exception:
        pass
    main()
