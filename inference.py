#!/usr/bin/env python3
"""
root_inference.py — Unified realtime inference for ISL (Static MLP + Dynamic models)

New in this build
  • Context-aware sentence formation:
      - Detects spelled named entities (from static buffer) and applies subject–verb agreement:
          I + ADITYA  →  "I am Aditya."
      - Adds minimal glue words only where needed (am/is/are + my/our/your for kinship nouns):
          He + brother → "He is my brother."
          (uses 'our' if 'we' appears in context, 'your' if 'you' appears)
      - Uses spaCy small (if available) for POS/NER; otherwise falls back to rules.
  • The **formed sentence is drawn on the video window** (live preview each frame + a “Final:” line when you press `s`).
  • Kept: static spelling buffer + special labels (E1/E2 → E for spelling; 9a/9b → 9), dynamic label cleanup (strip "N. " prefixes),
    idle/active auto-windowing with pre/post context, flip toggle, text color toggle, bold/wrapped overlay, performance opts.

Hotkeys (complete)
  q: quit
  t: toggle AUTO/MANUAL windowing
  a: toggle Active (AUTO only)  | starts OFF (idle)
  f: toggle Flip (mirror)
  g: toggle overlay font color (white ↔ black)
  1: STATIC token mode
  2: DYNAMIC token mode
  m: toggle static Alphabet ↔ Numeral
  Space:
    - MANUAL + recording: start/stop the window
    - Otherwise: commit current spelled word (insert a word boundary)
  Enter: accept last prediction again
  b: backspace (pop last char from spelling buffer, else last token)
  c: clear all tokens and spelling buffer
  s: finalize sentence (stores “Final:” line on-screen)

Default dynamic paths
  labels: dynamic/data/top_100/aug_keypoints/label_to_id.json
  ckpt+params: dynamic/data/top_100/ctr_gcn/{ckpt_best.pt, params.json}
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# ---------- MediaPipe ----------
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_hands    = mp.solutions.hands
except Exception as e:
    raise SystemExit("MediaPipe required (pip install mediapipe>=0.10): " + str(e))

# ================= STATIC (MLP) =================
class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class StaticPredictor:
    def __init__(self, static_dir: Path, domain: str, device: torch.device):
        assert domain in ("Alphabet", "Numeral")
        enc = static_dir/"data/encoder"/("alphabets_le.pkl" if domain=="Alphabet" else "numerals_le.pkl")
        pth = static_dir/"data/model"/("alphabets_model.pth" if domain=="Alphabet" else "numerals_model.pth")
        if not enc.exists() or not pth.exists():
            raise FileNotFoundError(f"Missing static artifacts for {domain}: {enc} / {pth}")
        self.encoder = joblib.load(str(enc))
        classes = getattr(self.encoder, 'classes_', None)
        if classes is None:
            raise RuntimeError(f"Encoder at {enc} has no classes_.")
        self.model = MLP(input_dim=126, num_classes=len(classes)).to(device)
        state = torch.load(pth, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(X.astype(np.float32)).to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=1).float().cpu().numpy()

    def id_to_label(self, idx: int) -> str:
        return self.encoder.inverse_transform([idx])[0]

# =============== DYNAMIC (Models) ===============
SEQ_LEN  = 200
FEAT_DIM = 258  # 33*4 + 21*3 + 21*3
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

# ---- dynamic feature helpers ----
def extract_258_from_holistic(frame_bgr, holistic) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = holistic.process(rgb)
    vec = []
    if res.pose_landmarks and res.pose_landmarks.landmark:
        for lm in res.pose_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        vec.extend([0.0]*(33*4))
    if res.left_hand_landmarks and res.left_hand_landmarks.landmark:
        for lm in res.left_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0]*(21*3))
    if res.right_hand_landmarks and res.right_hand_landmarks.landmark:
        for lm in res.right_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0]*(21*3))
    arr = np.asarray(vec, dtype=DTYPE)
    if arr.shape[0] != FEAT_DIM:
        if arr.shape[0] > FEAT_DIM: arr = arr[:FEAT_DIM]
        else: arr = np.pad(arr, (0, FEAT_DIM - arr.shape[0]), mode="constant")
    return arr

def slice_to_joints_xyz(x_258: np.ndarray) -> np.ndarray:
    T = x_258.shape[0]
    pose  = x_258[:, :33*4].reshape(T, 33, 4)[..., :3]
    lhand = x_258[:, 33*4 : 33*4 + 21*3].reshape(T, 21, 3)
    rhand = x_258[:, 33*4 + 21*3 : ].reshape(T, 21, 3)
    return np.concatenate([pose, lhand, rhand], axis=1).astype(np.float32)  # (T,75,3)

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
    P[LHAND_ROOT] = POSE_WRIST_L; P[RHAND_ROOT] = POSE_WRIST_R
    bones = np.zeros_like(joints)
    for v in range(V):
        p = int(P[v])
        bones[:, v, :] = joints[:, v, :] - (joints[:, p, :] if p >= 0 else 0.0)
    return bones

# ---- CTR-GCN ----
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
        self.register_buffer("A_base", torch.from_numpy(A).float())
        self.A_learn = nn.Parameter(torch.zeros_like(self.A_base))
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
    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ---- Optional sequence models ----
class LSTMHead(nn.Module):
    def __init__(self, feat_dim, hidden, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden, 128)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, num_classes)
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
    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_packed_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        B, Tmax, D = H.shape
        device = H.device
        pad_mask = (torch.arange(Tmax, device=device)[None, :] >= lengths.to(device)[:, None])
        scores = self.attn_v(torch.tanh(self.attn_W(H))).squeeze(-1)
        scores = scores.masked_fill(pad_mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=1)
        ctx = (alpha.unsqueeze(-1) * H).sum(1)
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
    def forward(self, x, key_padding_mask=None):
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

# ---- params + builder ----
def load_params_from_ckpt_and_sidecar(ckpt_path: Path) -> Dict:
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
            params = json.loads(side.read_text(encoding="utf-8"))
    if not params:
        raise SystemExit("Could not find run params (neither in ckpt['params'] nor params.json next to ckpt).")
    return params

def _normalize_model_name(name: str) -> str:
    n = (name or "").lower().strip()
    aliases = {"ctr-gcn":"ctr_gcn", "ctr_gcn_bihand":"ctr_gcn",
               "bilstm+att":"bilstm_att", "bilstm_attn":"bilstm_att",
               "relpos_transformer":"relpos"}
    return aliases.get(n, n)

def build_model_from_params(num_classes: int, params: Dict):
    model_name_raw = params.get("model", "")
    model_name = _normalize_model_name(model_name_raw)
    if model_name == "ctr_gcn":
        c_in = 3 + (3 if params.get("use_bones", False) else 0) + (3 if params.get("use_vel", False) else 0)
        model = CTRGCN(num_classes, c_in=c_in, channels=(64,128,256), blocks=(2,2,2),
                       kernel_t=int(params.get("kernel_t", 9)), dropout=float(params.get("dropout", 0.3)))
    elif model_name == "lstm":
        model = LSTMHead(FEAT_DIM, hidden=int(params.get("hidden", 128)),
                         num_classes=num_classes, dropout=float(params.get("dropout", 0.3)))
    elif model_name == "bilstm_att":
        model = BiLSTMAttn(num_classes=num_classes, hidden=int(params.get("hidden", 128)),
                           layers=int(params.get("layers", 2)), dropout=float(params.get("dropout", 0.3)),
                           feat_dim=FEAT_DIM)
    elif model_name == "relpos":
        model = RelPosTransformer(num_classes=num_classes, d_in=FEAT_DIM,
                                  d_model=int(params.get("d_model", 256)),
                                  layers=int(params.get("tr_layers", params.get("layers", 6))),
                                  nhead=int(params.get("nhead", 8)), dropout=float(params.get("dropout", 0.2)),
                                  max_len=max(SEQ_LEN, 512))
    else:
        raise SystemExit(f"Unknown model in params.json: '{model_name_raw}' (normalized='{model_name}').")
    return model, model_name

def impute_short_gaps(seq: np.ndarray, max_gap: int = 5) -> np.ndarray:
    x = seq.copy()
    POSE_DIM = 33*4; HAND_DIM = 21*3
    L_SLICE  = slice(POSE_DIM, POSE_DIM + HAND_DIM)
    R_SLICE  = slice(POSE_DIM + HAND_DIM, POSE_DIM + 2*HAND_DIM)
    for slc in (L_SLICE, R_SLICE):
        sub = x[:, slc]
        miss = (np.abs(sub).sum(axis=1) == 0.0)
        if not miss.any():
            continue
        for c in range(sub.shape[1]):
            vals = sub[:, c]
            i = 0
            while i < len(vals):
                if miss[i]:
                    j = i
                    while j < len(vals) and miss[j]:
                        j += 1
                    gap = j - i
                    if gap <= max_gap:
                        li = i - 1; ri = j
                        left_ok  = (li >= 0 and not miss[li])
                        right_ok = (ri < len(vals) and not miss[ri])
                        if left_ok and right_ok:
                            v0, v1 = vals[li], vals[ri]
                            for k in range(gap):
                                vals[i+k] = v0 + (v1 - v0) * ((k+1)/(gap+1))
                        elif left_ok:
                            vals[i:j] = vals[li]
                        elif right_ok:
                            vals[i:j] = vals[ri]
                    i = j
                else:
                    i += 1
            sub[:, c] = vals
        x[:, slc] = sub
    return x

def build_inputs_for_model(seq_258: np.ndarray, valid_len: int, model_name: str, params: Dict, device, half: bool):
    T = seq_258.shape[0]
    if model_name == "ctr_gcn":
        j = slice_to_joints_xyz(seq_258)            # (T,75,3)
        if params.get("normalize_body", False):
            j = body_center_scale(j)
        feats = [j]
        if params.get("use_bones", False):
            feats.append(bone_vectors(j))           # (T,75,3)
        if params.get("use_vel", False):
            v = np.zeros_like(j); v[1:] = j[1:] - j[:-1]
            feats.append(v)
        x = np.concatenate(feats, axis=-1)          # (T,75,C)
        xt = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).contiguous().to(device)  # (1,C,T,V=75)
        if half: xt = xt.half()
        return {"x": xt, "mask": None, "lengths": torch.tensor([valid_len], device=device)}
    else:
        X = torch.from_numpy(seq_258).unsqueeze(0).to(device)  # (1,T,258)
        if half: X = X.half()
        mask = torch.arange(T, device=device)[None,:] >= torch.tensor([valid_len], device=device)[:,None]
        return {"x": X, "mask": mask, "lengths": torch.tensor([valid_len], device=device)}

@dataclass
class DynPredictor:
    id_to_label: Dict[int, str]
    model: Any
    device: torch.device
    params: Dict[str, Any]
    model_name: str
    use_half: bool = False
    @torch.inference_mode()
    def predict(self, seq_258: np.ndarray, seq_len: int) -> Tuple[str, float]:
        L = min(seq_258.shape[0], seq_len)
        x = seq_258[:L].astype(np.float32)
        x = impute_short_gaps(x, max_gap=5)
        if L < seq_len:
            pad = np.zeros((seq_len - L, seq_258.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
        batch = build_inputs_for_model(x, L, self.model_name, self.params, self.device, self.use_half)
        if self.model_name == "ctr_gcn":
            logits = self.model(batch["x"])  # (1,C,T,V)
        elif self.model_name in ("lstm","bilstm_att"):
            logits = self.model(batch["x"], batch["lengths"])  # (1,T,258)
        else:
            logits = self.model(batch["x"], key_padding_mask=batch["mask"])  # relpos
        prob = torch.softmax(logits.float(), dim=1)[0]
        top = int(prob.argmax().item())
        return self.id_to_label[top], float(prob[top].item())

def clean_dyn_label(lbl: str) -> str:
    # Remove leading "N. " if present (e.g., "12. Word" -> "Word")
    return re.sub(r"^\s*\d+\.\s*", "", lbl).strip()

def build_dynamic(aug_root: Path, ctr_dir: Path, force_cpu: bool=False, use_half: bool=False) -> DynPredictor:
    # labels
    label_json = aug_root/"label_to_id.json"
    if not label_json.exists():
        raise FileNotFoundError(f"label_to_id.json not found at {label_json}")
    label_to_id_raw = json.loads(label_json.read_text(encoding="utf-8"))
    id_to_label = {int(v): clean_dyn_label(k) for k, v in label_to_id_raw.items()}

    # params & checkpoint
    ckpt = ctr_dir/"ckpt_best.pt"
    if not ckpt.exists():
        cands = list(ctr_dir.rglob("ckpt_best.pt"))
        if not cands:
            raise FileNotFoundError(f"ckpt_best.pt not found under {ctr_dir}")
        ckpt = cands[0]
    params = load_params_from_ckpt_and_sidecar(ckpt)

    device = torch.device("cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"[Dynamic device] {device}")

    model, model_name = build_model_from_params(num_classes=len(id_to_label), params=params)
    model = model.to(device).eval()

    ck = torch.load(ckpt, map_location=device)
    if "model_state" in ck:
        sd = ck["model_state"]
    else:
        sd = ck.get("state_dict", ck)
    model.load_state_dict(sd, strict=False)
    if use_half and device.type == 'cuda':
        try:
            model = model.half()
        except Exception:
            pass

    return DynPredictor(id_to_label=id_to_label, model=model, device=device, params=params,
                        model_name=model_name, use_half=(use_half and device.type=='cuda'))

# =========== FEATURES & WINDOWS ===========
@dataclass
class FrameBundle:
    ts: float
    frame_disp: np.ndarray      # full-res frame for display
    frame_proc: np.ndarray      # downscaled frame for processing (BGR)
    left_ok: bool
    right_ok: bool
    hands_126: Optional[np.ndarray]  # (126,) or None
    feat_258: Optional[np.ndarray] = None  # filled only when Holistic is run

def run_hands(frame_proc_bgr, hands_proc) -> Tuple[bool,bool,Optional[np.ndarray]]:
    rgb = cv2.cvtColor(frame_proc_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res_h = hands_proc.process(rgb)
    feats = {"Left": [0.0]*63, "Right": [0.0]*63}
    left_ok = right_ok = False
    if res_h.multi_hand_landmarks:
        for lm, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            side = handed.classification[0].label  # "Left" or "Right"
            coords = [c for p in lm.landmark for c in (p.x, p.y, p.z)]
            feats[side] = coords
            if side == "Left": left_ok = True
            if side == "Right": right_ok = True
    hands_126 = np.array(feats["Left"] + feats["Right"], dtype=np.float32) if (left_ok or right_ok) else None
    return left_ok, right_ok, hands_126

def extract_features_lazy(hands_proc, frame_bgr: np.ndarray, proc_scale: float) -> FrameBundle:
    h, w = frame_bgr.shape[:2]
    if proc_scale != 1.0:
        frame_proc = cv2.resize(frame_bgr, (int(w*proc_scale), int(h*proc_scale)), interpolation=cv2.INTER_LINEAR)
    else:
        frame_proc = frame_bgr
    left_ok, right_ok, hands_126 = run_hands(frame_proc, hands_proc)
    return FrameBundle(ts=time.time(), frame_disp=frame_bgr, frame_proc=frame_proc,
                       left_ok=left_ok, right_ok=right_ok, hands_126=hands_126, feat_258=None)

def fill_feat_258_for_frames(frames: List[FrameBundle], hol) -> None:
    for fb in frames:
        if fb.feat_258 is None:
            fb.feat_258 = extract_258_from_holistic(fb.frame_proc, hol)

class WindowState:
    IDLE = 0
    RECORDING = 1

@dataclass
class WindowManager:
    mode_auto: bool = True
    min_len: int = 12
    max_len: int = 200
    on_thresh: int = 5
    off_thresh: int = 7
    pre_k: int = 0
    post_k: int = 0

    state: int = WindowState.IDLE
    on_count: int = 0
    off_count: int = 0
    buffer: List[FrameBundle] = field(default_factory=list)
    prebuf: deque = field(default_factory=lambda: deque(maxlen=120))
    post_remaining: int = 0

    def reset(self):
        self.state = WindowState.IDLE
        self.on_count = 0
        self.off_count = 0
        self.buffer.clear()
        self.post_remaining = 0

    def manual_toggle(self):
        if self.state == WindowState.IDLE:
            self.state = WindowState.RECORDING
            if self.pre_k and len(self.prebuf)>0:
                self.buffer = list(self.prebuf)[-self.pre_k:]
            else:
                self.buffer = []
            self.on_count = 0
            self.off_count = 0
        else:
            self.state = WindowState.IDLE

    def feed(self, fb: FrameBundle, active: bool) -> Optional[List[FrameBundle]]:
        hands_present = (fb.left_ok or fb.right_ok)
        finished = None
        self.prebuf.append(fb)

        if self.mode_auto:
            if not active:
                self.reset(); return None
            if self.state == WindowState.IDLE:
                if hands_present:
                    self.on_count += 1
                    if self.on_count >= self.on_thresh:
                        self.state = WindowState.RECORDING
                        if self.pre_k and len(self.prebuf)>0:
                            self.buffer = list(self.prebuf)[-self.pre_k:]
                        else:
                            self.buffer = []
                        self.buffer.append(fb)
                        self.off_count = 0
                else:
                    self.on_count = 0
            else:  # RECORDING
                self.buffer.append(fb)
                if not hands_present:
                    self.off_count += 1
                    if self.off_count >= self.off_thresh and self.post_remaining == 0:
                        self.post_remaining = self.post_k
                else:
                    self.off_count = 0
                if self.post_remaining > 0:
                    self.post_remaining -= 1
                    if self.post_remaining == 0 and len(self.buffer) >= self.min_len:
                        finished = self.buffer.copy(); self.reset()
                if len(self.buffer) >= self.max_len and finished is None:
                    finished = self.buffer.copy(); self.reset()
        else:
            if self.state == WindowState.RECORDING:
                self.buffer.append(fb)
                if len(self.buffer) >= self.max_len:
                    finished = self.buffer.copy(); self.reset()
        return finished

# ============ SENTENCE FORMATION (spaCy + rules) ============
FONT = cv2.FONT_HERSHEY_SIMPLEX

PRONOUNS = {"i","we","you","they","he","she","it"}
PRONOUN_CANON = {"i":"I", "we":"We", "you":"You", "they":"They", "he":"He", "she":"She", "it":"It"}
KINSHIP = {
    "brother","sister","mother","father","mom","dad","wife","husband",
    "son","daughter","uncle","aunt","cousin","grandfather","grandmother",
    "friend","teacher","boss"
}

def subj_verb(subj: str) -> str:
    s = subj.lower()
    if s == "i": return "am"
    if s in {"we","you","they"}: return "are"
    return "is"

def default_possessive(context_tokens: List[str]) -> str:
    ctx = {t.lower() for t in context_tokens}
    if "we" in ctx: return "our"
    if "you" in ctx: return "your"
    return "my"

class SentenceFormer:
    """Context-aware sentence builder using spaCy (if available) + rules."""
    def __init__(self, mode: str = "spacy"):
        self.mode = mode
        self.nlp = None
        if mode == "spacy":
            try:
                import spacy
                # try to load small English model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    # allow code to run even if the model isn't installed
                    self.nlp = spacy.blank("en")
                    print("[SentenceFormer] 'en_core_web_sm' not found. Using blank 'en' (rules-only quality).")
            except Exception as e:
                print(f"[SentenceFormer] spaCy unavailable: {e}. Falling back to rules.")
                self.mode = "rules"

    def is_named_entity(self, token: str) -> bool:
        # heuristic: capitalized and not a pronoun/determiner; spaCy check when available
        if token and token[0].isupper() and token.lower() not in PRONOUNS:
            return True
        if self.nlp and hasattr(self.nlp, "pipe"):
            doc = self.nlp(token)
            if doc.ents:
                return True
            # consider proper noun POS
            for t in doc:
                if t.pos_ == "PROPN":
                    return True
        return False

    def form(self, tokens: List[str], named_entity_idxs: set[int]) -> str:
        if not tokens:
            return ""
        # Subject selection
        subj_idx = None
        for i, t in enumerate(tokens):
            if t.lower() in PRONOUNS:
                subj_idx = i; break
        if subj_idx is None:
            # if first token looks like a name, use it as subject
            if 0 in named_entity_idxs or self.is_named_entity(tokens[0]):
                subj_idx = 0
        if subj_idx is None:
            # fallback: just join tokens
            s = " ".join(tokens)
            return s[0:1].upper() + s[1:] + ("." if s and s[-1] not in ".!?" else "")

        subj = PRONOUN_CANON.get(tokens[subj_idx].lower(), tokens[subj_idx].capitalize())
        cop = subj_verb(subj)

        # Find main complement
        comp = ""
        comp_idx = None
        if subj_idx + 1 < len(tokens):
            nxt = tokens[subj_idx+1]
            if (subj_idx+1) in named_entity_idxs or self.is_named_entity(nxt):
                # e.g., I + ADITYA
                comp = nxt.capitalize()
                comp_idx = subj_idx+1
            elif nxt.lower() in KINSHIP:
                poss = default_possessive(tokens)
                comp = f"{poss} {nxt.lower()}"
                comp_idx = subj_idx+1
            else:
                comp = nxt
                comp_idx = subj_idx+1
        else:
            # no complement; just return subject
            return f"{subj}."

        # Gather the rest (keep order, minimal glue)
        tail = []
        for i, t in enumerate(tokens):
            if i in (subj_idx, comp_idx):
                continue
            tail.append(t)

        # Build sentence
        base = f"{subj} {cop} {comp}"
        if tail:
            base += " " + " ".join(tail)
        base = base.strip()
        if base and base[-1] not in ".!?":
            base += "."
        # Capitalize sentence
        return base[0:1].upper() + base[1:]

# ---------- Text drawing ----------
def draw_wrapped_tokens(img, tokens: List[str], origin: Tuple[int,int], max_width: int,
                        color=(255,255,255), scale=0.9, thickness=2, line_gap=10):
    x0, y0 = origin
    x, y = x0, y0
    space_w, _ = cv2.getTextSize(" ", FONT, scale, thickness)[0]
    for tok in tokens:
        word = tok
        (w, h), _ = cv2.getTextSize(word, FONT, scale, thickness)
        if x + w > x0 + max_width:  # wrap
            x = x0
            y += h + line_gap
        cv2.putText(img, word, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
        x += w + space_w
    return y

def draw_wrapped_text(img, text: str, origin: Tuple[int,int], max_width: int,
                      color=(255,255,255), scale=0.8, thickness=2, line_gap=8):
    words = text.split() if text else []
    return draw_wrapped_tokens(img, words, origin, max_width, color, scale, thickness, line_gap)

# =================== APP ===================
@dataclass
class Context:
    static_domain: str = "Alphabet"   # Only for STATIC
    token_mode: str = "dynamic"       # 'static' or 'dynamic'
    tokens: List[str] = field(default_factory=list)
    last_pred: str = ''
    last_conf: float = 0.0
    spell_buf: str = ''               # live buffer for static letters/digits
    named_entity_idxs: set[int] = field(default_factory=set)  # indices in tokens list that are spelled names
    final_sentence: str = ""          # stored on 's'

def map_static_to_char(label: str, domain: str) -> str:
    """Map special static labels to their character for spelling, preserving display labels elsewhere."""
    if domain == "Alphabet":
        if label in ("E1","E2"): return "E"
        return label[:1].upper()
    else:
        if label.lower() in ("9a","9b"): return "9"
        return label

class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
        self.device = device
        self.ctx = Context(static_domain="Alphabet", token_mode=("dynamic" if args.default_dynamic else "static"))

        fps = max(1, args.fps)
        pre_from_s  = int(round((args.pre_secs or 0.0) * fps))
        post_from_s = int(round((args.post_secs or 0.0) * fps))
        pre_k  = max(0, args.pre_frames,  pre_from_s)
        post_k = max(0, args.post_frames, post_from_s)
        self.wm = WindowManager(mode_auto=(args.mode=='auto'), min_len=args.min_window, max_len=args.max_window,
                                on_thresh=args.on_thresh, off_thresh=args.off_thresh, pre_k=pre_k, post_k=post_k)
        self.flip = args.flip
        self.active = args.active  # for AUTO only
        self.font_white = (args.font_color == 'white')
        self.proc_scale = float(args.proc_scale)

        # Static models
        static_dir = Path(args.static_dir)
        self.static_alpha = StaticPredictor(static_dir, "Alphabet", device)
        self.static_num   = StaticPredictor(static_dir, "Numeral", device)

        # Dynamic model
        aug_root = Path(args.aug_root)
        ctr_dir  = Path(args.ctr_dir)
        self.dynamic = build_dynamic(aug_root, ctr_dir, force_cpu=args.cpu, use_half=args.half)

        # Holistic config
        self.holistic_complexity = int(args.holistic_complexity)

        # Sentence former (spaCy if available)
        self.sentencer = SentenceFormer(mode=args.nlp)

    def current_static(self) -> StaticPredictor:
        return self.static_alpha if self.ctx.static_domain == 'Alphabet' else self.static_num

    def static_majority(self, feats_126: List[np.ndarray]) -> Tuple[str, float]:
        be = self.current_static()
        X = np.stack(feats_126, axis=0).astype(np.float32)
        probs = be.predict_probs(X)  # (N,C)
        sums = probs.sum(axis=0)
        top_idx = int(np.argmax(sums))
        label = be.id_to_label(top_idx)
        conf = float(probs[:, top_idx].mean())
        return label, conf

    def _commit_spelling_buffer(self):
        """Commit the current spelling buffer as a word token (if non-empty) and mark it as a named entity."""
        if self.ctx.spell_buf:
            idx = len(self.ctx.tokens)
            # Title-case for display; keep original casing rules simple
            word = self.ctx.spell_buf.strip()
            if word.isupper():
                word = word.title()
            self.ctx.tokens.append(word)
            self.ctx.named_entity_idxs.add(idx)
            self.ctx.spell_buf = ''

    def run(self):
        cap = cv2.VideoCapture(self.args.cam)
        if self.args.width and self.args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_proc:
            print("Press 'h' for help.")
            hol = mp_holistic.Holistic(model_complexity=self.holistic_complexity, refine_face_landmarks=False,
                                       enable_segmentation=False, min_detection_confidence=0.35, min_tracking_confidence=0.5)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        print("[WARN] Camera frame not available.")
                        break
                    if self.flip:
                        frame = cv2.flip(frame, 1)

                    fb = extract_features_lazy(hands_proc, frame, proc_scale=self.proc_scale)

                    # Run Holistic only when needed for dynamic windows
                    if self.ctx.token_mode=='dynamic' and self.wm.state==WindowState.RECORDING:
                        fb.feat_258 = extract_258_from_holistic(fb.frame_proc, hol)

                    finished = self.wm.feed(fb, active=self.active)

                    # If we just transitioned to RECORDING (AUTO or MANUAL), retro-fill preframes' feat_258 for dynamic
                    if self.wm.state == WindowState.RECORDING and self.ctx.token_mode=='dynamic':
                        fill_feat_258_for_frames(self.wm.buffer, hol)

                    # HUD base
                    hud = fb.frame_disp.copy()
                    status = 'REC' if self.wm.state == WindowState.RECORDING else 'IDLE'
                    txt_color = (255,255,255) if self.font_white else (0,0,0)
                    cv2.putText(hud,
                                f"Mode: {'AUTO' if self.wm.mode_auto else 'MANUAL'} | Window: {status} | Active: {'ON' if self.active else 'OFF'} | Flip: {'ON' if self.flip else 'OFF'}",
                                (10, 24), FONT, 0.65, txt_color, 2, cv2.LINE_AA)
                    if self.wm.state == WindowState.RECORDING:
                        cv2.putText(hud, f"frames={len(self.wm.buffer)}", (10, 50), FONT, 0.65, txt_color, 2, cv2.LINE_AA)
                    if self.ctx.last_pred:
                        cv2.putText(hud, f"Last: {self.ctx.last_pred} ({self.ctx.last_conf:.2f})", (10, 74), FONT, 0.6, txt_color, 2, cv2.LINE_AA)

                    # Draw tokens + current spelling buffer preview
                    tokens_for_draw = self.ctx.tokens.copy()
                    if self.ctx.spell_buf:
                        # show partial word during spelling
                        preview = self.ctx.spell_buf if not self.ctx.spell_buf.isupper() else self.ctx.spell_buf.title()
                        tokens_for_draw.append(preview)
                    y_after_tokens = draw_wrapped_tokens(hud, tokens_for_draw, origin=(10, 100),
                                                         max_width=hud.shape[1]-20, color=txt_color, scale=0.9, thickness=2, line_gap=10)

                    # Live sentence preview
                    live_sentence = self.sentencer.form(self.ctx.tokens + ([preview] if self.ctx.spell_buf else []),
                                                        self.ctx.named_entity_idxs.copy() |
                                                        ({len(self.ctx.tokens)} if self.ctx.spell_buf else set()))
                    y_after_sentence = draw_wrapped_text(hud, "Sentence: " + (live_sentence or ""), origin=(10, y_after_tokens+20),
                                                         max_width=hud.shape[1]-20, color=txt_color, scale=0.8, thickness=2, line_gap=8)

                    # Finalized sentence (after pressing 's')
                    if self.ctx.final_sentence:
                        draw_wrapped_text(hud, "Final: " + self.ctx.final_sentence,
                                          origin=(10, y_after_sentence+18),
                                          max_width=hud.shape[1]-20, color=txt_color, scale=0.8, thickness=2, line_gap=8)

                    # Footer help
                    help_line = "q quit | Space start/stop/commit | t auto/manual | a active | f flip | g font | 1 static | 2 dynamic | m A↔N | Enter accept | b back | c clear | s finalize"
                    cv2.putText(hud, help_line, (10, hud.shape[0]-12), FONT, 0.50, txt_color, 1, cv2.LINE_AA)

                    cv2.imshow("ISL Unified Inference", hud)

                    # -------- Key handling --------
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('h'):
                        print("""
Keys:
  q: quit
  t: toggle AUTO/MANUAL windowing
  a: toggle Active (AUTO only). Starts OFF (idle)
  f: toggle Flip (mirror)
  g: toggle font color (white ↔ black)
  Space:
    - MANUAL + recording: start/stop the window
    - Otherwise: commit current spelled word
  1: STATIC token mode
  2: DYNAMIC token mode
  m: toggle Alphabet↔Numeral (STATIC only)
  Enter: accept last prediction again
  b: backspace (pop last char from spelling buffer, else last token)
  c: clear all tokens and spelling buffer
  s: finalize sentence (stores 'Final:' on screen)
                        """)
                    elif key == ord('t'):
                        self.wm.mode_auto = not self.wm.mode_auto; self.wm.reset()
                    elif key == ord('a'):
                        self.active = not self.active
                        if not self.active: self.wm.reset()
                    elif key == ord('f'):
                        self.flip = not self.flip
                    elif key == ord('g'):
                        self.font_white = not self.font_white
                    elif key == ord(' '):
                        if not self.wm.mode_auto and self.wm.state == WindowState.RECORDING:
                            # stop manual window
                            if len(self.wm.buffer) >= self.wm.min_len:
                                finished = self.wm.buffer.copy()
                            self.wm.reset()
                        else:
                            # commit current spelled word
                            self._commit_spelling_buffer()
                    elif key == ord('1'):
                        self.ctx.token_mode = 'static'
                    elif key == ord('2'):
                        self.ctx.token_mode = 'dynamic'
                        # commit any ongoing spelling before switching
                        self._commit_spelling_buffer()
                    elif key == ord('m') and self.ctx.token_mode == 'static':
                        self.ctx.static_domain = 'Numeral' if self.ctx.static_domain == 'Alphabet' else 'Alphabet'
                    elif key == 13:  # Enter
                        if self.ctx.last_pred:
                            if self.ctx.token_mode == 'static':
                                ch = map_static_to_char(self.ctx.last_pred, self.ctx.static_domain)
                                self.ctx.spell_buf += ch
                            else:
                                self._commit_spelling_buffer()
                                self.ctx.tokens.append(self.ctx.last_pred)
                    elif key == ord('b'):
                        if self.ctx.spell_buf:
                            self.ctx.spell_buf = self.ctx.spell_buf[:-1]
                        elif self.ctx.tokens:
                            # if last token was a named entity, unmark it
                            last_idx = len(self.ctx.tokens) - 1
                            if last_idx in self.ctx.named_entity_idxs:
                                self.ctx.named_entity_idxs.discard(last_idx)
                            self.ctx.tokens.pop()
                    elif key == ord('c'):
                        self.ctx.tokens.clear(); self.ctx.spell_buf = ''; self.ctx.named_entity_idxs.clear(); self.ctx.final_sentence = ""
                    elif key == ord('s'):
                        # finalize — flush buffer then store final sentence (on-screen, not printed)
                        self._commit_spelling_buffer()
                        self.ctx.final_sentence = self.sentencer.form(self.ctx.tokens, self.ctx.named_entity_idxs)

                    # -------- Window finished -> produce a token --------
                    if finished is not None:
                        if self.ctx.token_mode == 'dynamic':
                            fill_feat_258_for_frames(finished, hol)
                            feats_258 = [x.feat_258 for x in finished]
                            seq_258 = np.stack(feats_258, axis=0).astype(np.float32)
                            label, conf = self.dynamic.predict(seq_258, seq_len=self.args.seq)
                            # On dynamic: flush spelling buffer, then add the word
                            self._commit_spelling_buffer()
                            self.ctx.tokens.append(label)
                        else:
                            hands_feats = [x.hands_126 for x in finished if x.hands_126 is not None]
                            if hands_feats:
                                raw_label, conf = self.static_majority(hands_feats)
                                ch = map_static_to_char(raw_label, self.ctx.static_domain)
                                self.ctx.spell_buf += ch
                                label = raw_label
                            else:
                                label, conf = ("<no-hands>", 0.0)

                        self.ctx.last_pred, self.ctx.last_conf = label, conf

            finally:
                hol.close()

        cap.release()
        cv2.destroyAllWindows()

# =================== CLI ===================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified ISL inference (Static MLP + Dynamic models)")
    # camera
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    # processing performance
    ap.add_argument('--proc_scale', type=float, default=1.0, help='Downscale factor for processing (0.5~1.0)')
    ap.add_argument('--holistic_complexity', type=int, default=1, choices=[0,1,2], help='0=fastest, 1=default, 2=highest quality')
    ap.add_argument('--half', action='store_true', help='Use FP16 for dynamic model on CUDA')
    # windowing
    ap.add_argument('--mode', choices=['auto','manual'], default='auto')
    ap.add_argument('--min_window', type=int, default=12)
    ap.add_argument('--max_window', type=int, default=200)
    ap.add_argument('--on_thresh', type=int, default=5)
    ap.add_argument('--off_thresh', type=int, default=7)
    # AUTO pre/post context
    ap.add_argument('--pre_frames', type=int, default=0, help='Add this many frames BEFORE hands appear (AUTO/MANUAL start)')
    ap.add_argument('--post_frames', type=int, default=0, help='Add this many frames AFTER hands disappear (AUTO only)')
    ap.add_argument('--pre_secs', type=float, default=0.0, help='Override: seconds BEFORE (converted using fps)')
    ap.add_argument('--post_secs', type=float, default=0.0, help='Override: seconds AFTER (converted using fps)')
    # gates & flipping & font
    ap.add_argument('--active', action='store_true', help='Start Active (AUTO). Default OFF (idle)')
    ap.add_argument('--flip', dest='flip', action='store_true', help='Mirror input horizontally before prediction (default)')
    ap.add_argument('--no-flip', dest='flip', action='store_false')
    ap.add_argument('--font_color', choices=['white','black'], default='white', help='Overlay font color (white=bright, black=dark)')
    ap.set_defaults(flip=True)
    # static dir
    ap.add_argument('--static_dir', type=str, default='static')
    # dynamic defaults
    ap.add_argument('--aug_root', type=str, default='dynamic/data/top_100/aug_keypoints')
    ap.add_argument('--ctr_dir',  type=str, default='dynamic/data/top_100/ctr_gcn')
    ap.add_argument('--seq', type=int, default=200, help='Sequence length for dynamic models (pad/truncate)')
    ap.add_argument('--cpu', action='store_true', help='Force CPU for dynamic model')
    # sentence former
    ap.add_argument('--nlp', choices=['spacy','rules'], default='spacy',
                    help='Use spaCy small for POS/NER if available (fallback to rules if not installed).')
    # start mode
    ap.add_argument('--default_dynamic', action='store_true', help='Start with dynamic token mode')
    return ap.parse_args()

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try: cv2.setUseOptimized(True)
    except: pass
    try: cv2.setNumThreads(1)
    except: pass
    args = parse_args()
    app = App(args)
    app.run()

if __name__ == '__main__':
    main()
