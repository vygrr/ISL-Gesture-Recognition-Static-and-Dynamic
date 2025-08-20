#!/usr/bin/env python3
"""
inference.py — realtime tester for CTR-GCN / LSTM / BiLSTM+Attn / RelPos Transformer.

What this does:
  • Captures a short window from a camera (press Space to start/stop), builds a (T,258) keypoint sequence per frame
    using MediaPipe Holistic (pose+hands).
  • Recreates the exact input features your training used:
      - CTR-GCN path: joints (75x3) with optional body-centering/scale, +bones, +velocity per 'params'.
      - LSTM / BiLSTM+Attn / RelPos path: flat 258-D (no bones/vel; matches train_alt.py).
  • Loads labels from --data/label_to_id.json, checkpoint from --ckpt, and reads model config from:
      1) ckpt["params"] if present; else
      2) params.json next to the ckpt file (required).
  • Shows top-K predictions after each captured window; optionally saves a debug MP4 to --debug_dir.

Controls:
  Space = start/stop a window and run a prediction
  d     = toggle live landmark drawing while capturing
  r     = reset current window (if recording) or clear last prediction (if idle)
  q     = quit

Notes:
  • Camera is selected strictly by index via --cam.
  • No seated-crop or camera-name logic.
"""

from __future__ import annotations
import argparse, json, math, os, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp

# ---------------- Constants (must mirror training) ----------------
SEQ_LEN  = 200
FEAT_DIM = 258  # 33*4 + 21*3 + 21*3
DTYPE    = np.float32

# Pose/hand layout for joints/bones work (used by CTR-GCN path)
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

# slices for small gap impute (hands)
POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
L_SLICE  = slice(POSE_DIM, POSE_DIM + HAND_DIM)                 # 132:195
R_SLICE  = slice(POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM)  # 195:258

# ---------------- MediaPipe (capture) ----------------
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

def draw_pose_and_hands(frame_bgr, results):
    if results and results.pose_landmarks:
        mp_draw.draw_landmarks(frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    if results and results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                               mp_draw.DrawingSpec(color=(0,128,0), thickness=2))
    if results and results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                               mp_draw.DrawingSpec(color=(0,0,128), thickness=2))

def extract_258(frame_bgr, holistic):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
    return arr, res

def draw_panel_text(frame, lines, panel_w=None):
    h, w = frame.shape[:2]
    if panel_w is None: panel_w = max(360, w // 3)
    panel_h = 24 * (len(lines) + 1)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    y = 28
    for line, color in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 24

def draw_prediction_block(frame, top_labels, top_probs, infer_ms=None):
    lines = [("Prediction (last window)", (0, 255, 255))]
    for lbl, p in zip(top_labels, top_probs):
        lines.append((f"{lbl[:38]:<38} {p*100:5.1f}%", (0, 255, 0)))
    if infer_ms is not None:
        lines.append((f"Infer: {infer_ms:.1f} ms", (200, 200, 200)))
    lines.append(("(Space=start/stop | d=toggle live draw | r=reset | q=quit)", (180, 180, 220)))
    draw_panel_text(frame, lines)

# ---------------- Repairs / transforms (match training) ----------------
def impute_short_gaps(seq: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """ Fill tiny gaps in hand tracks to reduce spurious zeros. """
    x = seq.copy()
    for slc in (L_SLICE, R_SLICE):
        sub = x[:, slc]
        miss = (np.abs(sub).sum(axis=1) == 0.0)
        if not miss.any(): continue
        for c in range(sub.shape[1]):
            vals = sub[:, c]
            i = 0
            while i < len(vals):
                if miss[i]:
                    j = i
                    while j < len(vals) and miss[j]: j += 1
                    gap = j - i
                    if gap <= max_gap:
                        li = i - 1
                        ri = j
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
    # pose
    for a,b in POSE_EDGES:
        if P[b] == -1 and a != b: P[b] = a
        if P[a] == -1 and b != a: P[a] = b
    # left hand
    for (u,v) in HAND_EDGES:
        gu, gv = V_POSE+u, V_POSE+v
        if P[gv] == -1: P[gv] = gu
    # right hand
    for (u,v) in HAND_EDGES:
        gu, gv = V_POSE+V_LHAND+u, V_POSE+V_LHAND+v
        if P[gv] == -1: P[gv] = gu
    P[LHAND_ROOT] = POSE_WRIST_L; P[RHAND_ROOT] = POSE_WRIST_R
    bones = np.zeros_like(joints, dtype=np.float32)
    for v in range(V):
        p = int(P[v])
        bones[:, v, :] = joints[:, v, :] - (joints[:, p, :] if p >= 0 else 0.0)
    return bones

# ---------------- Models (mirror training code) ----------------
class CTRGCNBlock(nn.Module):
    def __init__(self, C_in, C_out, V=V_ALL, kernel_t=9, stride=1, dropout=0.3):
        super().__init__()
        self.theta = nn.Conv2d(C_in, C_out//4, 1)
        self.phi   = nn.Conv2d(C_in, C_out//4, 1)
        self.g     = nn.Conv2d(C_in, C_out,   1)

        # graph
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

# ---------------- Params & model builder ----------------
def load_params_from_ckpt_and_sidecar(ckpt_path: Path) -> Dict:
    """Return params dict from checkpoint['params'] or params.json next to it; defaults otherwise."""
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
        raise SystemExit("Could not find run params (neither in ckpt['params'] nor params.json next to --ckpt).")
    return params

def _normalize_model_name(name: str) -> str:
    n = (name or "").lower().strip()
    # Map ctr_gcn_bihand → ctr_gcn (same architecture; only training-time augmentation differs)
    aliases = {
        "ctr-gcn": "ctr_gcn",
        "ctr_gcn_bihand": "ctr_gcn",
        "bilstm+att": "bilstm_att",
        "bilstm_attn": "bilstm_att",
        "relpos_transformer": "relpos",
    }
    return aliases.get(n, n)

def build_model_from_params(num_classes: int, params: Dict):
    model_name_raw = params.get("model", "")
    model_name = _normalize_model_name(model_name_raw)
    if model_name == "ctr_gcn":
        c_in = 3 + (3 if params.get("use_bones", False) else 0) + (3 if params.get("use_vel", False) else 0)
        model = CTRGCN(num_classes,
                       c_in=c_in,
                       channels=(64,128,256),
                       blocks=(2,2,2),
                       kernel_t=int(params.get("kernel_t", 9)),
                       dropout=float(params.get("dropout", 0.3)))
    elif model_name == "lstm":
        model = LSTMHead(FEAT_DIM,
                         hidden=int(params.get("hidden", 128)),
                         num_classes=num_classes,
                         dropout=float(params.get("dropout", 0.3)))
    elif model_name == "bilstm_att":
        model = BiLSTMAttn(num_classes=num_classes,
                           hidden=int(params.get("hidden", 128)),
                           layers=int(params.get("layers", 2)),
                           dropout=float(params.get("dropout", 0.3)),
                           feat_dim=FEAT_DIM)
    elif model_name == "relpos":
        model = RelPosTransformer(num_classes=num_classes,
                                  d_in=FEAT_DIM,
                                  d_model=int(params.get("d_model", 256)),
                                  layers=int(params.get("tr_layers", params.get("layers", 6))),
                                  nhead=int(params.get("nhead", 8)),
                                  dropout=float(params.get("dropout", 0.2)),
                                  max_len=max(SEQ_LEN, 512))
    else:
        raise SystemExit(f"Unknown model in params.json: '{model_name_raw}' (normalized='{model_name}').")
    return model, model_name

# ---------------- Build per-model inputs ----------------
def build_inputs_for_model(seq_258: np.ndarray, valid_len: int, model_name: str, params: Dict, device):
    """Return dict with tensors ready for the chosen model."""
    T = seq_258.shape[0]
    if model_name == "ctr_gcn":
        j = slice_to_joints_xyz(seq_258)            # (T,75,3)
        if params.get("normalize_body", False):
            j = body_center_scale(j)
        feats = [j]
        if params.get("use_bones", False):
            b = bone_vectors(j)                     # (T,75,3)
            feats.append(b)
        if params.get("use_vel", False):
            v = np.zeros_like(j); v[1:] = j[1:] - j[:-1]
            feats.append(v)
        x = np.concatenate(feats, axis=-1)          # (T,75,C)
        xt = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).contiguous().to(device)  # (1,C,T,V=75)
        return {"x": xt, "mask": None, "lengths": torch.tensor([valid_len], device=device)}
    else:
        X = torch.from_numpy(seq_258).unsqueeze(0).to(device)  # (1,T,258)
        mask = torch.arange(T, device=device)[None,:] >= torch.tensor([valid_len], device=device)[:,None]
        return {"x": X, "mask": mask, "lengths": torch.tensor([valid_len], device=device)}

# ---------------- Debug save ----------------
def save_debug_clip(frames_bgr: List[np.ndarray], out_path: Path, fps=30):
    """Re-run holistic to draw overlays for a clean debug MP4."""
    if not frames_bgr: return False
    hol = mp_holistic.Holistic(static_image_mode=False, model_complexity=2,
                               enable_segmentation=False, refine_face_landmarks=False,
                               min_detection_confidence=0.35, min_tracking_confidence=0.5)
    h, w = frames_bgr[0].shape[:2]
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_bgr:
        res = hol.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        vis = f.copy()
        draw_pose_and_hands(vis, res)
        out.write(vis)
    out.release()
    try: hol.close()
    except: pass
    return True

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # data/model
    ap.add_argument("--data", required=True, help="Augmented data folder containing label_to_id.json")
    ap.add_argument("--ckpt", required=True, help="Path to exact checkpoint .pt (e.g., ckpt_best.pt or ckpt_last.pt)")
    # capture
    ap.add_argument("--cam", type=int, default=0, help="Camera index (integer)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)
    # runtime
    ap.add_argument("--seq", type=int, default=SEQ_LEN)
    ap.add_argument("--min_frames", type=int, default=24)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    ap.add_argument("--live_draw", action="store_true", help="Draw pose+hands live while capturing")
    # optional debug save
    ap.add_argument("--debug_dir", type=str, default="", help="If set, save annotated MP4s here")
    ap.add_argument("--debug_fps", type=int, default=30)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try: cv2.setUseOptimized(True)
    except: pass
    try: cv2.setNumThreads(1)
    except: pass

    # labels
    data_root = Path(args.data)
    label_to_id = json.loads((data_root / "label_to_id.json").read_text(encoding="utf-8"))
    id_to_label = {int(v): k for k, v in label_to_id.items()}
    num_classes = len(id_to_label)

    # params & model (model name is read from params; no --model arg)
    ckpt_path = Path(args.ckpt).resolve()
    params = load_params_from_ckpt_and_sidecar(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type.upper()}" + (f": {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    model, model_name = build_model_from_params(num_classes, params)
    model = model.to(device).eval()
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model_state"], strict=False)
    print(f"[Model] {model_name} | normalize_body={params.get('normalize_body', False)} "
          f"use_bones={params.get('use_bones', False)} use_vel={params.get('use_vel', False)}")

    # holistic (capture)
    hol = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=int(params.get("holistic_complexity", 2)),
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=float(params.get("det_conf", 0.35)),
        min_tracking_confidence=float(params.get("trk_conf", 0.5)),
    )

    # camera by index only
    cap = cv2.VideoCapture(int(args.cam))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.cam}")

    # state
    dbg_dir = Path(args.debug_dir) if args.debug_dir else None
    if dbg_dir: dbg_dir.mkdir(parents=True, exist_ok=True)
    recording = False
    window_feats, window_frames = [], []
    last_labels, last_probs, last_ms = [], [], None
    window_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera read failed")
                break
            frame = cv2.flip(frame, 1)

            feats, res = extract_258(frame, hol)
            show = frame.copy()
            if args.live_draw:
                draw_pose_and_hands(show, res)

            if recording:
                window_feats.append(feats)
                window_frames.append(frame.copy())
                draw_panel_text(show, [
                    (f"RECORDING window #{window_id}", (0, 255, 255)),
                    (f"Frames: {len(window_frames)}  (Space: stop | r: reset | d: draw | q: quit)", (200,200,200)),
                ])
            else:
                if last_labels:
                    draw_prediction_block(show, last_labels, last_probs, infer_ms=last_ms)
                else:
                    draw_panel_text(show, [
                        ("IDLE", (0,255,255)),
                        ("Press Space to START a detection window", (0,255,0)),
                        ("(q: quit | d: LIVE draw | r: clear)", (200,200,200)),
                    ])

            cv2.imshow("Realtime Inference (CTR-GCN / LSTM / BiLSTM+Att / RelPos)", show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d'):
                args.live_draw = not args.live_draw
            elif key == ord('r'):
                if recording:
                    window_feats.clear(); window_frames.clear()
                else:
                    last_labels, last_probs, last_ms = [], [], None
            elif key == 32:  # Space
                if not recording:
                    recording = True
                    window_feats.clear(); window_frames.clear()
                    window_id += 1
                else:
                    recording = False
                    n = len(window_frames)
                    if n < args.min_frames:
                        last_labels, last_probs, last_ms = ["(too few frames)"], [0.0], None
                        continue

                    # Build window sequence (T,258)
                    L = min(n, args.seq)
                    x = np.stack(window_feats[:L], axis=0).astype(DTYPE)  # (L,258)
                    x = impute_short_gaps(x, max_gap=5)
                    if L < args.seq:
                        pad = np.zeros((args.seq - L, FEAT_DIM), dtype=DTYPE)
                        x = np.concatenate([x, pad], axis=0)
                    # Build model-specific tensors
                    batch = build_inputs_for_model(x, L, model_name, params, device)

                    # Inference
                    t0 = time.time()
                    with torch.inference_mode():
                        try:
                            autocast_ctx = torch.amp.autocast
                        except AttributeError:
                            from torch.cuda.amp import autocast as autocast_ctx
                        with autocast_ctx(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type=="cuda")):
                            if model_name == "ctr_gcn":
                                logits = model(batch["x"])
                            elif model_name in ("lstm","bilstm_att"):
                                logits = model(batch["x"], batch["lengths"])
                            else:  # relpos
                                logits = model(batch["x"], key_padding_mask=batch["mask"])
                            probs = torch.softmax(logits, dim=1)
                    last_ms = (time.time() - t0) * 1000.0

                    probs = probs.float().squeeze(0).cpu().numpy()
                    k = int(np.clip(args.topk, 1, len(id_to_label)))
                    idx = np.argsort(-probs)[:k]
                    last_labels = [id_to_label[int(i)] for i in idx]
                    last_probs  = [float(probs[int(i)]) for i in idx]

                    # Optional debug save
                    if dbg_dir:
                        ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        top1 = last_labels[0] if last_labels else "NA"
                        top1 = "".join(c if c.isalnum() or c in "-_." else "_" for c in top1)[:40]
                        acc1 = f"{last_probs[0]:.2f}" if last_probs else "0.00"
                        fname = f"w{window_id:04d}_{ts}_{top1}_acc{acc1}.mp4"
                        out_path = dbg_dir / fname
                        ok_save = save_debug_clip(window_frames[:L], out_path=out_path, fps=args.debug_fps)
                        print(f"[DEBUG] Saved → {out_path}" if ok_save else "[DEBUG] Save skipped")

    finally:
        try: cap.release()
        except: pass
        cv2.destroyAllWindows()
        try: hol.close()
        except: pass

if __name__ == "__main__":
    main()
