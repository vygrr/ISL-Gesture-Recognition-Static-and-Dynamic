#!/usr/bin/env python3
"""
train_alt.py — Train alternative sequence models on augmented keypoints.

Models:
  • lstm         : single-layer LSTM → MLP head
  • bilstm_att   : BiLSTM encoder + additive attention pooling → MLP head
  • relpos       : Transformer encoder with relative position bias

Data expectation (from augment.py):
  <data_root>/
    label_to_id.json
    index_train.csv
    index_val.csv
    index_test.csv (optional)
    train/<label_id>/*.npz, val/<label_id>/*.npz, ...

Auto save-dir selection:
  If --save is NOT given and --data points to one of:
    dynamic/data/include_50/aug_keypoints
    dynamic/data/include/aug_keypoints
    dynamic/data/top_<K>/aug_keypoints
  outputs go to:
    dynamic/data/<subset>/<model_name>        # subset in {include_50, include, top_<K>}
    where model_name ∈ {lstm, bilstm_att, relpos}
Otherwise pass --save explicitly.

Checkpoints & logs:
  - ckpt_best.pt    (best by val macro-F1; tie-break by lower val loss)
  - ckpt_last.pt    (last epoch for resume)
  - log.csv         (per-epoch metrics)
  - params.json     (run config)

Robust path resolution:
  The loader tolerates old absolute paths in index_*.csv (e.g., from another machine).
  It reconstructs candidates under the current --data root until a match exists.
"""

from __future__ import annotations
import argparse, json, math, os
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
FEAT_DIM = 258  # 33*4 (pose) + 21*3 (left hand) + 21*3 (right hand) = 132 + 63 + 63

# ---------------------- IO utils ----------------------
def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def auto_save_dir(data_root: Path, model_folder: str) -> Optional[Path]:
    """
    If data_root ends with .../aug_keypoints and its parent is include_50/include/top_*
    return dynamic/data/<subset>/<model_folder>, else None.
    """
    if data_root.name != "aug_keypoints":
        return None
    subset = data_root.parent.name  # include_50 | include | top_<K>
    if subset.startswith("include") or subset.startswith("top_"):
        # data_root: .../data/<subset>/aug_keypoints
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
        # estimate usable length: last frame that isn't all zeros
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

# ---------------------- Models ----------------------
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
        _, (h, _) = self.lstm(packed)          # h: (num_layers, B, H)
        h = h[-1]                               # (B,H)
        z = torch.tanh(self.fc1(h))
        z = self.drop(z)
        return self.out(z)

class BiLSTMAttn(nn.Module):
    """BiLSTM + masked additive attention over time."""
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
        # mask: True where padded
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
        q = self.q(x).view(B,T,self.nhead,self.dk).transpose(1,2)  # (B,H,T,dk)
        k = self.k(x).view(B,T,self.nhead,self.dk).transpose(1,2)
        v = self.v(x).view(B,T,self.nhead,self.dk).transpose(1,2)
        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.dk)  # (B,H,T,T)
        # relative bias
        pos = torch.arange(T, device=x.device)
        rel = torch.clamp(pos[None,:]-pos[:,None] + (self.max_len-1), 0, 2*self.max_len-2)
        bias = self.rel_bias(rel).permute(2,0,1).unsqueeze(0)  # (1,H,T,T)
        attn = attn + bias
        # key padding mask: True -> mask
        if key_padding_mask is not None:
            m = key_padding_mask[:, None, None, :]  # (B,1,1,T)
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
    def forward(self, x, key_padding_mask=None):  # x: (B,T,D)
        x = self.proj(x)
        for b in self.blocks:
            x = b(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        x = x.mean(1)   # mean pool over valid tokens (mask handled inside attention)
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
def evaluate(model, loader, device, model_kind: str, use_amp: bool, num_classes: int, ls_eps: float):
    model.eval()
    crit = LabelSmoothingCE(ls_eps)
    loss_sum, correct, total = 0.0, 0, 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for X, Y, L in loader:
        X, Y, L = X.to(device), Y.to(device), L.to(device)
        if model_kind == "relpos":
            mask = (torch.arange(X.size(1), device=device)[None,:] >= L[:,None])  # PAD mask
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type=="cuda")):
            if model_kind == "lstm":
                logits = model(X, L)
            elif model_kind == "bilstm_att":
                logits = model(X, L)
            else:  # relpos
                logits = model(X, key_padding_mask=mask)
            loss = crit(logits, Y)
        loss_sum += float(loss.item()) * Y.size(0)
        pred = logits.argmax(1)
        correct += (pred == Y).sum().item()
        total    += Y.numel()
        for t, p in zip(Y.cpu().numpy(), pred.cpu().numpy()):
            conf[int(t), int(p)] += 1
    return {"loss": loss_sum/max(1,total),
            "acc":  correct/max(1,total),
            "macro_f1": _macro_f1_from_conf(conf)}

def better_by_f1_then_loss(curr, best):
    if best is None: return True
    if curr["macro_f1"] > best["macro_f1"] + 1e-12: return True
    if abs(curr["macro_f1"] - best["macro_f1"]) <= 1e-12 and curr["loss"] < best["loss"] - 1e-12: return True
    return False

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder that contains label_to_id.json & index_*.csv (augment output root)")
    ap.add_argument("--save", type=str, default="", help="Output folder (auto-selected if --data under data/include_*/top_*/aug_keypoints)")
    ap.add_argument("--model", choices=["lstm","bilstm_att","relpos"], required=True)

    # shared hparams
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ls_eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", type=str, default="", help="Path to resume, or 'auto' (looks for ckpt_last.pt)")

    # LSTM/BiLSTM specifics
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)

    # RelPos specifics
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--tr_layers", type=int, default=6)

    args = ap.parse_args()

    # Repro
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data).resolve()
    # Save root
    model_folder = args.model
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

    # Labels / splits
    l2id = json.loads((data_root / "label_to_id.json").read_text(encoding="utf-8"))
    num_classes = len(l2id)
    tr_paths, tr_labels = load_index(data_root / "index_train.csv", data_root)
    va_paths, va_labels = load_index(data_root / "index_val.csv",   data_root)
    if not tr_paths or not va_paths:
        raise SystemExit("No training/validation data found. Run augment first.")
    train_loader = DataLoader(NpzSeq(tr_paths, tr_labels), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=collate)
    val_loader   = DataLoader(NpzSeq(va_paths, va_labels), batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=collate)

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type.upper()}" + (f": {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    if args.model == "lstm":
        model = LSTMHead(FEAT_DIM, hidden=args.hidden, num_classes=num_classes, dropout=args.dropout).to(device)
    elif args.model == "bilstm_att":
        model = BiLSTMAttn(num_classes=num_classes, hidden=args.hidden, layers=args.layers,
                           dropout=args.dropout, feat_dim=FEAT_DIM).to(device)
    else:
        model = RelPosTransformer(num_classes=num_classes, d_in=FEAT_DIM, d_model=args.d_model,
                                  layers=args.tr_layers, nhead=args.nhead, dropout=args.dropout,
                                  max_len=max(SEQ_LEN,512)).to(device)

    # Optim & sched
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type=="cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    # Save params
    params = {
        "model": args.model, "num_classes": num_classes, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "wd": args.wd, "dropout": args.dropout, "amp": bool(args.amp), "ls_eps": float(args.ls_eps),
        "hidden": args.hidden, "layers": args.layers,
        "d_model": args.d_model, "nhead": args.nhead, "tr_layers": args.tr_layers,
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
            model.load_state_dict(ck["model_state"], strict=False)
            if "opt_state" in ck:
                try: optim.load_state_dict(ck["opt_state"])
                except Exception: pass
            if "sched_state" in ck:
                try: sched.load_state_dict(ck["sched_state"])
                except Exception: pass
            if "epoch" in ck:
                start_epoch = int(ck["epoch"]) + 1
            print(f"[RESUME] start_epoch={start_epoch}")

    # Train
    best: Optional[Dict[str, float]] = None
    logs = []
    crit = LabelSmoothingCE(args.ls_eps)

    for epoch in range(start_epoch, args.epochs + 1):
        # ---- Train ----
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"train {epoch:03d}", leave=False)
        for X, Y, L in pbar:
            X, Y, L = X.to(device), Y.to(device), L.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type=="cuda")):
                if args.model in ("lstm","bilstm_att"):
                    logits = model(X, L)
                else:
                    mask = (torch.arange(X.size(1), device=device)[None,:] >= L[:,None])
                    logits = model(X, key_padding_mask=mask)
                loss = crit(logits, Y)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            loss_sum += float(loss.item()) * Y.size(0)
            correct  += (logits.argmax(1) == Y).sum().item()
            total    += Y.numel()
        tr_loss = loss_sum / max(1,total)
        tr_acc  = correct / max(1,total)
        sched.step()

        # ---- Val ----
        va = evaluate(model, val_loader, device, args.model, args.amp, num_classes, args.ls_eps)

        logs.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va["loss"], "val_acc": va["acc"], "val_f1": va["macro_f1"]})
        pd.DataFrame(logs).to_csv(log_csv, index=False)
        print(f"[{epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va['loss']:.4f} acc={va['acc']:.4f} f1={va['macro_f1']:.4f}")

        # Save last + maybe best
        ck = {"model_state": model.state_dict(), "opt_state": optim.state_dict(),
              "sched_state": sched.state_dict(), "epoch": epoch, "params": params}
        torch.save(ck, last_path)
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
