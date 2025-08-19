#!/usr/bin/env python3
"""
train.py

Trains two PyTorch MLPs (Alphabet & Numeral) on the landmark features:

- Loads from data/load/alphabets_data.npz and numerals_data.npz
- Stratified 80/10/10 split (train/val/test)
- Up to 1000 epochs, ReduceLROnPlateau (no hard early stop)
- Per-batch progress bar inside each epoch
- Saves BEST (by val loss) model + LabelEncoder to:
    data/model/alphabets.pth, data/encoder/alphabets.pkl
    data/model/numerals.pth,  data/encoder/numerals.pkl
- Prints final test-set accuracy and classification report
"""

import time
import random
import argparse
import numpy as np
from pathlib import Path
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score

from tqdm import tqdm

# ------------------ Reproducibility ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------ Model ------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):  # x: [B, 126]
        return self.net(x)

# ------------------ Data ------------------
def load_npz(npz_path: Path):
    arr = np.load(npz_path, allow_pickle=True)
    return arr["X"], arr["y"]

def stratified_indices(X, y_enc, val_frac=0.1, test_frac=0.1, seed=42):
    # split off test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(X, y_enc))
    X_tv, y_tv = X[train_val_idx], y_enc[train_val_idx]
    # split train/val
    rel_val = val_frac / (1 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss2.split(X_tv, y_tv))
    train_idx = train_val_idx[train_idx_rel]
    val_idx   = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx

def make_loaders(X, y, batch_size=64, seed=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    train_idx, val_idx, test_idx = stratified_indices(X, y_enc, 0.1, 0.1, seed)

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y_enc).long()

    ds_all = TensorDataset(X_t, y_t)
    train_ds = Subset(ds_all, train_idx)
    val_ds   = Subset(ds_all, val_idx)
    test_ds  = Subset(ds_all, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, le

# ------------------ Train/Eval ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            total_loss += criterion(out, yb).item() * xb.size(0)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    return avg_loss, acc, preds, labels

def train_one(dataset_name, npz_path, model_out, encoder_out, epochs=1000, batch_size=64, lr=1e-3, seed=42):
    print(f"\n=== {dataset_name} ===")
    X, y = load_npz(npz_path)
    train_loader, val_loader, test_loader, le = make_loaders(X, y, batch_size, seed)

    model = MLP(input_dim=X.shape[1], num_classes=len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=10, min_lr=1e-6, verbose=True)

    best_state = None
    best_val_loss = float("inf")
    start = time.time()

    for epoch in range(1, epochs+1):
        # train loop with per-batch progress
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * xb.size(0)
            pred = out.argmax(1)
            correct += pred.eq(yb).sum().item()
            total += yb.size(0)
            pbar.set_postfix(loss=f"{run_loss/total:.4f}", acc=f"{correct/total:.4f}")

        train_loss = run_loss / len(train_loader.dataset)
        train_acc  = correct / len(train_loader.dataset)

        # val
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    elapsed = time.time() - start
    print(f"{dataset_name} training finished in {elapsed/60:.1f} min (best val_loss={best_val_loss:.4f})")

    # Save best
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, model_out)
    joblib.dump(le, encoder_out)
    print(f"💾 Saved model → {model_out}")
    print(f"💾 Saved encoder → {encoder_out}")

    # Test set evaluation
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\n--- {dataset_name} Test ---")
    print(f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=le.classes_))

def get_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--load_dir",   type=Path, default=here / "data" / "load")
    p.add_argument("--model_dir",  type=Path, default=here / "data" / "model")
    p.add_argument("--encoder_dir",type=Path, default=here / "data" / "encoder")
    return p.parse_args()

if __name__ == "__main__":
    set_seed(42)
    args = get_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.encoder_dir.mkdir(parents=True, exist_ok=True)

    alpha_npz = args.load_dir / "alphabets_data.npz"
    num_npz   = args.load_dir / "numerals_data.npz"

    train_one("Alphabets", alpha_npz,
              args.model_dir / "alphabets_model.pth",
              args.encoder_dir / "alphabets_le.pkl",
              epochs=args.epochs, batch_size=args.batch, lr=args.lr, seed=args.seed)

    train_one("Numerals",  num_npz,
              args.model_dir / "numerals_model.pth",
              args.encoder_dir / "numerals_le.pkl",
              epochs=args.epochs, batch_size=args.batch, lr=args.lr, seed=args.seed)
