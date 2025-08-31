# Dynamic Module (Words) — CTR‑GCN & Alternatives

This module handles **dynamic sign recognition** from short pose+hands sequences.
End‑to‑end flow: **split CSVs → augmentation → keypoints → training → evaluation → realtime**.

---

## 1) Augmentation → keypoints
`augment.py` reads your INCLUDE split CSVs + raw videos, crops and trims each clip, and writes compact keypoint tensors.
It is **resume‑safe**, can verify/repair existing `.npz`, and auto‑selects output directories based on `--subset`.

**Inputs (expected under `--root`, default: `dynamic/data`)**
```
include_train.csv
include_val.csv
[include_test.csv]
raw_videos/...   # path patterns inside the CSVs
```

**Outputs (auto‑selected unless `--out` is given)**
```
dynamic/data/include_50/aug_keypoints/
dynamic/data/include/aug_keypoints/
dynamic/data/top_<K>/aug_keypoints/
  ├─ label_to_id.json
  ├─ index_{train,val,test}.csv
  ├─ train/<label_id>/*.npz
  └─ val/<label_id>/*.npz
```

**Example**
```bash
# Top‑100 setup with gentle idle‑trim and 4 workers
python dynamic/augment.py \
  --root dynamic/data \
  --subset topk --top_k 100 \
  --trim_idle --workers 4
```

---

## 2) Train models
### CTR‑GCN (recommended)
```bash
python dynamic/train.py \  --data dynamic/data/top_100/aug_keypoints \  --normalize_body --use_bones --use_vel \  --bihand --bihand_p 0.5 --bihand_ramp_epoch 0 \  --epochs 60 --batch 64 --amp
```
If `--save` is not provided and `--data` matches one of the known subsets, outputs go to:
```
dynamic/data/<subset>/ctr_gcn/
  ckpt_best.pt  ckpt_last.pt  params.json  log.csv
```

### Alternatives (for experiments)
- `lstm` — single‑layer LSTM → MLP
- `bilstm_att` — BiLSTM + additive attention pooling
- `relpos` — Transformer encoder with relative position bias

```bash
python dynamic/train_alt.py \  --data dynamic/data/top_100/aug_keypoints \  --model relpos --epochs 60 --batch 64 --amp
```

---

## 3) Evaluate
```bash
python dynamic/eval.py \  --data dynamic/data/top_100/aug_keypoints \  --ckpt dynamic/data/top_100/ctr_gcn/ckpt_best.pt
# Prints macro‑F1/acc/loss, reads params from ckpt/params.json.
```

---

## 4) Realtime test (dynamic only)
```bash
python dynamic/inference.py \  --data dynamic/data/top_100/aug_keypoints \  --ckpt dynamic/data/top_100/ctr_gcn/ckpt_best.pt
```

**Tips**
- Set `--amp` for mixed‑precision on CUDA.
- Use `--resume` (`train.py` auto‑resumes from `ckpt_last.pt` if present).
- For left‑handed signers at inference, try `--flip` (mirrors the view).
