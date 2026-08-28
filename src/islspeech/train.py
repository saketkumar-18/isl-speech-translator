"""Train the ISL sequence classifier: GRU+attention vs frame-MLP ablation.

Evaluation protocol (honest for this data regime)
--------------------------------------------------
The curated vocabulary has 82 words / 99 videos: most words have a single
dictionary recording, ~17 words have 2-3 sign variants (different native
signers / renditions). Two protocols are reported:

1. LEAVE-ONE-VARIANT-OUT (LOVO) — the headline generalization metric.
   For every word with >= 2 variants, hold out each variant in turn, train
   on everything else (including the word's remaining variants), and predict
   the held-out variant. This measures exactly the deployment scenario:
   a *new recording/signer* of a vocabulary sign. Fixed epoch count, no
   early stopping on the hold-out (no leakage).

2. FULL-VOCABULARY FIT — the deployed model is trained on ALL 99 sequences
   (train accuracy reported as a sanity check, clearly labeled).

The frame-MLP (mean-pool, no temporal modeling) baseline runs the identical
LOVO protocol; the gap quantifies the value of temporal modeling.

Usage:
    python -m islspeech.train --data data/landmarks.npz --out artifacts/
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from . import FRAME_FEAT, HAND_FEAT, SEQ_LEN
from .models import FrameMLPClassifier, GRUAttentionClassifier

RNG = np.random.default_rng(23035010051)  # roll no. seed, reproducible


# ---------------------------------------------------------------- augmentation
def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """seq: (T, 128). Spatial + temporal augmentation on normalized landmarks."""
    out = seq.copy()
    T = out.shape[0]

    # 1) Gaussian noise on coordinates (presence channels untouched)
    if rng.random() < 0.8:
        noise = rng.normal(0, 0.01, size=(T, 2 * HAND_FEAT)).astype(np.float32)
        out[:, :2 * HAND_FEAT] += noise

    # 2) random scale jitter per hand slot
    for slot in range(2):
        if rng.random() < 0.5:
            s = float(rng.uniform(0.85, 1.15))
            out[:, slot * HAND_FEAT:(slot + 1) * HAND_FEAT] *= s

    # 3) rotation about the camera axis (x, y of every landmark)
    if rng.random() < 0.6:
        theta = float(rng.uniform(-0.2, 0.2))  # ~+-11 deg
        c, s = np.cos(theta), np.sin(theta)
        for slot in range(2):
            base = slot * HAND_FEAT
            xs = out[:, base + 0::3].copy()
            ys = out[:, base + 1::3].copy()
            out[:, base + 0::3] = c * xs - s * ys
            out[:, base + 1::3] = s * xs + c * ys

    # 4) horizontal mirror with handedness-slot swap
    if rng.random() < 0.5:
        left = out[:, 0:HAND_FEAT].copy()
        right = out[:, HAND_FEAT:2 * HAND_FEAT].copy()
        out[:, 0:HAND_FEAT] = right
        out[:, HAND_FEAT:2 * HAND_FEAT] = left
        out[:, 0:HAND_FEAT:3] *= -1  # flip x of new slot0
        out[:, HAND_FEAT:2 * HAND_FEAT:3] *= -1
        p0 = out[:, 2 * HAND_FEAT].copy()
        out[:, 2 * HAND_FEAT] = out[:, 2 * HAND_FEAT + 1]
        out[:, 2 * HAND_FEAT + 1] = p0

    # 5) temporal speed warp via re-resampling
    if rng.random() < 0.5:
        speed = float(rng.uniform(0.85, 1.15))
        n_eff = max(2, int(round(T / speed)))
        idx = np.clip(np.floor(np.arange(T) * (n_eff - 1) / (T - 1)).astype(int), 0, T - 1)
        out = _resample(out[idx], T)
    return out


def _resample(src: np.ndarray, T: int) -> np.ndarray:
    n = len(src)
    if n == 1:
        return np.repeat(src, T, axis=0)
    idx = np.floor(np.arange(T) * (n - 1) / (T - 1)).astype(int)
    return src[idx]


def presence_mask(seq: np.ndarray) -> np.ndarray:
    """(T,) float32: 1 where any hand present."""
    return (seq[:, 2 * HAND_FEAT] + seq[:, 2 * HAND_FEAT + 1] > 0).astype(np.float32)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment_flag: bool):
        self.X, self.y, self.aug = X, y, augment_flag

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        seq = self.X[i]
        if self.aug:
            seq = augment(seq, RNG)
        mask = presence_mask(seq)
        return (torch.from_numpy(seq), torch.from_numpy(mask), int(self.y[i]))


# ---------------------------------------------------------------- training loop
def train_one_epoch(model, loader, opt, device, label_smoothing=0.1):
    model.train()
    tot, n = 0.0, 0
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        opt.zero_grad()
        logits, _ = model(x, m)
        loss = crit(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    ds = SeqDataset(X, np.zeros(len(X), dtype=int), False)
    loader = DataLoader(ds, batch_size=64)
    preds = []
    for x, m, _ in loader:
        logits, _ = model(x.to(device), m.to(device))
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def fit_simple(make_model, X, y, num_classes, epochs=40, batch=32, lr=2e-3, device="cpu"):
    """Fixed-epoch training (no validation peeking) -> final model."""
    model = make_model(num_classes).to(device)
    loader = DataLoader(SeqDataset(X, y, True), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        train_one_epoch(model, loader, opt, device)
        sched.step()
    return model


# ---------------------------------------------------------------- LOVO protocol
def lovo_indices(y: np.ndarray):
    """Yield (word_id, hold_idx, train_idx) for every variant of multi-variant words."""
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        for hold in idx:
            tr = np.setdiff1d(np.arange(len(y)), [hold])
            yield int(c), int(hold), tr


def run_lovo(make_model, X, y, classes, epochs, device, tag):
    """Leave-one-variant-out. Returns (accuracy, per-word results)."""
    per_word = {}
    correct = total = 0
    runs = list(lovo_indices(y))
    for k, (c, hold, tr) in enumerate(runs):
        model = fit_simple(make_model, X[tr], y[tr], len(classes), epochs=epochs, device=device)
        pred = int(predict(model, X[[hold]], device)[0])
        ok = int(pred == y[hold])
        correct += ok
        total += 1
        w = classes[c]
        per_word.setdefault(w, []).append(ok)
        print(f"  [{tag}] {k+1}/{len(runs)} hold-out '{w}' -> "
              f"{'OK' if ok else 'WRONG (pred=' + classes[pred] + ')'}", flush=True)
    acc = correct / max(total, 1)
    return acc, {"total_runs": total, "correct": correct, "per_word": per_word}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/landmarks.npz")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=40, help="epochs per LOVO run")
    ap.add_argument("--final-epochs", type=int, default=80, help="epochs for deployed model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    d = np.load(args.data, allow_pickle=False)
    X, labels = d["X"], d["labels"]
    classes = sorted(set(labels.tolist()))
    cls2id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2id[l] for l in labels])
    n_multi = sum(1 for c in np.unique(y) if (y == c).sum() >= 2)
    print(f"data: {X.shape} | classes: {len(classes)} | multi-variant words: {n_multi} | device: {device}")

    # ---- Protocol 1: leave-one-variant-out, both architectures
    t0 = time.time()
    acc_g, det_g = run_lovo(GRUAttentionClassifier, X, y, classes, args.epochs, device, "GRU")
    t1 = time.time()
    acc_m, det_m = run_lovo(FrameMLPClassifier, X, y, classes, args.epochs, device, "MLP")
    t2 = time.time()

    summary = {
        "protocol": "leave-one-variant-out (words with >=2 sign variants)",
        "gru_attn": {"lovo_accuracy": acc_g, "runs": det_g["total_runs"],
                     "correct": det_g["correct"], "per_word": det_g["per_word"],
                     "train_seconds": round(t1 - t0, 1)},
        "frame_mlp": {"lovo_accuracy": acc_m, "runs": det_m["total_runs"],
                      "correct": det_m["correct"], "per_word": det_m["per_word"],
                      "train_seconds": round(t2 - t1, 1)},
        "num_classes": len(classes),
        "num_sequences": int(X.shape[0]),
        "multi_variant_words": int(n_multi),
    }
    print(f"LOVO SUMMARY: GRU+attn {acc_g:.3f} | frame-MLP {acc_m:.3f} "
          f"({det_g['total_runs']} hold-outs)")
    json.dump(summary, open(os.path.join(args.out, "cv_results.json"), "w", encoding="utf-8"), indent=1)

    # ---- Protocol 2: deployed model on ALL data
    final = fit_simple(GRUAttentionClassifier, X, y, len(classes),
                       epochs=args.final_epochs, device=device)
    train_pred = predict(final, X, device)
    train_acc = float(accuracy_score(y, train_pred))
    train_f1 = float(f1_score(y, train_pred, average="macro", zero_division=0))
    print(f"FULL-VOCAB FIT (sanity, train set): acc={train_acc:.3f} macro-F1={train_f1:.3f}")

    torch.save(final.state_dict(), os.path.join(args.out, "gru_attn_final.pt"))
    json.dump({"classes": classes, "seq_len": SEQ_LEN, "frame_feat": FRAME_FEAT,
               "feature_spec": "v1",
               "full_vocab_train_accuracy": train_acc,
               "full_vocab_train_macro_f1": train_f1,
               "lovo": {k: summary[k] for k in ("gru_attn", "frame_mlp")}},
              open(os.path.join(args.out, "model_meta.json"), "w", encoding="utf-8"), indent=1)
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
