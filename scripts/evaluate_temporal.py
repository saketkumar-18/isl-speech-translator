"""Temporal-perturbation generalization: where sequence modeling provably matters.

Protocol
--------
For every sequence we synthesize K *unseen* perturbed views (speed warp,
Gaussian noise, rotation, scale jitter, mirror) that were never in training.
We train each architecture on the ORIGINAL sequences only and evaluate on the
held-out perturbed views. This isolates the question the capstone cares about:
does modeling temporal order help recognize a sign performed differently?

A frame-MLP (mean-pool) is invariant to frame ORDER by construction, so any
advantage of the GRU+attention here is attributable to temporal modeling.

Usage:
    python scripts/evaluate_temporal.py --data data/landmarks.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from islspeech import FRAME_FEAT, HAND_FEAT, SEQ_LEN  # noqa: E402
from islspeech.models import FrameMLPClassifier, GRUAttentionClassifier  # noqa: E402
from islspeech.train import SeqDataset, train_one_epoch, _resample  # noqa: E402

RNG = np.random.default_rng(7)


def perturb(seq: np.ndarray, rng: np.random.Generator, strong: bool = False) -> np.ndarray:
    """A *deterministic-holdout* perturbation (not the train-time augment)."""
    out = seq.copy()
    T = out.shape[0]
    # speed warp (temporal!)
    speed = float(rng.uniform(0.6, 1.5) if strong else rng.uniform(0.8, 1.25))
    n_eff = max(2, int(round(T / speed)))
    idx = np.clip(np.floor(np.arange(T) * (n_eff - 1) / (T - 1)).astype(int), 0, T - 1)
    out = _resample(out[idx], T)
    # noise
    out[:, :2 * HAND_FEAT] += rng.normal(0, 0.02 if strong else 0.01,
                                         size=(T, 2 * HAND_FEAT)).astype(np.float32)
    # rotation
    theta = float(rng.uniform(-0.3, 0.3) if strong else rng.uniform(-0.15, 0.15))
    c, s = np.cos(theta), np.sin(theta)
    for slot in range(2):
        base = slot * HAND_FEAT
        xs = out[:, base + 0::3].copy()
        ys = out[:, base + 1::3].copy()
        out[:, base + 0::3] = c * xs - s * ys
        out[:, base + 1::3] = s * xs + c * ys
    # scale
    sc = float(rng.uniform(0.8, 1.2))
    out[:, :2 * HAND_FEAT] *= sc
    return out


def fit(make_model, X, y, nc, epochs, device, batch=32, lr=2e-3):
    model = make_model(nc).to(device)
    loader = DataLoader(SeqDataset(X, y, False), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        train_one_epoch(model, loader, opt, device)
        sched.step()
    return model


@torch.no_grad()
def acc_on(model, X, y, device):
    model.eval()
    ds = SeqDataset(X, y, False)
    loader = DataLoader(ds, batch_size=64)
    preds, ys = [], []
    for x, m, yy in loader:
        logits, _ = model(x.to(device), m.to(device))
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(yy.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    return float((preds == ys).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/landmarks.npz")
    ap.add_argument("--views", type=int, default=4, help="held-out perturbed views per sequence")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    d = np.load(args.data, allow_pickle=False)
    X, labels = d["X"], d["labels"]
    classes = sorted(set(labels.tolist()))
    cls2id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2id[l] for l in labels])
    print(f"data: {X.shape} | classes: {len(classes)} | device: {device}")

    # build held-out perturbed views
    Xv, yv = [], []
    for i in range(len(X)):
        for _ in range(args.views):
            Xv.append(perturb(X[i], RNG))
            yv.append(y[i])
    Xv = np.stack(Xv).astype(np.float32)
    yv = np.array(yv)
    print(f"held-out perturbed views: {Xv.shape}")

    results = {}
    for name, mk in [("gru_attn", GRUAttentionClassifier), ("frame_mlp", FrameMLPClassifier)]:
        model = fit(mk, X, y, len(classes), args.epochs, device)
        train_acc = acc_on(model, X, y, device)
        pert_acc = acc_on(model, Xv, yv, device)
        results[name] = {"train_acc": train_acc, "perturbed_acc": pert_acc}
        print(f"{name:10s} train={train_acc:.3f}  perturbed={pert_acc:.3f}", flush=True)

    gap = results["gru_attn"]["perturbed_acc"] - results["frame_mlp"]["perturbed_acc"]
    print(f"\nTEMPORAL ADVANTAGE (GRU - MLP on perturbed views): {gap:+.3f}")
    import json
    json.dump(results | {"temporal_advantage": gap, "views_per_seq": args.views},
              open("artifacts/temporal_eval.json", "w"), indent=1)
    print("TEMPORAL_EVAL_DONE")


if __name__ == "__main__":
    main()
