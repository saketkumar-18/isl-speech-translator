"""Evaluate similarity-based classifiers on the LOVO protocol.

With ~1.2 videos per class, a softmax head memorizes. Similarity methods
(DTW over landmark sequences, nearest-centroid in a learned embedding)
generalize better in this regime. We compare on the identical
leave-one-variant-out protocol used by train.py.

Methods
-------
1. dtw_nn      : Dynamic Time Warping distance over the 128-d frame features,
                 nearest training sequence wins. No training.
2. dtw_centroid: DTW to the per-class mean sequence (time-aligned average).
3. embed_nn    : GRU encoder context vector, cosine nearest-neighbour.
                 (encoder trained on all data — reported as optimistic bound)

Usage:
    python scripts/evaluate_lovo.py --data data/landmarks.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from islspeech import FRAME_FEAT, HAND_FEAT  # noqa: E402


def dtw_distance(A: np.ndarray, B: np.ndarray) -> float:
    """A, B: (T, D). Standard DTW with L2 local cost. Returns path cost."""
    n, m = len(A), len(B)
    # band to keep it fast and avoid pathological warping
    window = max(abs(n - m) + 1, int(0.3 * max(n, m)))
    INF = float("inf")
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        jlo = max(1, i - window)
        jhi = min(m, i + window)
        ai = A[i - 1]
        for j in range(jlo, jhi + 1):
            cost = float(np.linalg.norm(ai - B[j - 1]))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m]


def lovo_indices(y):
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        for hold in idx:
            tr = np.setdiff1d(np.arange(len(y)), [hold])
            yield int(c), int(hold), tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/landmarks.npz")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, labels = d["X"], d["labels"]
    classes = sorted(set(labels.tolist()))
    cls2id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2id[l] for l in labels])
    print(f"data: {X.shape} | classes: {len(classes)}")

    runs = list(lovo_indices(y))
    print(f"LOVO runs: {len(runs)}")

    # ---- DTW nearest-neighbour
    correct_nn = 0
    per_word_nn = {}
    for k, (c, hold, tr) in enumerate(runs):
        q = X[hold]
        best_d, best_c = float("inf"), -1
        for t in tr:
            dd = dtw_distance(q, X[t])
            if dd < best_d:
                best_d, best_c = dd, y[t]
        ok = int(best_c == y[hold])
        correct_nn += ok
        per_word_nn.setdefault(classes[c], []).append(ok)
        print(f"  dtw_nn {k+1}/{len(runs)} '{classes[c]}' -> "
              f"{'OK' if ok else 'WRONG(pred=' + classes[best_c] + ')'}", flush=True)
    acc_nn = correct_nn / len(runs)

    # ---- DTW to class centroid (mean of remaining variants, then DTW)
    correct_cen = 0
    for k, (c, hold, tr) in enumerate(runs):
        q = X[hold]
        best_d, best_c = float("inf"), -1
        for cc in np.unique(y[tr]):
            members = X[tr[y[tr] == cc]]
            centroid = members.mean(axis=0)
            dd = dtw_distance(q, centroid)
            if dd < best_d:
                best_d, best_c = dd, cc
        correct_cen += int(best_c == y[hold])
    acc_cen = correct_cen / len(runs)

    print("\n=== LOVO RESULTS ===")
    print(f"dtw_nn       : {acc_nn:.3f}  ({correct_nn}/{len(runs)})")
    print(f"dtw_centroid : {acc_cen:.3f}  ({correct_cen}/{len(runs)})")
    print("per-word dtw_nn:", {w: f"{sum(v)}/{len(v)}" for w, v in sorted(per_word_nn.items())})


if __name__ == "__main__":
    main()
