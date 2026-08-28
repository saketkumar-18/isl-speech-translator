"""Frame-level feature normalization.

THIS MODULE IS THE SINGLE SOURCE OF TRUTH for the feature spec. The browser
(`web/features.js`) implements the exact same math; `tests/test_js_parity.py`
proves they agree bit-for-bit (within 1e-6) on shared golden cases.

Spec v1
-------
Input per frame: up to 2 hands. Each hand = 21 MediaPipe landmarks (x, y, z),
image-normalized coords as produced by HandLandmarker on the RAW (unmirrored)
frame, plus a handedness label ("Left"/"Right").

Per hand:
  1. translate so the wrist (landmark 0) is at the origin
  2. scale by s = max over landmarks of ||p_i|| (L2 norm); if s < 1e-9 use 1
     -> translation- and scale-invariant

Hand slots: slot 0 = "Left", slot 1 = "Right" (handedness label). If both
hands report the same label, fall back to geometry: smaller wrist-x = slot 0.
A single hand goes into the slot matching its label; the other slot is zeros.

Frame vector (128-d):
  [ slot0 (63), slot1 (63), presence_slot0, presence_slot1 ]

Sequence: uniform resampling of the detected-frames list to SEQ_LEN frames
(degenerate frames with no hands are kept as zero vectors; the model learns
to use attention over the informative frames).
"""
from __future__ import annotations

import numpy as np

from . import COORDS, FRAME_FEAT, HAND_FEAT, NUM_LANDMARKS, SEQ_LEN


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: (21, 3) float -> wrist-centered, scale-normalized (21, 3)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (NUM_LANDMARKS, COORDS):
        raise ValueError(f"expected ({NUM_LANDMARKS},{COORDS}), got {lm.shape}")
    centered = lm - lm[0]
    s = float(np.max(np.linalg.norm(centered, axis=1)))
    if s < 1e-9:
        s = 1.0
    return centered / s


def order_hands(hands: list[dict]) -> list[tuple[int, dict]]:
    """Assign each detected hand to a slot index (0=Left, 1=Right).

    hands: list of {"landmarks": (21,3), "handedness": "Left"|"Right"}
    Returns list of (slot, hand). Colliding labels fall back to wrist-x order.
    """
    if len(hands) == 0:
        return []
    if len(hands) == 1:
        h = hands[0]
        slot = 0 if str(h.get("handedness", "Left")).lower().startswith("l") else 1
        return [(slot, h)]
    # two hands
    labels = [str(h.get("handedness", "")).lower() for h in hands]
    if labels[0] != labels[1] and all(l in ("left", "right") for l in labels):
        out = []
        for h, l in zip(hands, labels):
            out.append((0 if l == "left" else 1, h))
        return out
    # collision / unknown: geometric fallback (smaller wrist x -> slot 0)
    order = sorted(range(len(hands)), key=lambda i: float(hands[i]["landmarks"][0][0]))
    return [(slot, hands[i]) for slot, i in enumerate(order)]


def frame_vector(hands: list[dict]) -> np.ndarray:
    """Build the 128-d frame feature from detected hands (may be empty)."""
    vec = np.zeros(FRAME_FEAT, dtype=np.float32)
    for slot, hand in order_hands(hands):
        norm = normalize_hand(np.asarray(hand["landmarks"], dtype=np.float64))
        vec[slot * HAND_FEAT:(slot + 1) * HAND_FEAT] = norm.reshape(-1).astype(np.float32)
        vec[2 * HAND_FEAT + slot] = 1.0
    return vec


def resample_sequence(frames: list[np.ndarray], seq_len: int = SEQ_LEN) -> np.ndarray:
    """Uniformly resample a list of frame vectors to (seq_len, FRAME_FEAT).

    Deterministic: index i maps to floor(i * (n-1) / (seq_len-1)). Single-frame
    input is repeated. Empty input yields zeros.
    """
    n = len(frames)
    out = np.zeros((seq_len, FRAME_FEAT), dtype=np.float32)
    if n == 0:
        return out
    if n == 1:
        out[:] = frames[0]
        return out
    idx = np.floor(np.arange(seq_len) * (n - 1) / (seq_len - 1)).astype(int)
    for i, j in enumerate(idx):
        out[i] = frames[j]
    return out


def sequence_from_hands(per_frame_hands: list[list[dict]], seq_len: int = SEQ_LEN) -> np.ndarray:
    """Convenience: per-frame hand lists -> (seq_len, FRAME_FEAT) array."""
    return resample_sequence([frame_vector(h) for h in per_frame_hands], seq_len)
