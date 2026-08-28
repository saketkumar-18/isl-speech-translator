"""Extract MediaPipe hand landmarks from ISLRTC videos -> .npz cache.

Uses the MediaPipe Tasks API (HandLandmarker) — the SAME model the browser
loads (hand_landmarker.task, float16), so train-time and inference-time
landmarks come from an identical detector.

Runs offline once; the trained model consumes the cache.

Usage:
    python -m islspeech.extract --videos data/videos --out data/landmarks.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np

from . import FRAME_FEAT, SEQ_LEN
from .features import frame_vector, resample_sequence

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hand_landmarker.task")


def label_from_filename(fn: str) -> str:
    """'thank_you__v0.mp4' -> 'thank you'."""
    base = os.path.splitext(os.path.basename(fn))[0]
    base = re.sub(r"__v\d+$", "", base)
    return base.replace("_", " ").strip().lower()


def make_landmarker(model_path: str):
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(opts)


def extract_video(path: str, landmarker, max_frames: int = 240, clock: list | None = None):
    """Return (per_frame_hands, fps, total_frames).

    `clock` is a single-element list holding the running timestamp (ms) that
    must be monotonically increasing across ALL calls on one landmarker
    instance (MediaPipe VIDEO-mode requirement).
    """
    if clock is None:
        clock = [0.0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(np.ceil(total / max_frames))) if total > max_frames else 1

    per_frame = []
    fi = 0
    dt_ms = 1000.0 / (fps / step) if fps > 0 else 33.0
    dt_ms = max(dt_ms, 1.0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, int(clock[0]))
            hands = []
            if res.hand_landmarks:
                for i, lm_list in enumerate(res.hand_landmarks):
                    lm = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float64)
                    cat = res.handedness[i][0].category_name if res.handedness and i < len(res.handedness) else "Left"
                    hands.append({"landmarks": lm, "handedness": cat})
            per_frame.append(hands)
            clock[0] += dt_ms
        fi += 1
    cap.release()
    clock[0] += 100.0  # gap between videos
    return per_frame, fps, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out", default="data/landmarks.npz")
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"hand landmarker model not found at {model_path}; download it first", file=sys.stderr)
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.videos, "*.mp4")))
    if not files:
        print("no videos found", file=sys.stderr)
        sys.exit(1)

    landmarker = make_landmarker(model_path)

    seqs, labels, names, empty = [], [], [], 0
    clock = [0.0]
    t0 = time.time()
    for i, f in enumerate(files):
        label = label_from_filename(f)
        try:
            per_frame, fps, total = extract_video(f, landmarker, max_frames=args.max_frames, clock=clock)
        except Exception as e:
            print(f"  SKIP {f}: {e}", flush=True)
            continue
        frames = [frame_vector(h) for h in per_frame]
        has_hand = [bool(h) for h in per_frame]
        if not any(has_hand):
            empty += 1
            print(f"  EMPTY {os.path.basename(f)}", flush=True)
            continue
        seq = resample_sequence(frames, SEQ_LEN)
        seqs.append(seq)
        labels.append(label)
        names.append(os.path.basename(f))
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            el = time.time() - t0
            print(f"[{i+1}/{len(files)}] {os.path.basename(f)} -> {label} "
                  f"({len(per_frame)} frames, hand in {sum(has_hand)}) {el:.0f}s", flush=True)

    landmarker.close()
    X = np.stack(seqs).astype(np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, labels=np.array(labels), names=np.array(names))
    meta = {"num_sequences": int(X.shape[0]), "seq_len": SEQ_LEN, "frame_feat": FRAME_FEAT,
            "empty_videos": empty, "labels": sorted(set(labels))}
    json.dump(meta, open(args.out.replace(".npz", "_meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"EXTRACT_DONE sequences={X.shape[0]} classes={len(meta['labels'])} empty={empty}")


if __name__ == "__main__":
    main()
