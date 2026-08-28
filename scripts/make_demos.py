"""Export a few real landmark sequences as web/demos.json (demo buttons).

Picks one video per chosen word, extracts raw per-frame hands with the
MediaPipe Tasks HandLandmarker (downsampled to <= 24 frames), and writes
JSON the browser feeds through the same features.js pipeline.

Usage:
    python scripts/make_demos.py --videos data/videos --out web/demos.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

DEMO_WORDS = {
    "hello": "hello__v0.mp4",
    "thank you": "thank_you__v0.mp4",
    "water": "water__v0.mp4",
    "help": "help__v0.mp4",
    "yes": "yes__v0.mp4",
    "sorry": "sorry__v0.mp4",
}
MAX_FRAMES = 24


def extract(path, landmarker, clock):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(np.ceil(total / MAX_FRAMES)))
    frames = []
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
                    lm = [c for p in lm_list for c in (p.x, p.y, p.z)]
                    cat = res.handedness[i][0].category_name if res.handedness and i < len(res.handedness) else "Left"
                    hands.append({"lm": [round(v, 6) for v in lm], "h": cat})
            frames.append(hands)
            clock[0] += dt_ms
        fi += 1
    cap.release()
    clock[0] += 100.0
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out", default="web/demos.json")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)
    demos = {}
    clock = [0.0]
    for word, fn in DEMO_WORDS.items():
        path = os.path.join(args.videos, fn)
        if not os.path.exists(path):
            print("skip (missing):", fn)
            continue
        frames = extract(path, landmarker, clock)
        if any(frames):
            demos[word] = {"frames": frames}
            print(f"{word}: {len(frames)} frames")
    landmarker.close()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(demos, f)
    print(f"DEMOS_DONE {len(demos)} words -> {args.out} ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
