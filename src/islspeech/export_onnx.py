"""Export the trained PyTorch model to ONNX for on-device (WASM) inference.

Usage:
    python -m islspeech.export_onnx --meta artifacts/model_meta.json \
        --weights artifacts/gru_attn_final.pt --out web/models/
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import FRAME_FEAT, SEQ_LEN
from .models import GRUAttentionClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="artifacts/model_meta.json")
    ap.add_argument("--weights", default="artifacts/gru_attn_final.pt")
    ap.add_argument("--out", default="web/models")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(args.meta, encoding="utf-8"))
    classes = meta["classes"]
    model = GRUAttentionClassifier(len(classes))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, SEQ_LEN, FRAME_FEAT)
    dummy_mask = torch.ones(1, SEQ_LEN)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, mask):
            logits, attn = self.m(x, mask)
            probs = torch.softmax(logits, dim=-1)
            return probs, attn

    onnx_path = os.path.join(args.out, "isl_gru_attn.onnx")
    torch.onnx.export(Wrapper(model), (dummy, dummy_mask), onnx_path,
                      input_names=["frames", "mask"], output_names=["probs", "attention"],
                      opset_version=17, do_constant_folding=True)

    # verify with onnxruntime + parity vs torch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    x = np.random.default_rng(0).normal(size=(1, SEQ_LEN, FRAME_FEAT)).astype(np.float32)
    m = np.ones((1, SEQ_LEN), dtype=np.float32)
    probs_ort, attn_ort = sess.run(None, {"frames": x, "mask": m})
    with torch.no_grad():
        logits, attn_t = model(torch.from_numpy(x), torch.from_numpy(m))
        probs_t = torch.softmax(logits, -1).numpy()
    diff = float(np.max(np.abs(probs_ort - probs_t)))
    print(f"onnx<->torch max prob diff: {diff:.2e}")
    assert diff < 1e-4, "ONNX export parity check failed"

    # ship labels + metadata next to the model
    json.dump({"classes": classes, "seq_len": SEQ_LEN, "frame_feat": FRAME_FEAT,
               "feature_spec": "v1"},
              open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8"), indent=1)
    sz = os.path.getsize(onnx_path)
    print(f"EXPORT_DONE {onnx_path} ({sz/1024:.0f} KB, {len(classes)} classes)")


if __name__ == "__main__":
    main()
