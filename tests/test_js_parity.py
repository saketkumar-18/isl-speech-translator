"""JS <-> Python feature-spec parity test.

Generates golden cases (random hands per frame), computes frame vectors and
resampled sequences with the PYTHON implementation, then runs the exact same
inputs through web/features.js in Node and asserts max abs diff < 1e-6.

This is the production-critical test: the browser must compute identical
features to what the model was trained on, or accuracy collapses.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from islspeech import FRAME_FEAT, SEQ_LEN
from islspeech.features import frame_vector, resample_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_JS = os.path.join(ROOT, "web", "features.js")

RUNNER = r"""
import { readFileSync } from "fs";
import { pathToFileURL } from "url";

const { frameVector, resampleSequence } = await import(pathToFileURL(process.argv[2]).href);

const golden = JSON.parse(readFileSync(process.argv[3], "utf-8"));
const out = { frames: [], seq: null };

const pyFrames = [];
for (const fr of golden.frames) {
  const hands = fr.map(h => ({ landmarks: Float64Array.from(h.lm), handedness: h.h }));
  const v = frameVector(hands);
  out.frames.push(Array.from(v));
  pyFrames.push(v);
}
out.seq = Array.from(resampleSequence(pyFrames, golden.seq_len));
process.stdout.write(JSON.stringify(out));
"""


def make_golden(n_frames=37, seed=1234):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_frames):
        n_hands = int(rng.integers(0, 3))  # 0, 1, or 2 hands
        hands = []
        for j in range(n_hands):
            lm = rng.normal(0.5, 0.15, size=(21, 3))
            h = rng.choice(["Left", "Right", "Right"])  # includes collisions
            hands.append({"lm": lm.reshape(-1).tolist(), "h": str(h)})
        frames.append(hands)
    return {"frames": frames, "seq_len": SEQ_LEN}


def node_available():
    return shutil.which("node") is not None


@pytest.mark.skipif(not node_available(), reason="node not installed")
def test_js_python_parity():
    golden = make_golden()

    # Python reference
    py_frame_vecs = []
    for fr in golden["frames"]:
        hands = [{"landmarks": np.array(h["lm"], dtype=np.float64).reshape(21, 3),
                  "handedness": h["h"]} for h in fr]
        py_frame_vecs.append(frame_vector(hands))
    py_seq = resample_sequence(py_frame_vecs, SEQ_LEN)

    with tempfile.TemporaryDirectory() as td:
        # features.js uses ESM exports -> copy to .mjs so Node treats it as ESM
        mjs = os.path.join(td, "features.mjs")
        shutil.copyfile(FEATURES_JS, mjs)
        runner = os.path.join(td, "runner.mjs")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(RUNNER)
        golden_path = os.path.join(td, "golden.json")
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(golden, f)

        res = subprocess.run(
            ["node", runner, mjs, golden_path],
            capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, f"node failed: {res.stderr}"
        js = json.loads(res.stdout)

    # compare frame vectors
    assert len(js["frames"]) == len(py_frame_vecs)
    max_diff = 0.0
    for jv, pv in zip(js["frames"], py_frame_vecs):
        d = float(np.max(np.abs(np.array(jv, dtype=np.float64) - pv.astype(np.float64))))
        max_diff = max(max_diff, d)
    assert max_diff < 1e-6, f"frame vector parity diff {max_diff}"

    # compare resampled sequence
    js_seq = np.array(js["seq"], dtype=np.float64).reshape(SEQ_LEN, FRAME_FEAT)
    seq_diff = float(np.max(np.abs(js_seq - py_seq.astype(np.float64))))
    assert seq_diff < 1e-6, f"sequence parity diff {seq_diff}"
    print(f"parity OK: frame max diff={max_diff:.2e}, seq max diff={seq_diff:.2e}")


@pytest.mark.skipif(not node_available(), reason="node not installed")
def test_js_python_parity_edge_cases():
    """Empty frames, single frame, all-empty sequence."""
    golden = {"frames": [[], [], []], "seq_len": SEQ_LEN}
    with tempfile.TemporaryDirectory() as td:
        mjs = os.path.join(td, "features.mjs")
        shutil.copyfile(FEATURES_JS, mjs)
        runner = os.path.join(td, "runner.mjs")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(RUNNER)
        golden_path = os.path.join(td, "golden.json")
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(golden, f)
        res = subprocess.run(["node", runner, mjs, golden_path],
                             capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stderr
        js = json.loads(res.stdout)
    js_seq = np.array(js["seq"], dtype=np.float64)
    assert np.all(js_seq == 0)
    for fv in js["frames"]:
        assert np.all(np.array(fv) == 0)
