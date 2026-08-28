# ISL Voice — Indian Sign Language → Speech Translator

**Capstone project · Saket Kumar · IIT Guwahati (Roll 23035010051)**

Real-time translation of **Indian Sign Language (ISL)** hand signs into text
and speech, running **entirely on-device** in the browser. A bidirectional GRU
with attention models the *temporal dynamics* of signs over MediaPipe hand
landmarks; the result is exported to ONNX and executed in-browser via
WebAssembly. No server, no upload, no cost.

> 🤟 **Live demo:** <https://isl-voice.vercel.app> *(URL finalized on deploy)*

---

## 1. Problem & Motivation

India's Deaf community (~18.5 M with hearing disability, Census 2011)
communicates in ISL, which most hearing people do not understand, while
certified interpreters are scarce. The result is exclusion in exactly the
settings where communication matters most — hospitals, emergency services,
banks. An on-device, zero-cost ISL→speech bridge directly addresses this gap.
See [`docs/ETHICS.md`](docs/ETHICS.md) for the full accessibility impact and
ethics statement.

**Scope (stated honestly):** the system recognizes **82 isolated dictionary
signs** from the official ISLRTC data.gov.in dictionary — greetings, needs,
medical/emergency terms, family, daily life, questions, numbers. Continuous
fluent-sentence translation remains an open research problem.

## 2. System Architecture

```
┌────────────────────────── ON-DEVICE (browser) ──────────────────────────┐
│ webcam ──► MediaPipe HandLandmarker (WASM/GPU)                          │
│            21 landmarks × 2 hands per frame                             │
│        ──► features.js  (spec v1: wrist-center, scale-norm, hand slots) │
│            sliding window of 48 frames → (48, 128) tensor               │
│        ──► ONNX Runtime Web (WASM): BiGRU + attention                   │
│            top-k words + per-frame attention weights                    │
│        ──► sentence builder ──► Web Speech API (speechSynthesis, en-IN) │
└──────────────────────────────────────────────────────────────────────────┘

OFFLINE TRAINING PIPELINE (this repo)
  ISLRTC dictionary videos ──► scripts/download_videos.py
      ──► islspeech.extract  (same hand_landmarker.task model as browser)
      ──► islspeech.train    (group-aware 5-fold CV, augmentation)
      ──► islspeech.export_onnx  (torch → ONNX, parity-verified)
```

### 2.1 Feature spec (v1) — the train/serve contract

Per frame, each detected hand (21 MediaPipe landmarks, x,y,z) is
**wrist-centered** and **scaled by its max landmark radius** → translation-
and scale-invariant. Hands are assigned to slots by handedness label
(0=Left, 1=Right; collisions fall back to wrist-x order). Frame vector:
`[slot0(63), slot1(63), pres0, pres1]` = **128-d**. Sequences are uniformly
resampled to **48 frames**.

The Python implementation (`src/islspeech/features.py`) and the browser
implementation (`web/features.js`) are **proven identical** by
`tests/test_js_parity.py`, which runs both on shared golden cases and asserts
max abs diff < 1e-6. This is the production-critical invariant: any drift
would silently collapse accuracy.

### 2.2 Temporal modeling (the capstone's technical angle)

Many ISL signs share hand shapes and differ only in **motion** (*come* vs
*go*). Static pose classifiers cannot separate these. We therefore model the
sequence:

- **Primary model:** input projection → 2-layer **bidirectional GRU** (128 h)
  → **additive attention pooling** → MLP head. Attention weights are
  visualized live in the UI ("which frames the model reads").
- **Ablation baseline:** frame-MLP over mean-pooled features (no temporal
  modeling). The CV gap quantifies the value of sequence modeling.

### 2.3 Evaluation protocol (honest for this data regime)

The curated vocabulary has 82 words / 99 videos: most words have a single
dictionary recording; ~17 words have 2–3 sign variants (different renditions
by native signers). With one video per word, class-stratified k-fold would
leave entire classes unseen in training — a meaningless protocol. We instead
report:

1. **Leave-one-variant-out (LOVO)** — the headline generalization metric.
   For every word with ≥2 variants, each variant is held out in turn and the
   model is trained on everything else (fixed epochs, no peeking). This
   measures exactly the deployment scenario: *a new recording of a known
   vocabulary sign*.
2. **Full-vocabulary fit** — the deployed model trains on all 99 sequences;
   train accuracy is reported as a sanity check, clearly labeled as such.

The frame-MLP baseline (mean-pooled, no temporal modeling) runs the identical
LOVO protocol; the gap quantifies the value of sequence modeling.

## 3. Results

### 3.1 Temporal-perturbation generalization (the temporal-modeling claim)

Each of the 99 training sequences gets 4 **held-out perturbed views** (speed
warp ×0.6–1.5, Gaussian noise, ±17° rotation, scale jitter) that were never
seen in training. Models train on originals only. The frame-MLP is invariant
to frame *order* by construction, so any GRU advantage is attributable to
temporal modeling.

| Model | Train acc | Held-out perturbed views | 
|---|---|---|
| Frame-MLP (no temporal) | 0.838 | 0.669 |
| **BiGRU + attention** | 1.000 | **0.841** |

**Temporal advantage: +17.2 points.** Sequence modeling is what lets the
system recognize a sign performed at a different speed or with perturbation —
exactly the real-world condition (no two performances of a sign are
identical). (`artifacts/temporal_eval.json`)

### 3.2 Leave-one-variant-out (honest generalization ceiling)

For the ~17 words with 2–3 ISLRTC sign variants, each variant is held out in
turn (30 hold-outs). All methods — GRU+attention, frame-MLP, and DTW —
score **0.067**. This is reported deliberately: with ~1.2 videos per class,
*no* classifier generalizes to an unseen recording, and ISLRTC "Sign_2/3"
variants are often genuinely different signs for one concept. The deployed
app therefore targets users performing signs as shown in the dictionary
(its primary use case: someone who learned a sign from the ISLRTC dictionary
and signs it to be understood), and surfaces confidence so uncertain
predictions are visible. Scaling to signer-independent recognition requires
multi-signer corpora — stated as future work. (`artifacts/cv_results.json`)

Deployed model (trained on all 99 sequences): train accuracy 0.980,
macro-F1 0.984 (`artifacts/model_meta.json` — sanity check, not a
generalization claim).

## 4. Repository Layout

```
src/islspeech/        # Python package: features, models, extract, train, export
web/                  # Static web app (index.html, app.js, features.js, styles.css)
  models/             # isl_gru_attn.onnx + labels.json (deployed artifacts)
tests/                # pytest: features, models, ONNX parity, JS/Python parity
scripts/              # download_videos.py, make_demos.py, evaluate_lovo.py, evaluate_temporal.py
docs/ETHICS.md        # Ethics, accessibility & impact statement
models/               # hand_landmarker.task (MediaPipe detector, shared)
```

## 5. Reproduce

```bash
uv venv .venv && source .venv/bin/activate      # or .venv\Scripts\activate
pip install -e ".[dev]" onnxscript
python scripts/download_videos.py               # ISLRTC videos (curated vocab)
python -m islspeech.extract                     # MediaPipe landmarks -> data/landmarks.npz
python -m islspeech.train                       # LOVO + final model -> artifacts/
python scripts/evaluate_temporal.py             # temporal-modeling experiment
python scripts/evaluate_lovo.py                 # DTW baselines on LOVO
python -m islspeech.export_onnx                 # -> web/models/isl_gru_attn.onnx
python scripts/make_demos.py                    # demo landmark clips -> web/demos.json
pytest                                          # 20 tests incl. JS parity
```

Serve locally: `python -m http.server 8000 --directory web` → http://localhost:8000

## 6. Testing

- `tests/test_features.py` — feature-spec unit tests (invariances, slots, resampling)
- `tests/test_models.py` — shapes, masking, determinism, **ONNX export parity**
- `tests/test_js_parity.py` — **Python ↔ JavaScript feature parity** (Node)

CI runs the full suite on every push (`.github/workflows/ci.yml`).

## 7. Limitations & Ethics

Isolated-sign scope; studio-trained (real-world variance will lower accuracy);
two-hand MediaPipe coverage only (no facial/body grammar); regional ISL
dialects partially covered via sign variants. **Assistive prototype — not a
substitute for certified interpreters in medical/legal settings.** Full
discussion: [`docs/ETHICS.md`](docs/ETHICS.md).

## 8. Data & License

- Data: ISLRTC Indian Sign Language dictionary (data.gov.in), used for
  research under its public educational purpose; videos are not re-hosted.
- Hand tracking: MediaPipe Hand Landmarker (Apache-2.0, Google).
- Code: MIT © Saket Kumar.
