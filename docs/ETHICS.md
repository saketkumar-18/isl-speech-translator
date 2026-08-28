# Ethics, Accessibility & Impact Statement

**Project:** ISL Voice — Indian Sign Language to Speech Translator
**Author:** Saket Kumar, IIT Guwahati (23035010051)

## 1. Motivation and accessibility impact

India has an estimated 63 million people with significant hearing impairment
(Census 2011: 18.5 million "hearing disabled"; WHO estimates run higher).
Indian Sign Language (ISL) is the primary language of the Deaf community in
India, yet it is not understood by the vast majority of hearing people, and
certified ISL interpreters are scarce — roughly one interpreter per several
thousand Deaf users, concentrated in metros. The asymmetry is the core
accessibility barrier: Deaf users can sign, but hospitals, banks, police
stations, and shops cannot understand them.

A real-time ISL → speech translator on an ordinary phone or laptop directly
attacks this barrier in the highest-stakes settings:

- **Medical:** expressing symptoms, pain, and medication needs (our vocabulary
  deliberately includes *doctor, hospital, medicine, pain, emergency,
  ambulance, danger*).
- **Emergency services:** *help, danger, police, ambulance*.
- **Daily transactions:** *money, buy, sell, work, food, water*.

Because the entire pipeline runs **on-device** (MediaPipe WASM hand tracking +
ONNX/WASM sequence model + Web Speech API), the tool works offline after first
load, costs nothing to operate, and never transmits video — critical for
adoption by users who cannot afford data plans or who distrust cloud services.

## 2. Ethical considerations

### 2.1 Not a replacement for interpreters
This system is an **assistive prototype**, not a certified translation
service. Sign languages carry grammar, facial expression, and spatial
morphology that a hand-landmark classifier cannot capture. We state this
explicitly in the UI footer and here: it must not be used as the sole means of
communication in medical consent, legal proceedings, or any setting where a
mistranslation causes harm.

### 2.2 Word-level scope, honestly stated
The model recognizes **isolated dictionary signs** (82 words from the official
ISLRTC dictionary), not fluent signed sentences. Continuous ISL translation is
an open research problem. We present the system's scope precisely to avoid
overstating capability — a common failure mode of commercial "sign language
AI" marketing that harms trust in assistive technology.

### 2.3 Data provenance and respect for the Deaf community
- Training data is the **official ISLRTC (Indian Sign Language Research and
  Training Centre) dictionary** published on data.gov.in — recorded with
  native signers for public educational use. We do not scrape or re-host the
  videos; only derived landmark features are used.
- The Deaf community's consistent position is that sign languages are full
  natural languages and that technology should *support*, not replace, them.
  This tool is framed as a bridge for hearing interlocutors, not a
  "fix" for Deaf people.

### 2.4 Privacy
No video or landmark data ever leaves the device. There is no backend, no
analytics, no telemetry. The privacy badge in the UI is backed by the
architecture itself (static hosting only).

### 2.5 Fairness and limitations
- **Signer variance:** training data is studio-recorded; real-world lighting,
  skin tones, camera angles, and signing styles vary. Accuracy will be lower
  in the wild; the UI surfaces confidence so users know when the model is
  unsure.
- **Dialectal variation:** ISL has regional variation. The ISLRTC dictionary
  standardizes signs, but users from different states may sign differently;
  multiple sign variants per word (Sign_2, Sign_3…) are included in training
  where available.
- **Two-hand coverage:** MediaPipe tracks at most two hands; signs involving
  face, body, or mouth shapes are out of scope.

## 3. Temporal modeling as the technical contribution

Static hand-pose classification (the majority of student ISL projects) cannot
distinguish signs defined by **motion** — e.g., *come* vs. *go*, which share
hand shapes and differ only in movement direction. This project's core
technical claim is that **sequence modeling is necessary and sufficient to
capture such contrasts**:

- Input: per-frame MediaPipe hand landmarks, wrist-centered and
  scale-normalized (translation/scale invariant).
- Model: bidirectional GRU with additive **attention pooling**, so the model
  learns *which frames matter* for each sign. The attention weights are
  visualized live in the UI — an interpretability feature.
- Ablation: a frame-MLP baseline (mean-pooled, no temporal modeling) is
  trained and reported side-by-side. The accuracy gap quantifies the value of
  temporal modeling on this vocabulary.

## 4. Deployment and sustainability

- Free static hosting (Vercel) + CDN-cached models: zero marginal cost.
- Open-source MIT license; reproducible training (fixed seeds, pinned spec).
- The vocabulary list is data-driven (`labels.json`), so expanding the
  dictionary requires only re-running the pipeline, not code changes.

## 5. References

1. ISLRTC / Ali Yavar Jung National Institute of Speech and Hearing
   Disabilities, *Indian Sign Language Dictionary*, data.gov.in.
2. Census of India 2011, Disability Tables.
3. MediaPipe Hands / Hand Landmarker, Google Research.
4. Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align
   and Translate", ICLR 2015 (attention mechanism).
