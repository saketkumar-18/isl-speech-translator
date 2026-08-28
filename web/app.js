/**
 * app.js — ISL Voice main application.
 *
 * Pipeline (all on-device):
 *   webcam frame -> MediaPipe HandLandmarker (JS/WASM) -> per-frame hands
 *   -> features.js (spec v1, parity-tested vs Python) -> sliding window of 48 frames
 *   -> ONNX GRU+attention (onnxruntime-web WASM) -> top-k words + attention weights
 *   -> sentence builder -> Web Speech API (speechSynthesis)
 */
import { FRAME_FEAT, SEQ_LEN, frameVector, resampleSequence, presenceMask } from "./features.js";

const ort = window.ort; // onnxruntime-web loaded via script tag (global `ort`)

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), camStatus = $("camStatus");
const liveSign = $("liveSign"), liveSignText = $("liveSignText"), liveSignConf = $("liveSignConf");
const sentenceEl = $("sentence"), topKEl = $("topK"), attnCanvas = $("attnCanvas");

const CONF_COMMIT = 0.70;      // min probability to auto-commit a word
const STABLE_HITS = 3;         // consecutive windows agreeing before commit
const COMMIT_COOLDOWN_MS = 1800;
const INFER_EVERY = 6;         // run model every N new frames

let session = null;            // ort.InferenceSession
let labels = [];               // class names
let landmarker = null;         // MediaPipe HandLandmarker
let running = false;
let frameBuf = [];             // ring buffer of Float32Array(128)
let framesSinceInfer = 0;
let stableClass = null, stableCount = 0, lastCommitAt = 0, lastCommitted = null;
let sentence = [];
let demos = null;

// ------------------------------------------------------------------ bootstrap
async function init() {
  const meta = await (await fetch("models/labels.json")).json();
  labels = meta.classes;
  $("vocabCount").textContent = labels.length;
  renderVocab();
  loadDemos();

  session = await ort.InferenceSession.create("models/isl_gru_attn.onnx", {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  $("modelBadge").textContent = `GRU+Attn · ${labels.length} signs`;
}

function renderVocab() {
  const grid = $("vocabList");
  grid.innerHTML = "";
  for (const w of labels) {
    const d = document.createElement("div");
    d.className = "vword";
    d.textContent = w;
    grid.appendChild(d);
  }
}

async function loadDemos() {
  try {
    demos = await (await fetch("demos.json")).json();
  } catch { demos = {}; }
  const wrap = $("demoButtons");
  for (const label of Object.keys(demos)) {
    const b = document.createElement("button");
    b.textContent = `🎬 ${label}`;
    b.addEventListener("click", () => runDemo(label));
    wrap.appendChild(b);
  }
}

// ------------------------------------------------------------------ mediapipe
async function initLandmarker() {
  const vision = await window.FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await window.HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    numHands: 2,
    runningMode: "VIDEO",
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
  });
}

// ------------------------------------------------------------------ camera
$("btnStart").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    camStatus.querySelector("p").textContent =
      "Camera unavailable or permission denied. You can still try the demo signs below.";
    return;
  }
  if (!landmarker) await initLandmarker();
  camStatus.hidden = true;
  liveSign.hidden = false;
  $("btnStop").hidden = false;
  $("btnStart").hidden = true;
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  running = true;
  frameBuf = [];
  requestAnimationFrame(loop);
});

$("btnStop").addEventListener("click", () => {
  running = false;
  const tracks = video.srcObject?.getTracks() || [];
  tracks.forEach(t => t.stop());
  video.srcObject = null;
  camStatus.hidden = false;
  liveSign.hidden = true;
  $("btnStop").hidden = true;
  $("btnStart").hidden = false;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
});

let lastVideoTime = -1;
function loop() {
  if (!running) return;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = landmarker.detectForVideo(video, performance.now());
    const hands = [];
    if (res.landmarks) {
      for (let i = 0; i < res.landmarks.length; i++) {
        const lm = new Float64Array(63);
        for (let j = 0; j < 21; j++) {
          lm[j * 3] = res.landmarks[i][j].x;
          lm[j * 3 + 1] = res.landmarks[i][j].y;
          lm[j * 3 + 2] = res.landmarks[i][j].z;
        }
        const cat = res.handedness?.[i]?.[0];
        hands.push({ landmarks: lm, handedness: cat ? cat.categoryName : "Left" });
      }
    }
    drawHands(res);
    frameBuf.push(frameVector(hands));
    if (frameBuf.length > SEQ_LEN) frameBuf.shift();
    framesSinceInfer++;
    if (framesSinceInfer >= INFER_EVERY && frameBuf.length >= SEQ_LEN) {
      framesSinceInfer = 0;
      inferWindow();
    }
  }
  requestAnimationFrame(loop);
}

function drawHands(res) {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!res.landmarks) return;
  const W = overlay.width, H = overlay.height;
  const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
                [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  ctx.lineWidth = 2;
  for (const lm of res.landmarks) {
    ctx.strokeStyle = "#35d0a5";
    for (const [a, b] of CONN) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x * W, lm[a].y * H);
      ctx.lineTo(lm[b].x * W, lm[b].y * H);
      ctx.stroke();
    }
    ctx.fillStyle = "#4da3ff";
    for (const p of lm) {
      ctx.beginPath();
      ctx.arc(p.x * W, p.y * H, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ------------------------------------------------------------------ inference
async function inferWindow(frames = frameBuf) {
  const flat = resampleSequence(frames, SEQ_LEN);
  const mask = presenceMask(flat, SEQ_LEN);
  const feeds = {
    frames: new ort.Tensor("float32", flat, [1, SEQ_LEN, FRAME_FEAT]),
    mask: new ort.Tensor("float32", mask, [1, SEQ_LEN]),
  };
  const t0 = performance.now();
  const out = await session.run(feeds);
  const dt = performance.now() - t0;
  const probs = out.probs.data;
  const attn = out.attention.data;
  const idx = Array.from(probs).map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);

  const [p1, i1] = idx[0];
  liveSignText.textContent = labels[i1];
  liveSignConf.textContent = `${(p1 * 100).toFixed(0)}% · ${dt.toFixed(0)} ms`;
  drawAttention(attn);
  renderTopK(idx.slice(0, 5));

  // commit logic
  if (p1 >= CONF_COMMIT) {
    if (stableClass === i1) stableCount++; else { stableClass = i1; stableCount = 1; }
  } else {
    stableClass = null; stableCount = 0;
  }
  const now = performance.now();
  if (stableCount >= STABLE_HITS && now - lastCommitAt > COMMIT_COOLDOWN_MS) {
    const word = labels[i1];
    if (word !== lastCommitted || now - lastCommitAt > 4000) {
      commitWord(word);
      lastCommitted = word;
    }
    stableCount = 0;
    lastCommitAt = now;
  }
}

function drawAttention(attn) {
  const ctx = attnCanvas.getContext("2d");
  const W = attnCanvas.width, H = attnCanvas.height;
  ctx.clearRect(0, 0, W, H);
  const n = attn.length;
  const bw = W / n;
  let mx = 0;
  for (let i = 0; i < n; i++) mx = Math.max(mx, attn[i]);
  for (let i = 0; i < n; i++) {
    const h = mx > 0 ? (attn[i] / mx) * (H - 4) : 0;
    ctx.fillStyle = `rgba(53, 208, 165, ${0.25 + 0.75 * (mx > 0 ? attn[i] / mx : 0)})`;
    ctx.fillRect(i * bw + 1, H - h, bw - 2, h);
  }
}

function renderTopK(top) {
  topKEl.innerHTML = "";
  for (const [p, i] of top) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="p">${labels[i]}</span> — ${(p * 100).toFixed(1)}%`;
    topKEl.appendChild(li);
  }
}

// ------------------------------------------------------------------ sentence
function commitWord(word) {
  sentence.push(word);
  renderSentence();
}

function renderSentence() {
  sentenceEl.innerHTML = "";
  if (sentence.length === 0) {
    sentenceEl.innerHTML = '<span class="placeholder">Sign in front of the camera — recognized words appear here…</span>';
    return;
  }
  for (const w of sentence) {
    const chip = document.createElement("span");
    chip.className = "word-chip";
    chip.textContent = w;
    sentenceEl.appendChild(chip);
  }
}

$("btnClear").addEventListener("click", () => { sentence = []; lastCommitted = null; renderSentence(); });

$("btnSpeak").addEventListener("click", () => {
  if (sentence.length === 0) return;
  const text = sentence.join(" ");
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-IN";
  u.rate = 0.95;
  const voices = speechSynthesis.getVoices();
  const hi = voices.find(v => v.lang === "hi-IN") || voices.find(v => v.lang.startsWith("en-IN"));
  if (hi) u.voice = hi;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
});

// ------------------------------------------------------------------ demo mode
async function runDemo(label) {
  const demo = demos[label];
  // demo.frames: array of frames; each frame = array of hands {lm: number[63], h: "Left"|"Right"}
  const frames = demo.frames.map(fr =>
    frameVector(fr.map(h => ({ landmarks: Float64Array.from(h.lm), handedness: h.h })))
  );
  await inferWindow(frames);
  // demo bypasses commit stability: add directly if confident
  const flat = resampleSequence(frames, SEQ_LEN);
  const feeds = {
    frames: new ort.Tensor("float32", flat, [1, SEQ_LEN, FRAME_FEAT]),
    mask: new ort.Tensor("float32", presenceMask(flat, SEQ_LEN), [1, SEQ_LEN]),
  };
  const out = await session.run(feeds);
  const probs = Array.from(out.probs.data);
  const i1 = probs.indexOf(Math.max(...probs));
  commitWord(labels[i1]);
}

init().catch(e => {
  console.error(e);
  camStatus.querySelector("p").textContent = "Failed to load model: " + e.message;
});
