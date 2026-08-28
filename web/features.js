/**
 * features.js — browser mirror of src/islspeech/features.py (spec v1).
 *
 * SINGLE SOURCE OF TRUTH is the Python module; this file must stay in
 * lock-step. tests/test_js_parity.py runs both implementations on shared
 * golden cases and asserts max abs diff < 1e-6.
 *
 * Spec v1:
 *  - per hand: wrist-centered, scaled by max L2 norm of centered landmarks
 *  - slots: 0 = "Left", 1 = "Right" (handedness label); collision -> wrist-x order
 *  - frame vector (128): [slot0(63), slot1(63), pres0, pres1]
 *  - sequence: uniform resample to SEQ_LEN frames, index floor(i*(n-1)/(L-1))
 */
export const NUM_LANDMARKS = 21;
export const COORDS = 3;
export const HAND_FEAT = NUM_LANDMARKS * COORDS; // 63
export const FRAME_FEAT = 2 * HAND_FEAT + 2;     // 128
export const SEQ_LEN = 48;

/** landmarks: Float64Array/Array of length 63 (x,y,z interleaved) -> normalized Float64Array(63) */
export function normalizeHand(lm) {
  const wx = lm[0], wy = lm[1], wz = lm[2];
  const centered = new Float64Array(63);
  let s = 0;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const cx = lm[i * 3] - wx;
    const cy = lm[i * 3 + 1] - wy;
    const cz = lm[i * 3 + 2] - wz;
    centered[i * 3] = cx; centered[i * 3 + 1] = cy; centered[i * 3 + 2] = cz;
    const n = Math.sqrt(cx * cx + cy * cy + cz * cz);
    if (n > s) s = n;
  }
  if (s < 1e-9) s = 1.0;
  for (let i = 0; i < 63; i++) centered[i] /= s;
  return centered;
}

/**
 * hands: [{landmarks: ArrayLike(63), handedness: "Left"|"Right"}]
 * Returns [[slot, hand], ...]
 */
export function orderHands(hands) {
  if (hands.length === 0) return [];
  if (hands.length === 1) {
    const h = hands[0];
    const slot = String(h.handedness || "Left").toLowerCase().startsWith("l") ? 0 : 1;
    return [[slot, h]];
  }
  const labels = hands.map(h => String(h.handedness || "").toLowerCase());
  if (labels[0] !== labels[1] && labels.every(l => l === "left" || l === "right")) {
    return hands.map((h, i) => [labels[i] === "left" ? 0 : 1, h]);
  }
  // collision/unknown: smaller wrist x -> slot 0
  const order = [0, 1].sort((a, b) => hands[a].landmarks[0] - hands[b].landmarks[0]);
  return order.map((idx, slot) => [slot, hands[idx]]);
}

/** Build the 128-d frame vector (Float32Array). */
export function frameVector(hands) {
  const vec = new Float32Array(FRAME_FEAT);
  for (const [slot, hand] of orderHands(hands)) {
    const norm = normalizeHand(hand.landmarks);
    vec.set(norm, slot * HAND_FEAT);
    vec[2 * HAND_FEAT + slot] = 1.0;
  }
  return vec;
}

/** Uniform resample of frame vectors to (SEQ_LEN, FRAME_FEAT) -> Float32Array(SEQ_LEN*FRAME_FEAT). */
export function resampleSequence(frames, seqLen = SEQ_LEN) {
  const out = new Float32Array(seqLen * FRAME_FEAT);
  const n = frames.length;
  if (n === 0) return out;
  if (n === 1) {
    for (let i = 0; i < seqLen; i++) out.set(frames[0], i * FRAME_FEAT);
    return out;
  }
  for (let i = 0; i < seqLen; i++) {
    const j = Math.floor(i * (n - 1) / (seqLen - 1));
    out.set(frames[j], i * FRAME_FEAT);
  }
  return out;
}

/** Presence mask (1 where any hand) -> Float32Array(seqLen). */
export function presenceMask(seqFlat, seqLen = SEQ_LEN) {
  const mask = new Float32Array(seqLen);
  for (let i = 0; i < seqLen; i++) {
    const base = i * FRAME_FEAT;
    mask[i] = (seqFlat[base + 2 * HAND_FEAT] + seqFlat[base + 2 * HAND_FEAT + 1]) > 0 ? 1 : 0;
  }
  return mask;
}
