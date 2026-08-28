/**
 * e2e_infer.mjs — end-to-end check of the EXACT browser pipeline in Node:
 *   demos.json (real ISLRTC landmark sequences)
 *   -> web/features.js (the browser's feature code)
 *   -> web/models/isl_gru_attn.onnx (the deployed model)
 * Asserts the top-1 prediction matches each demo's true label.
 */
import { readFileSync, copyFileSync } from "fs";
import ort from "onnxruntime-node";
copyFileSync("web/features.js", "features_e2e.mjs");
const { FRAME_FEAT, SEQ_LEN, frameVector, resampleSequence, presenceMask } =
  await import("./features_e2e.mjs");

const demos = JSON.parse(readFileSync("web/demos.json", "utf-8"));
const labels = JSON.parse(readFileSync("web/models/labels.json", "utf-8")).classes;

const session = await ort.InferenceSession.create("web/models/isl_gru_attn.onnx");

let ok = 0, n = 0;
for (const [word, demo] of Object.entries(demos)) {
  const frames = demo.frames.map(fr =>
    frameVector(fr.map(h => ({ landmarks: Float64Array.from(h.lm), handedness: h.h })))
  );
  const flat = resampleSequence(frames, SEQ_LEN);
  const mask = presenceMask(flat, SEQ_LEN);
  const feeds = {
    frames: new ort.Tensor("float32", flat, [1, SEQ_LEN, FRAME_FEAT]),
    mask: new ort.Tensor("float32", mask, [1, SEQ_LEN]),
  };
  const out = await session.run(feeds);
  const probs = Array.from(out.probs.data);
  const i1 = probs.indexOf(Math.max(...probs));
  const pred = labels[i1];
  const hit = pred === word;
  ok += hit ? 1 : 0; n++;
  const top3 = probs.map((p, i) => [p, labels[i]]).sort((a, b) => b[0] - a[0]).slice(0, 3)
    .map(([p, l]) => `${l}=${(p * 100).toFixed(1)}%`).join(" ");
  console.log(`${hit ? "OK " : "BAD"} ${word.padEnd(12)} -> ${pred.padEnd(12)} (${top3})`);
}
console.log(`\nE2E: ${ok}/${n} demos recognized correctly`);
process.exit(ok === n ? 0 : 1);
