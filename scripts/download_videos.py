"""Download curated ISLRTC videos (primary + sign variants) from Hugging Face."""
import json, os, re, sys, time, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.environ['LOCALAPPDATA'] + os.sep + 'Temp'
vocab = json.load(open(os.path.join(TMP, 'curated_vocab.json'), encoding='utf-8'))
idx = json.load(open(os.path.join(TMP, 'islrtc_index.json'), encoding='utf-8'))

BASE = "https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded/resolve/main/"
DEST = os.path.join(ROOT, "data", "videos")
os.makedirs(DEST, exist_ok=True)

# find variants: same base name with _(Sign_N) suffix
all_words = {}
for letter, paths in idx.items():
    for path in paths:
        fn = os.path.splitext(os.path.basename(path))[0]
        all_words[fn] = path

def variants_of(fn):
    """Return [fn, fn_(Sign_2), fn_(Sign_3)...] that exist."""
    out = [fn] if fn in all_words else []
    for i in range(2, 5):
        v = f"{fn}_(Sign_{i})"
        if v in all_words:
            out.append(v)
    return out

plan = []  # (label, variant_idx, remote_path, local_path)
for label, rel in sorted(vocab.items()):
    fn = os.path.splitext(os.path.basename(rel))[0]
    vs = variants_of(fn)
    for vi, v in enumerate(vs[:3]):  # cap 3 variants per word
        remote = all_words[v]
        safe = re.sub(r'[^\w\- ]', '', label).strip().replace(' ', '_')
        local = os.path.join(DEST, f"{safe}__v{vi}.mp4")
        plan.append((label, vi, remote, local))

print(f"plan: {len(plan)} videos for {len(vocab)} words")
json.dump([(l, vi, r, lp) for l, vi, r, lp in plan],
          open(os.path.join(DEST, '_plan.json'), 'w', encoding='utf-8'), indent=1)

done = 0
for label, vi, remote, local in plan:
    if os.path.exists(local) and os.path.getsize(local) > 10000:
        done += 1
        continue
    url = BASE + urllib.parse.quote(remote)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, local)
            sz = os.path.getsize(local)
            if sz < 10000:
                raise IOError(f"too small: {sz}")
            done += 1
            print(f"[{done}/{len(plan)}] {label} v{vi} ({sz//1024} KB)", flush=True)
            break
        except Exception as e:
            print(f"  retry {attempt+1} {label}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    else:
        print(f"FAILED: {label} v{vi} {remote}", flush=True)

print("DOWNLOAD_COMPLETE", done, "/", len(plan))
