"""Fetch an eval set of generators that postdate Community Forensics' training data.

CF's training set (arXiv:2411.04125, collected mid-2024) includes FLUX.1,
Midjourney V5/V6, DALL-E 2/3, Ideogram V1/V2, Firefly 2/3, Imagen 3, DeepFloyd
and the SD family. Everything fetched here shipped after that: these families
are the honest test of whether the detector generalises or just recognises its
own training distribution.

Two source shapes:
- "pair": Rapidata human-preference datasets. Each row holds two images from
  named models (model1/model2); only images whose model matches the filter are
  taken. The datasets-server re-encodes served images as JPEG -- one extra
  web-typical recompression, same as the protocol that built the main corpus.
- "col":  plain one-image-per-row datasets.
- "files": raw repo files fetched via resolve URLs (dataset has no viewer).

Downloads are deduped by content hash: the same underlying image appears in
many Rapidata pairs, and identical bytes must not be counted twice.

Known gaps, so the numbers are read honestly: no usable public source was found
for Midjourney v7 (no API, no scrape on HF) or Ideogram v3 (bitmind/ideogram-27k
is one 37 GB zip; ckoh04's "ideogram-4" folders have unverifiable provenance).
"""
import concurrent.futures as cf
import hashlib
import io
import json
import os
import random
import urllib.parse
import urllib.request

from PIL import Image

OUT = os.environ.get("OUT", os.path.expanduser("~/aidetect-data/corpus-unseen"))
random.seed(20260814)
Image.MAX_IMAGE_PIXELS = 300_000_000

# (name, dataset, mode, image_cols/path, model_substr, n, family)
SOURCES = [
    ("un_gptimage",   "Rapidata/OpenAI-4o_t2i_human_preference",        "pair", None, "4o",       80, "gpt-image-1-4o"),
    ("un_seedream3",  "Rapidata/Seedream-3_t2i_human_preference",       "pair", None, "seedream", 80, "seedream-3"),
    ("un_hunyuan21",  "Rapidata/HunyuanImage-2.1_t2i_human_preference", "pair", None, "hunyuan",  80, "hunyuanimage-2.1"),
    ("un_imagen4",    "Rapidata/Imagen4_t2i_human_preference",          "pair", None, "imagen-4", 80, "imagen-4"),
    ("un_hidream",    "Rapidata/Hidream_t2i_human_preference",          "pair", None, "hidream",  80, "hidream-i1"),
    ("un_aurora",     "Rapidata/xAI_Aurora_t2i_human_preferences",      "pair", None, "aurora",   80, "xai-aurora"),
    ("un_qwenimage",  "Ayush-Singh/qwen-image-base-genaibenchmark",     "col",  "image", None,    80, "qwen-image"),
    ("un_zimage",     "lrzjason/ZImageTurboGen",                        "col",  "image", None,    80, "z-image-turbo"),
    ("un_seedream45", "ash12321/seedream-4.5-generated-2k",             "col",  "image", None,    80, "seedream-4.5"),
    ("un_flux2",      "Sarim-Hash/liars-dividend-flux2-images",         "files", "FLUX.2/fake", None, 100, "flux.2-dev"),
]

UA = {"User-Agent": "Mozilla/5.0 (research-eval)"}


def get_json(url, timeout=90, tries=8):
    delay = 3.0
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                import time as _t
                _t.sleep(delay)
                delay = min(delay * 1.8, 60)
                continue
            raise
        except Exception as e:
            last = e
            import time as _t
            _t.sleep(delay)
            delay = min(delay * 1.8, 60)
    raise last


def total_rows(ds, cfg="default", split="train"):
    url = ("https://datasets-server.huggingface.co/rows?dataset=" + urllib.parse.quote(ds, safe="")
           + f"&config={cfg}&split={split}&offset=0&length=1")
    return get_json(url).get("num_rows_total", 0)


def fetch_rows(ds, offset, length, cfg="default", split="train"):
    url = ("https://datasets-server.huggingface.co/rows?dataset=" + urllib.parse.quote(ds, safe="")
           + f"&config={cfg}&split={split}&offset={offset}&length={length}")
    return get_json(url).get("rows", [])


def download(src):
    req = urllib.request.Request(src, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def save_image(raw, dest):
    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGB")
    if min(im.size) < 96:
        return None
    im.save(dest, "PNG", optimize=False)
    return im.size


def gather_pair(ds, model_substr, want, rng):
    """Collect (src_url, model_string) for images whose model matches.

    The same underlying image appears in many pairs (one per opponent), so the
    downstream hash-dedupe discards most of what is gathered; scan wide.
    """
    tot = total_rows(ds)
    recs, seen_models = [], {}
    per = 25
    hi = max(0, tot - per - 1)
    windows = [rng.randint(0, hi) if hi > 0 else 0 for _ in range(80)]
    for off in windows:
        if len(recs) >= want * 6:
            break
        try:
            rows = fetch_rows(ds, off, per)
        except Exception:
            continue
        for r in rows:
            row = r["row"]
            for i in ("1", "2"):
                m = (row.get(f"model{i}") or "").lower()
                seen_models[m] = seen_models.get(m, 0) + 1
                v = row.get(f"image{i}")
                if model_substr in m and isinstance(v, dict) and "src" in v:
                    recs.append((v["src"], row.get(f"model{i}")))
    print(f"    models seen: {json.dumps(seen_models)}")
    return recs


def gather_col(ds, col, want, rng):
    tot = total_rows(ds)
    recs = []
    per = 25
    hi = max(0, tot - per - 1)
    windows = [rng.randint(0, hi) if hi > 0 else 0 for _ in range(30)]
    for off in windows:
        if len(recs) >= want * 2:
            break
        try:
            rows = fetch_rows(ds, off, per)
        except Exception:
            continue
        for r in rows:
            v = r["row"].get(col)
            if isinstance(v, dict) and "src" in v:
                recs.append((v["src"], None))
    return recs


def gather_files(ds, prefix, want, rng):
    """List raw repo files under prefix via the tree API (paginated)."""
    base = f"https://huggingface.co/api/datasets/{ds}/tree/main/{urllib.parse.quote(prefix)}"
    url = base + "?limit=1000"
    entries = get_json(url)
    paths = [e["path"] for e in entries if e.get("type") == "file"
             and e["path"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    rng.shuffle(paths)
    return [(f"https://huggingface.co/datasets/{ds}/resolve/main/" + urllib.parse.quote(p), None)
            for p in paths[: want * 2]]


def do_source(spec):
    name, ds, mode, arg, model_substr, n, fam = spec
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    manifest_path = os.path.join(d, "_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    existing = [f for f in os.listdir(d) if f.endswith(".png")]
    if len(existing) >= n:
        return name, len(existing), "cached"

    # Per-source, per-round RNG: a rerun (ROUND=1, 2, ...) explores different
    # windows instead of re-fetching the same rows into the hash-dedupe.
    rng = random.Random(f"{name}|{os.environ.get('ROUND', '0')}|20260814")
    if mode == "pair":
        recs = gather_pair(ds, model_substr, n, rng)
    elif mode == "col":
        recs = gather_col(ds, arg, n, rng)
    else:
        recs = gather_files(ds, arg, n, rng)

    hashes = set(m.get("sha256") for m in manifest.values())
    idx = len(existing)
    for src, model in recs:
        if idx >= n:
            break
        try:
            raw = download(src)
        except Exception:
            continue
        h = hashlib.sha256(raw).hexdigest()
        if h in hashes:
            continue
        fname = f"{name}_{idx:05d}.png"
        size = None
        try:
            size = save_image(raw, os.path.join(d, fname))
        except Exception:
            continue
        if size is None:
            continue
        hashes.add(h)
        manifest[fname] = {"sha256": h, "model": model, "bytes": len(raw),
                           "size": list(size), "url_hint": src.split("?")[0][-120:]}
        idx += 1
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    final = len([f for f in os.listdir(d) if f.endswith(".png")])
    return name, final, "ok"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    labels = {}
    total = 0
    for spec in SOURCES:
        name, count, status = do_source(spec)
        print(f"{name:16s} {count:4d}  {status}", flush=True)
        total += count
        labels[name] = {"label": 1, "family": spec[6], "dataset": spec[1],
                        "column": spec[3] or "", "mode": spec[2]}
    with open(os.path.join(OUT, "_labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print("TOTAL", total)
