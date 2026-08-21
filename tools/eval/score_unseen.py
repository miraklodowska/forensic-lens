"""Score the unseen-generator corpus with the FROZEN shipped calibration.

No refitting happens here, deliberately: the question is what the shipped
extension does on generators its models never saw, so the constants come
straight from models/pipeline.json.

Before scoring anything new, the same fusion math is run over the main
corpus's cached logits and must reproduce the in-sample balanced accuracy
recorded in the calibration file to four decimals. If that check fails, the
scoring code is wrong and every downstream number would be noise.

Outputs, always as separate numbers, never blended:
- in-distribution bACC (main corpus, generators overlapping CF training data)
- unseen-generator recall, per family and pooled
- what bACC would be if the AI side were entirely unseen generators, paired
  with (a) all real families, (b) only real families outside CF's training
  real sets (COCO / CelebA / FFHQ / LAION / ImageNet).
"""
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as E  # noqa: E402

Image.MAX_IMAGE_PIXELS = 300_000_000

DATA = os.path.expanduser("~/aidetect-data")
UNSEEN = os.path.join(DATA, "corpus-unseen")
PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/pipeline.json")
CF_MODEL = os.path.join(E.MODELS, "cf-vit.onnx")
SG_MODEL = os.path.join(E.MODELS, "siglip-detector-int8.onnx")
CF_OUT = os.path.join(DATA, "logits_cf_unseen.json")
SG_OUT = os.path.join(DATA, "logits_siglip_unseen.json")
THR = 0.65
CF_TAGS = ["off440", "s768", "nc"]
SG_TAGS = ["sq", "ar", "nc"]

# Real families in the main corpus whose source datasets are NOT among CF's
# training real sets (COCO, CelebA, FFHQ, LAION, ImageNet). Judgement call,
# recorded so it can be challenged: bm-real ("web-photo") is bitmind's own
# scrape, not LAION, but web photos are LAION-like, so it is listed on the
# unseen side with that caveat rather than silently either way.
UNSEEN_REAL_FAMILIES = {
    "openimages-256", "openimages-full", "flickr", "faces-lowres", "web-photo",
    "textures", "objects", "dashcam", "mugshots", "sky-timelapse", "paintings",
}

SG_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SG_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def sg_tensor(img):
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - SG_MEAN) / SG_STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def sg_views(im):
    """Mirror siglip_eval.py exactly (sq + ar always, nc only when big enough)."""
    out = {"sq": im.resize((224, 224), Image.BILINEAR)}
    w, h = im.size
    if min(w, h) >= 224:
        l, t = (w - 224) // 2, (h - 224) // 2
        out["nc"] = im.crop((l, t, l + 224, t + 224))
    s = 224 / min(w, h)
    nw, nh = max(224, round(w * s)), max(224, round(h * s))
    r = im.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - 224) // 2, (nh - 224) // 2
    out["ar"] = r.crop((l, t, l + 224, t + 224))
    return out


def cf_views(im):
    """Mirror cache_logits.py for the three tags the shipped recipe uses."""
    CROP = 384
    out = {}
    w, h = im.size
    for tag, S in (("off440", 440), ("s768", 768)):
        s = S / min(w, h)
        if S > 440 and s > 2.0:
            continue  # never upscale beyond 2x for the big scale
        nw, nh = max(CROP, round(w * s)), max(CROP, round(h * s))
        r = im.resize((nw, nh), Image.BICUBIC)
        l, t = (nw - CROP) // 2, (nh - CROP) // 2
        out[tag] = r.crop((l, t, l + CROP, t + CROP))
    im2 = im
    if min(w, h) < CROP:
        s = CROP / min(w, h)
        im2 = im.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
    W, H = im2.size
    l, t = (W - CROP) // 2, (H - CROP) // 2
    out["nc"] = im2.crop((l, t, l + CROP, t + CROP))
    return out


def list_unseen():
    labels = json.load(open(os.path.join(UNSEEN, "_labels.json")))
    items = []
    for name, meta in labels.items():
        d = os.path.join(UNSEEN, name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"):
                items.append({"path": os.path.join(d, f), "source": name,
                              "label": meta["label"], "family": meta["family"]})
    return items


def run_model(items, model_path, view_fn, tensor_fn, out_path, output_kind):
    cache = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = [it for it in items if it["path"] not in cache]
    if not todo:
        return cache
    sess = E.make_session(model_path)
    iname = sess.get_inputs()[0].name
    t0 = time.time()
    for i, it in enumerate(todo):
        try:
            im = Image.open(it["path"]).convert("RGB")
            vs = view_fn(im)
            tags = list(vs.keys())
            batch = np.concatenate([tensor_fn(vs[t]) for t in tags], axis=0)
            out = sess.run(None, {iname: batch})[0]
            if output_kind == "single":
                lg = {t: float(v) for t, v in zip(tags, out.reshape(-1))}
            else:  # two-logit: log-odds of AI is logit0 - logit1
                lg = {t: float(out[j, 0] - out[j, 1]) for j, t in enumerate(tags)}
            cache[it["path"]] = {"source": it["source"], "label": it["label"],
                                 "family": it["family"], "size": list(im.size), "logits": lg}
        except Exception as e:
            cache[it["path"]] = {"source": it["source"], "label": it["label"],
                                 "family": it["family"], "err": str(e)[:120]}
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
            json.dump(cache, open(out_path, "w"))
    json.dump(cache, open(out_path, "w"))
    return cache


def fused_rows(cf_cache, sg_cache):
    rows = []
    for k in sorted(set(cf_cache) & set(sg_cache)):
        c, s = cf_cache[k], sg_cache[k]
        if "logits" not in c or "logits" not in s:
            continue
        cl = [c["logits"][t] for t in CF_TAGS if t in c["logits"]]
        sl = [s["logits"][t] for t in SG_TAGS if t in s["logits"]]
        if not cl or not sl:
            continue
        w, h = c.get("size") or [384, 384]
        rows.append({"path": k, "cf": float(np.mean(cl)), "sg": float(np.mean(sl)),
                     "y": c["label"], "fam": c["family"], "src": c["source"],
                     "minEdge": float(min(w, h))})
    return rows


def score(rows, fusion):
    w = fusion["weights"]
    a, t = fusion["gate"]["a"], fusion["gate"]["t"]
    ref = fusion["referenceEdge"]
    out = []
    for r in rows:
        o = math.log2(max(32, min(8192, r["minEdge"])) / ref)
        gate = 1 / (1 + math.exp(-a * (o - t)))
        s = w["cfMax"] * gate * r["cf"] + w["sg"] * r["sg"] + fusion["intercept"] + fusion["thresholdOffset"]
        out.append({**r, "p": 1 / (1 + math.exp(-s))})
    return out


def bacc(scored):
    ai = [r for r in scored if r["y"] == 1]
    real = [r for r in scored if r["y"] == 0]
    rc = float(np.mean([r["p"] >= THR for r in ai])) if ai else float("nan")
    sp = float(np.mean([r["p"] < THR for r in real])) if real else float("nan")
    return (rc + sp) / 2, rc, sp


def main():
    fusion = json.load(open(PIPELINE))["fusion"]

    # ---- validation: reproduce the recorded in-sample number ----
    cf_main = json.load(open(os.path.join(DATA, "logits_cf.json")))
    sg_main = json.load(open(os.path.join(DATA, "logits_siglip_int8.json")))
    main_rows = fused_rows(cf_main, sg_main)
    scored_main = score(main_rows, fusion)
    ba, rc, sp = bacc(scored_main)
    recorded = json.load(open(os.path.join(DATA, "calibration_v2.json")))["_inSampleBalancedAccuracy"]
    print(f"main corpus n={len(scored_main)}: bACC={ba*100:.2f}% recall={rc*100:.1f}% spec={sp*100:.1f}%")
    print(f"recorded in calibration_v2.json: {recorded*100:.2f}%")
    if abs(ba - recorded) > 5e-4:
        print("MISMATCH -- scoring code does not reproduce the calibration fit; stopping.")
        sys.exit(1)
    print("reproduction OK\n")

    # ---- unseen corpus ----
    items = list_unseen()
    print(f"unseen corpus: {len(items)} images")
    cf_cache = run_model(items, CF_MODEL, cf_views, E._to_tensor, CF_OUT, "single")
    sg_cache = run_model(items, SG_MODEL, sg_views, sg_tensor, SG_OUT, "two")
    rows = fused_rows(cf_cache, sg_cache)
    scored = score(rows, fusion)

    print(f"\n{'family':20s} {'n':>4s} {'recall':>7s} {'medP':>6s} {'medEdge':>8s}")
    fams = sorted(set(r["fam"] for r in scored))
    for f in fams:
        rs = [r for r in scored if r["fam"] == f]
        rec = float(np.mean([r["p"] >= THR for r in rs]))
        med = float(np.median([r["p"] for r in rs]))
        edge = float(np.median([r["minEdge"] for r in rs]))
        print(f"{f:20s} {len(rs):4d} {rec*100:6.1f}% {med:6.3f} {edge:8.0f}")

    pooled = float(np.mean([r["p"] >= THR for r in scored]))
    print(f"\npooled unseen-generator recall: {pooled*100:.2f}%  (n={len(scored)})")

    real_all = [r for r in scored_main if r["y"] == 0]
    sp_all = float(np.mean([r["p"] < THR for r in real_all]))
    real_un = [r for r in real_all if r["fam"] in UNSEEN_REAL_FAMILIES]
    sp_un = float(np.mean([r["p"] < THR for r in real_un]))

    print(f"real specificity, all families:        {sp_all*100:.2f}%  (n={len(real_all)})")
    print(f"real specificity, non-CF-training set: {sp_un*100:.2f}%  (n={len(real_un)})")
    print(f"\nbACC if AI side were all-unseen, vs all real:        {(pooled+sp_all)/2*100:.2f}%")
    print(f"bACC if AI side were all-unseen, vs non-CF-train real: {(pooled+sp_un)/2*100:.2f}%")

    json.dump({"scored": [{k: r[k] for k in ("path", "p", "y", "fam", "src", "minEdge")} for r in scored]},
              open(os.path.join(DATA, "scored_unseen.json"), "w"))


if __name__ == "__main__":
    main()
