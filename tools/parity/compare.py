"""Diff browser-produced scores against the Python reference.

Pillow and canvas are different resamplers, and these detectors read exactly the
frequencies a resampler touches, so this checks that the accuracy measured
offline actually survives the trip into the browser.

Usage:
    python3 tools/parity/compare.py browser-results.json
"""
import json
import math
import os
import sys

import numpy as np

CF = os.environ.get("CF_LOGITS", os.path.expanduser("~/aidetect-data/logits_cf.json"))
SG = os.environ.get("SG_LOGITS", os.path.expanduser("~/aidetect-data/logits_siglip_int8.json"))
PIPELINE = os.environ.get(
    "PIPELINE", os.path.join(os.path.dirname(__file__), "..", "..", "models", "pipeline.json")
)
CF_TAGS = ["off440", "s768", "nc"]
SG_TAGS = ["sq", "ar", "nc"]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def python_reference():
    cf = json.load(open(CF))
    sg = json.load(open(SG))
    spec = json.load(open(PIPELINE))["fusion"]
    w = spec["weights"]
    out = {}
    for path, c in cf.items():
        if "logits" not in c:
            continue
        s = sg.get(path)
        if not s or "logits" not in s:
            continue
        # Browser names files "<source>__<basename>"; rebuild that key.
        name = f"{os.path.basename(os.path.dirname(path))}__{os.path.basename(path)}"
        cfl = float(np.mean([c["logits"][t] for t in CF_TAGS if t in c["logits"]]))
        sgl = float(np.mean([s["logits"][t] for t in SG_TAGS if t in s["logits"]]))
        wpx, hpx = c.get("size") or [384, 384]
        octaves = math.log2(min(8192, max(32, min(wpx, hpx))) / spec["referenceEdge"])
        big, small = max(0.0, octaves), max(0.0, -octaves)
        z = (
            w["cf"] * cfl
            + w["sg"] * sgl
            + w["cfBig"] * cfl * big
            + w["cfSmall"] * cfl * small
            + spec["intercept"]
            + spec["thresholdOffset"]
        )
        out[name] = {"cf": cfl, "sg": sgl, "probability": sigmoid(z)}
    return out


def main():
    browser = json.load(open(sys.argv[1]))
    rows = browser["rows"] if isinstance(browser, dict) else browser
    ref = python_reference()

    pairs = []
    for r in rows:
        if "error" in r or r.get("probability") is None:
            continue
        m = ref.get(r["name"])
        if m is None:
            continue
        pairs.append((r["name"], r["cf"], m["cf"], r["sg"], m["sg"], r["probability"], m["probability"]))

    if not pairs:
        print("no overlapping images — check that the corpus paths match")
        return

    dcf = np.array([abs(p[1] - p[2]) for p in pairs])
    dsg = np.array([abs(p[3] - p[4]) for p in pairs])
    dp = np.array([abs(p[5] - p[6]) for p in pairs])
    agree = np.array([(p[5] >= 0.65) == (p[6] >= 0.65) for p in pairs])

    print(f"matched {len(pairs)} images\n")
    print(f"{'':14s} {'mean':>9s} {'p95':>9s} {'max':>9s}")
    for name, d in (("|Δ cf logit|", dcf), ("|Δ sg logit|", dsg), ("|Δ probability|", dp)):
        print(f"{name:14s} {d.mean():9.4f} {np.percentile(d, 95):9.4f} {d.max():9.4f}")
    print(f"\nverdict agreement at 0.65: {agree.mean() * 100:.1f}%  ({int(agree.sum())}/{len(agree)})")

    worst = sorted(pairs, key=lambda p: -abs(p[5] - p[6]))[:5]
    print("\nlargest probability gaps:")
    for n, bcf, pcf, bsg, psg, bp, pp in worst:
        print(f"  {n[:44]:44s} browser={bp:.3f} python={pp:.3f}  Δ={abs(bp - pp):.3f}")


if __name__ == "__main__":
    main()
