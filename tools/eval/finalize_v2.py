"""Fit the corrected size-gated fusion and emit the constants for pipeline.json.

The gate is multiplicative: w_eff(o) = cfMax * sigmoid(a*(o - t)), bounded in
[0, cfMax]. Unlike the additive interaction it replaces, it cannot invert the
fingerprint detector's vote when that detector saturates on small images.

`a` and `t` enter nonlinearly, so they are grid-searched; for each candidate the
remaining coefficients are a plain class-balanced logistic fit, and any solution
with a negative cfMax is rejected outright rather than clipped. Selection is by
leave-one-family-out mean, not in-sample fit, because the failure being repaired
was itself an in-sample artifact.
"""
import json
import math
import os

import numpy as np

CF = os.environ.get("CF_LOGITS", os.path.expanduser("~/aidetect-data/logits_cf.json"))
SG = os.environ.get("SG_LOGITS", os.path.expanduser("~/aidetect-data/logits_siglip_int8.json"))
OUT = os.environ.get("OUT", os.path.expanduser("~/aidetect-data/calibration_v2.json"))
THR = 0.65
LT = math.log(THR / (1 - THR))
REF = 384.0
CF_TAGS = ["off440", "s768", "nc"]
SG_TAGS = ["sq", "ar", "nc"]


def load():
    cf = {k: v for k, v in json.load(open(CF)).items() if "logits" in v}
    sg = {k: v for k, v in json.load(open(SG)).items() if "logits" in v}
    rows = []
    for k in sorted(set(cf) & set(sg)):
        c = [cf[k]["logits"][t] for t in CF_TAGS if t in cf[k]["logits"]]
        s = [sg[k]["logits"][t] for t in SG_TAGS if t in sg[k]["logits"]]
        if not c or not s:
            continue
        w, h = cf[k].get("size") or [384, 384]
        rows.append({
            "cf": float(np.mean(c)), "sg": float(np.mean(s)),
            "y": cf[k]["label"], "fam": cf[k]["family"], "src": cf[k]["source"],
            "minEdge": float(min(w, h)),
        })
    return rows


def fit_balanced(X, y, iters=300, l2=1e-2):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n1, n0 = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))
    sw = np.where(y == 1, 0.5 / n1, 0.5 / n0)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = Z.T @ (sw * (p - y)) + l2 * w
        H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + (l2 + 1e-7) * np.eye(Z.shape[1])
        w -= np.linalg.solve(H, g)
        b -= float(np.sum(sw * (p - y))) / (float(np.sum(sw * p * (1 - p))) + 1e-9)
    w_orig = w / sd
    return w_orig, float(b - w_orig @ mu)


def octaves(minEdge):
    return np.log2(np.clip(minEdge, 32, 8192) / REF)


def features(rows, a, t):
    cf = np.array([r["cf"] for r in rows])
    sg = np.array([r["sg"] for r in rows])
    o = octaves(np.array([r["minEdge"] for r in rows]))
    gate = 1 / (1 + np.exp(-a * (o - t)))
    return np.column_stack([cf * gate, sg])


def bacc(s, y):
    pred = s >= LT
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else 0.0
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else 0.0
    return (tpr + tnr) / 2, tpr, tnr


def lofo(rows, y, fam, a, t):
    X = features(rows, a, t)
    out = []
    for held in sorted(set(fam)):
        te = fam == held
        tr = ~te
        if len(set(y[tr])) < 2 or te.sum() == 0:
            continue
        w, b = fit_balanced(X[tr], y[tr])
        if w[0] < 0:
            return None, []
        s = X @ w + b + LT
        lab = int(y[te][0])
        rc = float((s[y == 1] >= LT).mean()) if lab == 0 else float((s[te] >= LT).mean())
        sp = float((s[te] < LT).mean()) if lab == 0 else float((s[y == 0] < LT).mean())
        out.append((held, lab, (rc + sp) / 2, rc, sp, int(te.sum())))
    return float(np.mean([o[2] for o in out])), out


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    fam = np.array([r["fam"] for r in rows])
    print(f"n={len(rows)}  real={(y==0).sum()}  ai={(y==1).sum()}  families={len(set(fam))}")
    sub = sum(1 for r in rows if r["minEdge"] < 256)
    print(f"sub-256px: {sub}\n")

    best = None
    print(f"{'a':>5} {'t':>6} {'cfMax':>8} {'in-samp':>9} {'LOFO':>8}")
    for a in (0.75, 1.0, 1.5, 2.0, 3.0):
        for t in (-2.0, -1.5, -1.0, -0.5, 0.0):
            X = features(rows, a, t)
            w, b = fit_balanced(X, y)
            if w[0] < 0:
                continue  # would invert CF's vote; reject rather than clip
            ins = bacc(X @ w + b + LT, y)[0]
            m, _ = lofo(rows, y, fam, a, t)
            if m is None:
                continue
            print(f"{a:5.2f} {t:6.2f} {w[0]:8.4f} {ins*100:8.2f}% {m*100:7.2f}%")
            if best is None or m > best[0]:
                best = (m, a, t, w, b, ins)

    m, a, t, w, b, ins = best
    X = features(rows, a, t)
    s = X @ w + b + LT
    ba, tpr, tnr = bacc(s, y)
    print(f"\nBEST a={a} t={t}  in-sample={ba*100:.2f}% (recall {tpr*100:.1f}% spec {tnr*100:.1f}%)  LOFO={m*100:.2f}%")

    # The specific regression under repair: real images must not be flagged
    # simply for being small.
    print("\nreal-image false-positive rate by size bucket:")
    for lo, hi in ((0, 160), (160, 256), (256, 384), (384, 10 ** 9)):
        mask = np.array([(r["minEdge"] >= lo) and (r["minEdge"] < hi) for r in rows]) & (y == 0)
        if mask.sum():
            print(f"  {lo:>4}-{hi if hi < 10**9 else 'inf':>4}px  n={int(mask.sum()):4d}  FP={float((s[mask] >= LT).mean())*100:6.2f}%")
    print("\nAI recall by size bucket:")
    for lo, hi in ((0, 160), (160, 256), (256, 384), (384, 10 ** 9)):
        mask = np.array([(r["minEdge"] >= lo) and (r["minEdge"] < hi) for r in rows]) & (y == 1)
        if mask.sum():
            print(f"  {lo:>4}-{hi if hi < 10**9 else 'inf':>4}px  n={int(mask.sum()):4d}  recall={float((s[mask] >= LT).mean())*100:6.2f}%")

    print("\neffective CF weight (must never be negative):")
    for edge in (1024, 512, 384, 256, 160, 128, 96, 64):
        g = 1 / (1 + math.exp(-a * (math.log2(max(32, min(8192, edge)) / REF) - t)))
        print(f"  {edge:5d}px  w_eff={w[0]*g:8.5f}")

    const = {
        "kind": "size-gated-linear",
        "weights": {"cfMax": float(w[0]), "sg": float(w[1])},
        "gate": {"a": float(a), "t": float(t)},
        "intercept": float(b),
        "thresholdOffset": LT,
        "referenceEdge": int(REF),
        "_inSampleBalancedAccuracy": float(ba),
        "_lofoMeanBalancedAccuracy": float(m),
        "_n": len(rows),
    }
    json.dump(const, open(OUT, "w"), indent=2)
    print("\n" + json.dumps(const, indent=2))


if __name__ == "__main__":
    main()
