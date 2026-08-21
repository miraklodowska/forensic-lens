"""Size-conditioned fusion.

The two detectors do not fail randomly, they fail predictably by image size:
CF-ViT reads native-resolution pixel statistics and goes blind once an image is
too small to crop 384px without upscaling, while SigLIP squashes everything to
224 and is unbothered by size but fires on real faces and paintings. Image size
is known for free at inference time, so we let the fusion use it -- the model
learns "trust the fingerprint reader when there are pixels to read".
"""
import os
import json, math, sys
import numpy as np

CF = os.path.expanduser("~/aidetect-data/logits_cf.json")
SG = os.path.expanduser("~/aidetect-data/logits_siglip.json")
THR = 0.65
LT = math.log(THR / (1 - THR))
CF_TAGS = ["off440", "s768", "nc"]
SG_TAGS = ["sq", "ar", "nc"]


def load(p):
    return {k: v for k, v in json.load(open(p)).items() if "logits" in v}


def build():
    cf, sg = load(CF), load(SG)
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


def feats(rows, kind):
    cf = np.array([r["cf"] for r in rows])
    sg = np.array([r["sg"] for r in rows])
    # how much native-resolution detail the CF crop actually had, in octaves
    scale = np.log2(np.clip(np.array([r["minEdge"] for r in rows]), 32, 8192) / 384.0)
    big = np.clip(scale, 0, None)          # 0 for small images, grows with size
    small = np.clip(-scale, 0, None)       # 0 for big images, grows as it shrinks
    if kind == "base":
        return np.column_stack([cf, sg])
    if kind == "size":
        return np.column_stack([cf, sg, scale])
    if kind == "gated":
        return np.column_stack([cf, sg, cf * big, cf * small])
    if kind == "gated+size":
        return np.column_stack([cf, sg, scale, cf * big, cf * small])
    if kind == "full":
        return np.column_stack([cf, sg, scale, cf * big, cf * small, sg * small])
    raise ValueError(kind)


def fit_balanced(X, y, iters=300, l2=1e-2):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n1, n0 = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))
    sw = np.where(y == 1, 0.5 / n1, 0.5 / n0)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = Z.T @ (sw * (p - y)) + l2 * w
        H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + (l2 + 1e-7) * np.eye(Z.shape[1])
        w -= np.linalg.solve(H, g)
        b -= float(np.sum(sw * (p - y))) / (float(np.sum(sw * p * (1 - p))) + 1e-9)
    return w / sd, b - float((w / sd) @ mu)


def bacc(s, y):
    pred = s >= LT
    return ((pred[y == 1].mean() if (y == 1).any() else 0) +
            ((~pred[y == 0]).mean() if (y == 0).any() else 0)) / 2


def loco(rows, kind):
    """Leave-one-family-out over BOTH generators and real sources.

    Holding out real sources too matters: specificity measured only on real
    images that were in the fit is an optimistic number, and false positives on
    unfamiliar real content (paintings, studio portraits) are the failure mode
    users actually notice.
    """
    X = feats(rows, kind)
    y = np.array([r["y"] for r in rows])
    fam = np.array([r["fam"] for r in rows])
    out = []
    for held in sorted(set(fam)):
        te = fam == held
        tr = ~te
        if len(set(y[tr])) < 2 or te.sum() == 0:
            continue
        w, b = fit_balanced(X[tr], y[tr])
        s = X @ w + b + LT
        lab = int(y[te][0])
        if lab == 1:
            recall = float((s[te] >= LT).mean())
            spec = float((s[(y == 0)] < LT).mean())
        else:
            recall = float((s[(y == 1)] >= LT).mean())
            spec = float((s[te] < LT).mean())
        out.append((held, lab, (recall + spec) / 2, recall, spec, int(te.sum())))
    return out


if __name__ == "__main__":
    rows = build()
    y = np.array([r["y"] for r in rows])
    print(f"n={len(rows)} real={(y==0).sum()} ai={(y==1).sum()}\n")

    print(f"{'features':12s} {'in-sample':>10s} {'LOCO mean':>10s} {'LOCO worst':>28s}")
    best = None
    for kind in ["base", "size", "gated", "gated+size", "full"]:
        X = feats(rows, kind)
        w, b = fit_balanced(X, y)
        ins = bacc(X @ w + b + LT, y)
        rs = loco(rows, kind)
        mean = float(np.mean([r[2] for r in rs]))
        worst = min(rs, key=lambda r: r[2])
        print(f"{kind:12s} {ins*100:9.2f}% {mean*100:9.2f}%   {worst[0]:>18s} {worst[2]*100:6.2f}%")
        if best is None or mean > best[0]:
            best = (mean, kind)

    kind = best[1]
    print(f"\n===== detail for '{kind}' =====")
    X = feats(rows, kind)
    w, b = fit_balanced(X, y)
    s = X @ w + b + LT
    print(f"in-sample bACC={bacc(s,y)*100:.2f}%  weights={np.round(w,4).tolist()} b={b:.4f}")
    print("\nleave-one-family-out:")
    for f_, lab, ba, rc, sp, n in sorted(loco(rows, kind), key=lambda r: r[2]):
        tag = "AI  " if lab else "REAL"
        print(f"  {tag} {f_:24s} n={n:4d}  bACC={ba*100:6.2f}%  recall={rc*100:5.1f}%  spec={sp*100:5.1f}%")
