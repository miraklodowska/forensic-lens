"""Fit the shipping calibration and emit the constants that go into pipeline.json.

Feature set is deliberately 'gated': the two model logits, plus each one's
interaction with how much native resolution was available. Image size is NOT a
standalone feature -- given one, the fit happily learns "big images are real",
which is true of this corpus (its Flux samples are 256px) and not of the world.
Letting size only modulate trust in the size-sensitive model keeps the prior out.
"""
import os
import json, math
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
        rows.append({"cf": float(np.mean(c)), "sg": float(np.mean(s)), "y": cf[k]["label"],
                     "fam": cf[k]["family"], "src": cf[k]["source"], "minEdge": float(min(w, h))})
    return rows


def feats(rows):
    cf = np.array([r["cf"] for r in rows])
    sg = np.array([r["sg"] for r in rows])
    scale = np.log2(np.clip(np.array([r["minEdge"] for r in rows]), 32, 8192) / 384.0)
    big = np.clip(scale, 0, None)
    small = np.clip(-scale, 0, None)
    return np.column_stack([cf, sg, cf * big, cf * small])


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
    tpr = float(pred[y == 1].mean()); tnr = float((~pred[y == 0]).mean())
    return (tpr + tnr) / 2, tpr, tnr


rows = build()
X, y = feats(rows), np.array([r["y"] for r in rows])
fam = np.array([r["fam"] for r in rows])
w, b = fit_balanced(X, y)
s = X @ w + b + LT
ba, tpr, tnr = bacc(s, y)

print(f"n={len(rows)}  real={(y==0).sum()}  ai={(y==1).sum()}  families={len(set(fam))}")
print(f"IN-SAMPLE  bACC={ba*100:.2f}%  recall={tpr*100:.1f}%  spec={tnr*100:.1f}%")

# honest generalization: hold out each family (real and AI alike)
outs = []
for held in sorted(set(fam)):
    te = fam == held; tr = ~te
    if len(set(y[tr])) < 2:
        continue
    wi, bi = fit_balanced(X[tr], y[tr])
    si = X @ wi + bi + LT
    lab = int(y[te][0])
    rc = float((si[y == 1] >= LT).mean()) if lab == 0 else float((si[te] >= LT).mean())
    sp = float((si[te] < LT).mean()) if lab == 0 else float((si[y == 0] < LT).mean())
    outs.append((held, lab, (rc + sp) / 2, rc, sp, int(te.sum())))
print(f"LOCO mean bACC = {np.mean([o[2] for o in outs])*100:.2f}%   "
      f"min = {min(o[2] for o in outs)*100:.2f}%")

print("\nleave-one-family-out:")
for f_, lab, b_, rc, sp, n in sorted(outs, key=lambda r: r[2]):
    print(f"  {'AI  ' if lab else 'REAL'} {f_:24s} n={n:4d} bACC={b_*100:6.2f}% recall={rc*100:5.1f}% spec={sp*100:5.1f}%")

print("\nper-source (full fit):")
src = np.array([r["src"] for r in rows])
for so in sorted(set(src)):
    m = src == so
    lab = int(y[m][0])
    ok = float(((s[m] >= LT) == (lab == 1)).mean())
    print(f"  {so:22s} lab={lab} n={int(m.sum()):4d} acc={ok*100:6.2f}%")

const = {
    "cfTags": CF_TAGS, "sgTags": SG_TAGS,
    "weights": {"cf": float(w[0]), "sg": float(w[1]), "cfBig": float(w[2]), "cfSmall": float(w[3])},
    "intercept": float(b), "thresholdOffset": LT,
    "inSampleBalancedAccuracy": float(ba), "locoMeanBalancedAccuracy": float(np.mean([o[2] for o in outs])),
    "n": len(rows),
}
print("\n" + json.dumps(const, indent=2))
json.dump(const, open(os.path.expanduser("~/aidetect-data/calibration.json"), "w"), indent=2)
