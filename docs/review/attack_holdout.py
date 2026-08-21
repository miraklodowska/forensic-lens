"""Adversarial re-evaluation of the Forensic Lens calibration.

Mirrors tools/eval/finalize.py exactly (features, fitter, threshold), then:
  A. reproduces the shipped LOFO protocol (sanity check vs claimed 78.75%)
  B. quantifies the in-sample half of LOFO: for held-out AI families the
     specificity half is computed on TRAINING real images (and vice versa)
  C. fully held-out repeated 2-fold at family level: both classes unseen
  D. same, but grouping families that share an underlying dataset/prompt set
"""
import os
import json, math, sys
import numpy as np

CF = os.path.expanduser("~/aidetect-data/logits_cf.json")
SG = os.path.expanduser("~/aidetect-data/logits_siglip_int8.json")  # shipped INT8
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
    tpr = float(pred[y == 1].mean())
    tnr = float((~pred[y == 0]).mean())
    return (tpr + tnr) / 2, tpr, tnr


rows = build()
X = feats(rows)
y = np.array([r["y"] for r in rows])
fam = np.array([r["fam"] for r in rows])
fams = sorted(set(fam))
real_fams = sorted({f for f, l in zip(fam, y) if l == 0})
ai_fams = sorted({f for f, l in zip(fam, y) if l == 1})
print(f"n={len(rows)} real_fams={len(real_fams)} ai_fams={len(ai_fams)}")

# ---- A: reproduce shipped LOFO exactly ----------------------------------
outs = []
for held in fams:
    te = fam == held; tr = ~te
    if len(set(y[tr])) < 2:
        continue
    wi, bi = fit_balanced(X[tr], y[tr])
    si = X @ wi + bi + LT
    lab = int(y[te][0])
    rc = float((si[y == 1] >= LT).mean()) if lab == 0 else float((si[te] >= LT).mean())
    sp = float((si[te] < LT).mean()) if lab == 0 else float((si[y == 0] < LT).mean())
    outs.append((held, lab, (rc + sp) / 2))
print(f"\n[A] shipped LOFO protocol reproduced: mean bACC = {np.mean([o[2] for o in outs])*100:.2f}%"
      f"  (claimed 78.75%)")

# ---- B: same LOFO but score the other class OUT-of-sample too -----------
# For a held-out AI family, the spec half must come from real images the
# refit never saw. Do it by nesting: for each (AI fam a, REAL fam r) pair,
# fit without both, recall on a, spec on r. Average over partners.
print("\n[B] pairwise fully-held-out LOFO (both classes unseen):")
pair_bacc = {}
cache = {}
for a in ai_fams:
    for r in real_fams:
        te_a = fam == a; te_r = fam == r
        tr = ~(te_a | te_r)
        key = (a, r)
        wi, bi = fit_balanced(X[tr], y[tr])
        si = X @ wi + bi + LT
        rc = float((si[te_a] >= LT).mean())
        sp = float((si[te_r] < LT).mean())
        pair_bacc[key] = (rc + sp) / 2
pb = np.array(list(pair_bacc.values()))
print(f"    mean over all {len(pb)} (AI,REAL) pairs: {pb.mean()*100:.2f}%  "
      f"median {np.median(pb)*100:.2f}%  min {pb.min()*100:.2f}%")
per_ai = {a: np.mean([pair_bacc[(a, r)] for r in real_fams]) for a in ai_fams}
per_real = {r: np.mean([pair_bacc[(a, r)] for a in ai_fams]) for r in real_fams}
print("    worst held-out AI fams:  ",
      {k: f"{v*100:.1f}" for k, v in sorted(per_ai.items(), key=lambda t: t[1])[:5]})
print("    worst held-out REAL fams:",
      {k: f"{v*100:.1f}" for k, v in sorted(per_real.items(), key=lambda t: t[1])[:5]})

# ---- C: repeated family-level 2-fold, both classes fully held out --------
def repeated_split(groups_real, groups_ai, group_of, n_rep=200, seed=20260813):
    rng = np.random.default_rng(seed)
    baccs = []
    for _ in range(n_rep):
        gr = list(groups_real); ga = list(groups_ai)
        rng.shuffle(gr); rng.shuffle(ga)
        f1 = set(gr[:len(gr)//2]) | set(ga[:len(ga)//2])
        f2 = set(gr[len(gr)//2:]) | set(ga[len(ga)//2:])
        for tr_g, te_g in ((f1, f2), (f2, f1)):
            tr = np.array([group_of[f] in tr_g for f in fam])
            te = np.array([group_of[f] in te_g for f in fam])
            wi, bi = fit_balanced(X[tr], y[tr])
            si = X @ wi + bi + LT
            b_, tp, tn = bacc(si[te], y[te])
            baccs.append(b_)
    return np.array(baccs)

ident = {f: f for f in fams}
bc = repeated_split(real_fams, ai_fams, ident)
print(f"\n[C] repeated family 2-fold, fully held out ({len(bc)} folds):")
print(f"    mean {bc.mean()*100:.2f}%  median {np.median(bc)*100:.2f}%  "
      f"5th pct {np.percentile(bc,5)*100:.2f}%  min {bc.min()*100:.2f}%  "
      f"frac<75%: {(bc<0.75).mean()*100:.0f}%")

# ---- D: same but with shared-source super-groups -------------------------
group = {}
for f in fams:
    if f.startswith("coco"): group[f] = "G:mscoco"
    elif f.startswith("openimages"): group[f] = "G:openimages"
    elif f in ("deepfloyd-if", "sd-1.4", "sd-2.1", "sdxl-elsa"): group[f] = "G:elsa-d3"
    elif f in ("midjourney-genimage", "wukong"): group[f] = "G:genimage"
    elif f in ("flux.1-dev", "realvisxl", "sdxl"): group[f] = "G:bm-whitepaper"
    else: group[f] = f
g_real = sorted({group[f] for f in real_fams})
g_ai = sorted({group[f] for f in ai_fams})
bd = repeated_split(g_real, g_ai, group)
print(f"\n[D] repeated 2-fold over shared-source groups "
      f"({len(g_real)} real, {len(g_ai)} ai groups, {len(bd)} folds):")
print(f"    mean {bd.mean()*100:.2f}%  median {np.median(bd)*100:.2f}%  "
      f"5th pct {np.percentile(bd,5)*100:.2f}%  min {bd.min()*100:.2f}%  "
      f"frac<75%: {(bd<0.75).mean()*100:.0f}%")
