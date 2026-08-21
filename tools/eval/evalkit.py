"""Evaluation harness: CF-ViT ONNX over the local corpus, several preprocessing strategies."""
import io, json, os, sys, math, time
import numpy as np
import onnxruntime as ort
from PIL import Image

CORPUS = os.path.expanduser("~/aidetect-data/corpus")
MODELS = os.environ.get("MODELS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/weights"))
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CROP = 384
RESIZE = 440

Image.MAX_IMAGE_PIXELS = 300_000_000


def _to_tensor(img):
    """PIL RGB 384x384 -> NCHW float32 normalized."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def crops_official(im):
    """shortest edge -> 440 (bicubic), center crop 384."""
    w, h = im.size
    s = RESIZE / min(w, h)
    nw, nh = max(CROP, round(w * s)), max(CROP, round(h * s))
    r = im.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - CROP) // 2, (nh - CROP) // 2
    return [r.crop((l, t, l + CROP, t + CROP))]


def crops_native_center(im):
    """center crop 384 at native resolution (upscale only if too small)."""
    w, h = im.size
    if min(w, h) < CROP:
        s = CROP / min(w, h)
        im = im.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
        w, h = im.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    return [im.crop((l, t, l + CROP, t + CROP))]


def crops_native_multi(im, n=5):
    """center + 4 quadrant crops at native resolution."""
    w, h = im.size
    if min(w, h) < CROP:
        s = CROP / min(w, h)
        im = im.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
        w, h = im.size
    out = []
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    pts = [(cx, cy), (0, 0), (w - CROP, 0), (0, h - CROP), (w - CROP, h - CROP)]
    for (l, t) in pts[:n]:
        out.append(im.crop((l, t, l + CROP, t + CROP)))
    return out


def crops_squash(im):
    return [im.resize((CROP, CROP), Image.BICUBIC)]


def crops_hybrid(im, n=5):
    """official view + native-resolution views (scale diversity)."""
    return crops_official(im) + crops_native_multi(im, n - 1)


STRATEGIES = {
    "official": crops_official,
    "native_center": crops_native_center,
    "native_multi5": lambda im: crops_native_multi(im, 5),
    "squash": crops_squash,
    "hybrid5": lambda im: crops_hybrid(im, 5),
}


def make_session(path, provider="cpu"):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 8
    provs = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=provs)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def list_corpus():
    labels = json.load(open(os.path.join(CORPUS, "_labels.json")))
    items = []
    for name, meta in labels.items():
        d = os.path.join(CORPUS, name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"):
                items.append({"path": os.path.join(d, f), "source": name,
                              "label": meta["label"], "family": meta["family"]})
    return items


def web_degrade(im, max_edge=1024, quality=82):
    """Simulate a web-served image: downscale + JPEG recompress."""
    w, h = im.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def run(model_path, strategy, items, degrade=False, agg="mean", log_every=200):
    sess = make_session(model_path)
    iname = sess.get_inputs()[0].name
    fn = STRATEGIES[strategy]
    out = []
    t0 = time.time()
    for i, it in enumerate(items):
        try:
            im = Image.open(it["path"]).convert("RGB")
            if degrade:
                im = web_degrade(im)
            views = fn(im)
            batch = np.concatenate([_to_tensor(v) for v in views], axis=0)
            logits = sess.run(None, {iname: batch})[0].reshape(-1)
            probs = sigmoid(logits)
            if agg == "mean":
                p = float(np.mean(probs))
            elif agg == "max":
                p = float(np.max(probs))
            elif agg == "meanlogit":
                p = float(sigmoid(np.mean(logits)))
            elif agg == "median":
                p = float(np.median(probs))
            out.append({**it, "p": p, "logits": [float(x) for x in logits]})
        except Exception as e:
            out.append({**it, "p": None, "err": str(e)[:100]})
        if log_every and (i + 1) % log_every == 0:
            print(f"    {i+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    return out


def balanced_accuracy(res, thr=0.65):
    tp = sum(1 for r in res if r["p"] is not None and r["label"] == 1 and r["p"] >= thr)
    fn_ = sum(1 for r in res if r["p"] is not None and r["label"] == 1 and r["p"] < thr)
    tn = sum(1 for r in res if r["p"] is not None and r["label"] == 0 and r["p"] < thr)
    fp = sum(1 for r in res if r["p"] is not None and r["label"] == 0 and r["p"] >= thr)
    tpr = tp / (tp + fn_) if (tp + fn_) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {"balanced_acc": (tpr + tnr) / 2, "tpr_recall_ai": tpr, "tnr_specificity_real": tnr,
            "tp": tp, "fn": fn_, "tn": tn, "fp": fp}


def auc(res):
    pos = [r["p"] for r in res if r["p"] is not None and r["label"] == 1]
    neg = [r["p"] for r in res if r["p"] is not None and r["label"] == 0]
    if not pos or not neg:
        return float("nan")
    allv = [(p, 1) for p in pos] + [(p, 0) for p in neg]
    allv.sort(key=lambda x: x[0])
    ranks = {}
    i = 0
    r = 1
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (r + (r + (j - i))) / 2
        for k in range(i, j + 1):
            ranks[k] = avg
        r += (j - i + 1)
        i = j + 1
    sp = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (sp - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_source(res, thr=0.65):
    by = {}
    for r in res:
        by.setdefault(r["source"], []).append(r)
    rows = []
    for s, rs in sorted(by.items()):
        lab = rs[0]["label"]
        ok = sum(1 for r in rs if r["p"] is not None and ((r["p"] >= thr) == (lab == 1)))
        n = sum(1 for r in rs if r["p"] is not None)
        med = float(np.median([r["p"] for r in rs if r["p"] is not None])) if n else float("nan")
        rows.append((s, lab, n, ok / n if n else float("nan"), med, rs[0]["family"]))
    return rows


def report(res, title, thr=0.65):
    m = balanced_accuracy(res, thr)
    a = auc(res)
    print(f"\n=== {title} ===")
    print(f"  balanced_acc@{thr} = {m['balanced_acc']*100:.2f}%   "
          f"AI recall={m['tpr_recall_ai']*100:.1f}%  real specificity={m['tnr_specificity_real']*100:.1f}%  AUC={a:.4f}")
    print(f"  {'source':22s} {'lab':>3s} {'n':>4s} {'acc':>7s} {'medP':>7s}  family")
    for s, lab, n, acc, med, fam in per_source(res, thr):
        print(f"  {s:22s} {lab:3d} {n:4d} {acc*100:6.1f}% {med:7.3f}  {fam}")
    return m, a
