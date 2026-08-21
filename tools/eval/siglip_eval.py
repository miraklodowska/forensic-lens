"""Evaluate the SigLIP detector, especially on the small images CF-ViT misses."""
import json, os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image
import evalkit as E

Image.MAX_IMAGE_PIXELS = 300_000_000
MODEL = os.environ.get("SIGLIP_MODEL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/weights/siglip-detector-int8.onnx"))
SIZE = 224
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
OUT = os.environ.get("SIGLIP_OUT", os.path.expanduser("~/aidetect-data/logits_siglip.json"))


def tensor(img):
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def views(im):
    """squash-to-224 (the model's own recipe) plus a native centre crop."""
    out = {}
    out["sq"] = im.resize((SIZE, SIZE), Image.BILINEAR)
    w, h = im.size
    # native-resolution centre window, only when the image is big enough
    if min(w, h) >= SIZE:
        l, t = (w - SIZE) // 2, (h - SIZE) // 2
        out["nc"] = im.crop((l, t, l + SIZE, t + SIZE))
    # aspect-preserving: shortest edge -> 224 then centre crop
    s = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    r = im.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - SIZE) // 2, (nh - SIZE) // 2
    out["ar"] = r.crop((l, t, l + SIZE, t + SIZE))
    return out


def main():
    items = E.list_corpus()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        keep = set(only.split(","))
        items = [i for i in items if i["source"] in keep]
    print("items:", len(items), flush=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    cache = {}
    if os.path.exists(OUT) and not only:
        cache = json.load(open(OUT))
        print("resuming:", len(cache), flush=True)

    t0 = time.time()
    n = 0
    for it in items:
        k = it["path"]
        if k in cache:
            continue
        try:
            im = Image.open(k).convert("RGB")
            vs = views(im)
            tags = list(vs.keys())
            batch = np.concatenate([tensor(vs[t]) for t in tags], axis=0)
            out = sess.run(None, {iname: batch})[0]
            # id2label = {0: 'ai', 1: 'hum'} -> log-odds of AI is logit0 - logit1
            lg = {t: float(out[i, 0] - out[i, 1]) for i, t in enumerate(tags)}
            cache[k] = {"source": it["source"], "label": it["label"],
                        "family": it["family"], "size": list(im.size), "logits": lg}
        except Exception as e:
            cache[k] = {"source": it["source"], "label": it["label"],
                        "family": it["family"], "err": str(e)[:100]}
        n += 1
        if n % 100 == 0:
            print(f"  {n} {time.time()-t0:.0f}s", flush=True)
            if not only:
                json.dump(cache, open(OUT, "w"))
    if not only:
        json.dump(cache, open(OUT, "w"))

    # quick per-source read-out
    from collections import defaultdict
    by = defaultdict(list)
    for k, v in cache.items():
        if "logits" in v:
            by[v["source"]].append(1 / (1 + np.exp(-v["logits"]["sq"])))
    print(f"\n{'source':22s} {'n':>4s} {'medP(ai)':>9s}")
    for s in sorted(by):
        a = np.array(by[s])
        print(f"{s:22s} {a.size:4d} {np.median(a):9.3f}")
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
