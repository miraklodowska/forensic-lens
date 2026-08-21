"""Run CF-ViT over the corpus once, caching per-crop logits so strategies can be composed offline."""
import json, os, sys, time
import numpy as np
from PIL import Image
import evalkit as E

Image.MAX_IMAGE_PIXELS = 300_000_000
CROP = 384
OUTJSON = os.path.expanduser("~/aidetect-data/logits_cf.json")
DEGRADED = "--degraded" in sys.argv
if DEGRADED:
    OUTJSON = os.path.expanduser("~/aidetect-data/logits_cf_degraded.json")


def crop_set(im):
    """Named crops spanning several scales; composed into strategies later."""
    out = {}
    w, h = im.size

    # scale views: shortest edge -> S, center crop 384
    for tag, S in (("off440", 440), ("s768", 768), ("s1200", 1200)):
        s = S / min(w, h)
        nw, nh = max(CROP, round(w * s)), max(CROP, round(h * s))
        # never upscale beyond 2x for the big scales (avoids inventing detail)
        if S > 440 and s > 2.0:
            continue
        r = im.resize((nw, nh), Image.BICUBIC)
        l, t = (nw - CROP) // 2, (nh - CROP) // 2
        out[tag] = r.crop((l, t, l + CROP, t + CROP))

    # native-resolution views
    im2 = im
    if min(w, h) < CROP:
        s = CROP / min(w, h)
        im2 = im.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
    W, H = im2.size
    cx, cy = (W - CROP) // 2, (H - CROP) // 2
    named = {
        "nc": (cx, cy),
        "ntl": (0, 0),
        "ntr": (W - CROP, 0),
        "nbl": (0, H - CROP),
        "nbr": (W - CROP, H - CROP),
    }
    for tag, (l, t) in named.items():
        out[tag] = im2.crop((l, t, l + CROP, t + CROP))

    out["sq"] = im.resize((CROP, CROP), Image.BICUBIC)
    return out


def main():
    items = E.list_corpus()
    print("items:", len(items), "degraded:", DEGRADED, flush=True)
    sess = E.make_session(os.path.join(E.MODELS, "cf-vit.onnx"))
    iname = sess.get_inputs()[0].name

    cache = {}
    if os.path.exists(OUTJSON):
        cache = json.load(open(OUTJSON))
        print("resuming with", len(cache), "cached", flush=True)

    t0 = time.time()
    done = 0
    for i, it in enumerate(items):
        key = it["path"]
        if key in cache:
            continue
        try:
            im = Image.open(key).convert("RGB")
            if DEGRADED:
                im = E.web_degrade(im)
            cs = crop_set(im)
            tags = list(cs.keys())
            batch = np.concatenate([E._to_tensor(cs[t]) for t in tags], axis=0)
            logits = sess.run(None, {iname: batch})[0].reshape(-1)
            cache[key] = {
                "source": it["source"], "label": it["label"], "family": it["family"],
                "size": list(im.size),
                "logits": {t: float(v) for t, v in zip(tags, logits)},
            }
        except Exception as e:
            cache[key] = {"source": it["source"], "label": it["label"], "family": it["family"],
                          "err": str(e)[:120]}
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"  {done} new / {i+1} seen  {el:.0f}s", flush=True)
            json.dump(cache, open(OUTJSON, "w"))
    json.dump(cache, open(OUTJSON, "w"))
    print("wrote", OUTJSON, len(cache), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
