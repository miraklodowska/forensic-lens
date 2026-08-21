"""Does padding beat upscaling for small images?

CF-ViT needs a 384x384 input. For a 256px image the default answer is to
upscale, which interpolates new pixels into existing ones and smears exactly the
high-frequency statistics the detector reads. Padding keeps every source pixel
at its original scale and fills the margin instead -- the model sees real
evidence plus a border, rather than plausible-looking invented detail.
"""
import numpy as np, onnxruntime as ort, os, json
from PIL import Image
import evalkit as E

CROP = 384
SRC = os.path.expanduser("~/aidetect-data/corpus")
SETS = ["ai_flux_h3", "ai_flux_h4", "ai_elsa_if", "real_coco256", "real_openimg", "real_lfw",
        "ai_realvis", "ai_sdxl_h4", "ai_aura"]


def upscaled(im):
    w, h = im.size
    s = CROP / min(w, h)
    r = im.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
    l, t = (r.size[0] - CROP) // 2, (r.size[1] - CROP) // 2
    return r.crop((l, t, l + CROP, t + CROP))


def padded(im, mode="reflect"):
    w, h = im.size
    a = np.asarray(im)
    if w >= CROP and h >= CROP:
        l, t = (w - CROP) // 2, (h - CROP) // 2
        return Image.fromarray(a[t:t + CROP, l:l + CROP])
    ph, pw = max(0, CROP - h), max(0, CROP - w)
    top, left = ph // 2, pw // 2
    a = np.pad(a, ((top, ph - top), (left, pw - left), (0, 0)), mode=mode)
    return Image.fromarray(a[:CROP, :CROP])


def tiled(im):
    """Repeat the image to fill 384 -- keeps native scale, no invented pixels."""
    w, h = im.size
    a = np.asarray(im)
    ry, rx = int(np.ceil(CROP / h)), int(np.ceil(CROP / w))
    t = np.tile(a, (ry, rx, 1))[:CROP, :CROP]
    return Image.fromarray(t)


sess = E.make_session(os.path.join(E.MODELS, "model.onnx"))
iname = sess.get_inputs()[0].name
labels = json.load(open(os.path.join(SRC, "_labels.json")))

print(f"{'source':16s} {'lab':>3s} {'n':>4s} {'upscale':>9s} {'reflect':>9s} {'edge':>9s} {'tile':>9s}   (median logit)")
res = {}
for s in SETS:
    d = os.path.join(SRC, s)
    if not os.path.isdir(d):
        continue
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))[:60]
    acc = {k: [] for k in ("up", "refl", "edge", "tile")}
    for f in files:
        im = Image.open(os.path.join(d, f)).convert("RGB")
        views = {"up": upscaled(im), "refl": padded(im, "reflect"),
                 "edge": padded(im, "edge"), "tile": tiled(im)}
        batch = np.concatenate([E._to_tensor(v) for v in views.values()], 0)
        out = sess.run(None, {iname: batch})[0].reshape(-1)
        for k, v in zip(views.keys(), out):
            acc[k].append(float(v))
    lab = labels[s]["label"]
    res[s] = {k: float(np.median(v)) for k, v in acc.items()}
    print(f"{s:16s} {lab:3d} {len(files):4d} "
          f"{res[s]['up']:9.3f} {res[s]['refl']:9.3f} {res[s]['edge']:9.3f} {res[s]['tile']:9.3f}")

# separation between AI and real medians per strategy
ai = [s for s in res if labels[s]["label"] == 1]
re_ = [s for s in res if labels[s]["label"] == 0]
print("\nmedian AI - median REAL (higher = better separation):")
for k in ("up", "refl", "edge", "tile"):
    a = np.median([res[s][k] for s in ai])
    r = np.median([res[s][k] for s in re_])
    print(f"  {k:6s} AI={a:7.3f}  REAL={r:7.3f}  gap={a-r:7.3f}")
