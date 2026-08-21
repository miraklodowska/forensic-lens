"""Add the sub-256px regime to the corpus, for both classes.

The size-gate bug survived review because the corpus could not see it: zero AI
images below 256px and only 116 real ones, so the fit never observed what
happens once Community Forensics saturates. Any refit against the old corpus
would reproduce the same blind spot.

Downscaling is the right way to fill the gap rather than a shortcut. It is
exactly what the web does to these images -- a thumbnail on a gallery page *is*
a downscaled generation -- so the label survives the transform and the resulting
distribution is closer to browsing than the originals were. JPEG recompression
is applied for the same reason.
"""
import io
import os
import random
import sys

from PIL import Image

CORPUS = os.environ.get("CORPUS", os.path.expanduser("~/aidetect-data/corpus"))
OUT = os.environ.get("OUT", CORPUS)
# Straddles the crop size (384) and the default minImageSize (128), so the fit
# sees both the healthy regime and the one where CF goes blind.
EDGES = [96, 128, 160, 224]
PER_SOURCE = int(os.environ.get("PER_SOURCE", "40"))
QUALITY = [72, 82, 92]
random.seed(20260813)

Image.MAX_IMAGE_PIXELS = 300_000_000


def degrade(im, edge, quality):
    """Downscale so the shortest edge is `edge`, then JPEG-recompress."""
    w, h = im.size
    s = edge / min(w, h)
    if s >= 1.0:
        return None  # already at or below target; nothing to simulate
    small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    labels_path = os.path.join(CORPUS, "_labels.json")
    import json

    labels = json.load(open(labels_path))
    made = {}

    for source, meta in sorted(labels.items()):
        src_dir = os.path.join(CORPUS, source)
        if not os.path.isdir(src_dir) or source.endswith("_small"):
            continue
        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
        if not files:
            continue
        random.shuffle(files)

        dest_name = f"{source}_small"
        dest_dir = os.path.join(OUT, dest_name)
        os.makedirs(dest_dir, exist_ok=True)
        existing = len([f for f in os.listdir(dest_dir) if f.endswith(".png")])
        if existing >= PER_SOURCE:
            made[dest_name] = existing
            continue

        n = 0
        for f in files:
            if n >= PER_SOURCE:
                break
            try:
                im = Image.open(os.path.join(src_dir, f)).convert("RGB")
            except Exception:
                continue
            edge = random.choice(EDGES)
            quality = random.choice(QUALITY)
            out = degrade(im, edge, quality)
            if out is None:
                continue
            out.save(os.path.join(dest_dir, f"{dest_name}_{n:05d}.png"), "PNG")
            n += 1

        made[dest_name] = n
        # The derived set keeps the parent's label and family, tagged so
        # leave-one-family-out still holds parent and child out together --
        # otherwise a downscaled twin of a held-out image leaks straight back in.
        labels[dest_name] = {
            "label": meta["label"],
            "family": meta["family"],
            "dataset": meta.get("dataset", ""),
            "column": meta.get("column", ""),
            "derivedFrom": source,
        }
        print(f"{dest_name:28s} {n:4d}  label={meta['label']}  family={meta['family']}", flush=True)

    json.dump(labels, open(labels_path, "w"), indent=2)
    total = sum(made.values())
    real = sum(v for k, v in made.items() if labels[k]["label"] == 0)
    print(f"\nadded {total} small images ({real} real, {total - real} AI)")


if __name__ == "__main__":
    main()
