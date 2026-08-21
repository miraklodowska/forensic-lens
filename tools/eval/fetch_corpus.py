"""Fetch a diverse real/AI evaluation corpus from HF datasets-server."""
import io, json, os, random, sys, urllib.parse, urllib.request
import concurrent.futures as cf
from PIL import Image

OUT = os.path.expanduser("~/aidetect-data/corpus")
random.seed(20260813)

# (name, dataset, config, split, image_column, label, n, generator_family)
SOURCES = [
    # ---------------- REAL ----------------
    ("real_bmreal",    "bitmind/bm-real",              "default", "train", "image", 0, 90, "web-photo"),
    ("real_coco",      "rafaelpadilla/coco2017",       "default", "val",   "image", 0, 90, "coco"),
    ("real_openimg",   "bitmind/open-image-v7-256",    "default", "train", "image", 0, 80, "openimages-256"),
    ("real_coco256",   "bitmind/MS-COCO-unique-256",   "default", "train", "image", 0, 70, "coco-256"),
    ("real_celebahq",  "bitmind/celeb-a-hq",           "default", "train", "image", 0, 70, "faces-hq"),
    ("real_lfw",       "bitmind/lfw",                  "default", "train", "image", 0, 60, "faces-lowres"),
    ("real_megalith",  "bitmind/megalith-small",       "chunk_0000", "train", "image", 0, 80, "flickr"),
    # ---------------- AI ----------------
    ("ai_nanobanana",  "bitmind/nano-banana",          "default", "train", "image", 1, 90, "gemini-2.5-flash-image"),
    ("ai_journeydb",   "bitmind/JourneyDB",            "default", "train", "image", 1, 70, "midjourney"),
    ("ai_genimg_mj",   "bitmind/GenImage_MidJourney",  "default", "train", "image", 1, 60, "midjourney-genimage"),
    ("ai_flux_h3",     "bitmind/white_paper_holdout_3___FLUX.1-dev", "default", "train", "image", 1, 60, "flux.1-dev"),
    ("ai_flux_h4",     "bitmind/white_paper_holdout_4___FLUX.1-dev", "default", "train", "image", 1, 50, "flux.1-dev"),
    ("ai_imagine",     "bitmind/bm-imagine",           "default", "train", "image", 1, 60, "meta-imagine"),
    ("ai_kling",       "bitmind/klingai-images",       "default", "train", "image", 1, 50, "kling"),
    ("ai_aura",        "bitmind/bm-aura-imagegen",     "default", "train", "image", 1, 60, "aura"),
    ("ai_wukong",      "bitmind/GenImage_wukong",      "default", "train", "image", 1, 50, "wukong"),
    ("ai_realvis",     "bitmind/white_paper_holdout_3___RealVisXL_V4.0", "default", "train", "image", 1, 50, "realvisxl"),
    ("ai_sdxl_h4",     "bitmind/white_paper_holdout_4___stable-diffusion-xl-base-1.0", "default", "train", "image", 1, 50, "sdxl"),
    # ELSA_D3 -> four generators in one row
    ("ai_elsa_if",     "elsaEU/ELSA_D3", "default", "validation", "image_gen0", 1, 45, "deepfloyd-if"),
    ("ai_elsa_sd14",   "elsaEU/ELSA_D3", "default", "validation", "image_gen1", 1, 45, "sd-1.4"),
    ("ai_elsa_sd21",   "elsaEU/ELSA_D3", "default", "validation", "image_gen2", 1, 45, "sd-2.1"),
    ("ai_elsa_sdxl",   "elsaEU/ELSA_D3", "default", "validation", "image_gen3", 1, 45, "sdxl-elsa"),
    # -------- extra REAL diversity (false-positive stress) --------
    ("real_dtd",       "bitmind/dtd",                  "default", "train", "image", 0, 70, "textures"),
    ("real_caltech",   "bitmind/caltech-256",          "default", "train", "image", 0, 80, "objects"),
    ("real_bdd",       "bitmind/bdd100k-real",         "default", "train", "image", 0, 70, "dashcam"),
    ("real_mugshot",   "bitmind/idoc-mugshots-images", "default", "train", "image", 0, 60, "mugshots"),
    ("real_sky",       "bitmind/Quick-Sky-Time-Dataset", "default", "train", "image", 0, 60, "sky-timelapse"),
    ("real_wikiart",   "huggan/wikiart",               "default", "train", "image", 0, 90, "paintings"),
    ("real_openimg2",  "bitmind/open-images-v7-subset", "default", "train", "image", 0, 80, "openimages-full"),
    ("real_coco_more", "rafaelpadilla/coco2017",       "default", "train", "image", 0, 90, "coco-train"),
]

UA = {"User-Agent": "Mozilla/5.0 (research-eval)"}


def get_json(url, timeout=90, tries=8):
    delay = 3.0
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                import time as _t
                _t.sleep(delay)
                delay = min(delay * 1.8, 60)
                continue
            raise
        except Exception as e:
            last = e
            import time as _t
            _t.sleep(delay)
            delay = min(delay * 1.8, 60)
    raise last


def total_rows(ds, cfg, split):
    url = ("https://datasets-server.huggingface.co/rows?dataset=" + urllib.parse.quote(ds, safe="")
           + f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}&offset=0&length=1")
    return get_json(url).get("num_rows_total", 0)


def fetch_rows(ds, cfg, split, offset, length):
    url = ("https://datasets-server.huggingface.co/rows?dataset=" + urllib.parse.quote(ds, safe="")
           + f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}"
           f"&offset={offset}&length={length}")
    return get_json(url).get("rows", [])


def download_image(src, dest):
    req = urllib.request.Request(src, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        raw = r.read()
    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGB")
    if min(im.size) < 96:
        return None
    im.save(dest, "PNG", optimize=False)
    return im.size


def do_source(spec):
    name, ds, cfg, split, col, label, n, fam = spec
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    existing = [f for f in os.listdir(d) if f.endswith(".png")]
    if len(existing) >= n:
        return name, len(existing), "cached"
    try:
        tot = total_rows(ds, cfg, split)
    except Exception as e:
        return name, 0, f"total_rows failed: {str(e)[:80]}"

    # sample from several random windows so we don't take one contiguous block
    want = n
    got = len(existing)
    windows = []
    per = 40
    nwin = max(1, (want // per) + 2)
    hi = max(0, tot - per - 1)
    for _ in range(nwin * 3):
        windows.append(random.randint(0, hi) if hi > 0 else 0)
    recs = []
    seen_src = set()
    for off in windows:
        if got + len(recs) >= want:
            break
        try:
            rows = fetch_rows(ds, cfg, split, off, per)
        except Exception:
            continue
        for r in rows:
            v = r["row"].get(col)
            if isinstance(v, dict) and "src" in v:
                if v["src"] in seen_src:
                    continue
                seen_src.add(v["src"])
                recs.append(v["src"])
        if len(recs) >= want * 2:
            break

    idx = got
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        for src in recs:
            if idx + ok >= want:
                break
            dest = os.path.join(d, f"{name}_{idx + len(futs):05d}.png")
            futs[ex.submit(download_image, src, dest)] = dest
        for fut in cf.as_completed(futs):
            try:
                if fut.result():
                    ok += 1
                else:
                    os.path.exists(futs[fut]) and os.remove(futs[fut])
            except Exception:
                p = futs[fut]
                os.path.exists(p) and os.remove(p)
    final = len([f for f in os.listdir(d) if f.endswith(".png")])
    return name, final, "ok"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    # Sequential across sources: the datasets-server rate-limits aggressive parallel use.
    for spec in SOURCES:
        name, count, status = do_source(spec)
        print(f"{name:18s} {count:4d}  {status}", flush=True)
        meta[name] = count
    # write labels manifest
    labels = {}
    for (name, ds, cfg, split, col, label, n, fam) in SOURCES:
        labels[name] = {"label": label, "family": fam, "dataset": ds, "column": col}
    with open(os.path.join(OUT, "_labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print("TOTAL", sum(meta.values()))
