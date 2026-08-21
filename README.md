# Forensic Lens

A Manifest V3 Chrome extension that scores the images on any page for AI
generation, running entirely on your own machine. No cloud inference, no API
calls, no local server — the models are packed into the extension and no image
data or telemetry is sent anywhere. For most images it makes **no network
request at all**: the content script hands the already-decoded pixels to the
scorer. See [Privacy](#privacy) for the exception.

Every analysed image gets a badge with a confidence score; anything at or above
the threshold (65% by default) is called out.

## How it works

```
content script            service worker              offscreen document
──────────────            ──────────────              ──────────────────
find visible images  ──▶  validate + LRU cache  ──▶   fetch bytes, decode
draw score badges    ◀──  route reply           ◀──   crop → ONNX → fuse
```

The three MV3 contexts each exist for a reason. Content scripts inherit the host
page's CSP, so they cannot host a WASM/WebGPU runtime. Service workers get torn
down after ~30 s of idle and have no GPU adapter, so a warm ONNX session cannot
live there either. An **offscreen document** is the one context with a canvas, a
GPU adapter and a lifetime the extension controls, so all decoding and inference
happens there. Inference runs on **WASM (SIMD)**, single-threaded.

WebGPU is deliberately not used, which live testing forced: inside an offscreen
document ONNX Runtime's WebGPU provider blocks the thread *synchronously* at
100% CPU and never returns, so the engine never becomes ready and the extension
looks installed while silently scoring nothing. Because the block is
synchronous, no timeout can recover it — the provider simply cannot be
attempted here. (It also mis-executes this build's INT8 operators, returning a
constant for every input; see [docs/EVALUATION.md](docs/EVALUATION.md).)

Getting pixels to the scorer takes two paths. Where the page permits it — a
same-origin image, or a cross-origin one served with CORS — the content script
draws the already-decoded image to a canvas and hands it over as a lossless PNG
`data:` URL. Nothing is re-downloaded, so analysis costs **zero network
requests** and works with networking disabled.

Where it does not — a cross-origin image without CORS taints the canvas and its
pixels cannot be read — the offscreen document falls back to fetching the URL,
and that is a real request: Chrome partitions its HTTP cache by top-level site,
so the page's cached copy is invisible to a `chrome-extension://` document.
Discovery analyses `currentSrc`, the exact variant the page displayed, never a
larger srcset candidate it never loaded, and the request carries
`credentials: 'omit'`. No image data leaves the browser on either path.

### Detection

Two complementary detectors, fused with a size gate:

| Model | Role | License |
|---|---|---|
| [Community Forensics DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (ViT-S/16 @384) | per-pixel generator fingerprints | MIT |
| [SigLIP2 AI-vs-human](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (@224, FP32 [ONNX export](https://huggingface.co/miraklodiwska1/siglip-ai-vs-human-onnx)) | semantic “does this look generated” | Apache-2.0 |

They fail in different places, and predictably so: the fingerprint reader needs
native-resolution pixels and goes blind on small images, while the semantic one
is size-insensitive but fires on real faces and paintings. Image size is free at
inference time, so it gates how far the first model is trusted. Full method,
measured numbers and known weaknesses are in **[docs/EVALUATION.md](docs/EVALUATION.md)**.

Measured on a 3,559-image corpus of real and AI images from public datasets:
**78.6% balanced accuracy** at the 0.65 threshold, **78.5%** under
leave-one-family-out cross-validation. Read that number carefully: the
fingerprint model's training data covers most of those generator families, so
it describes 2024-era generators. On a 1,082-image corpus of **generators the
detectors never saw** (GPT-image-1, Qwen-Image, Z-Image, FLUX.2, Imagen 4,
Seedream, and others), balanced accuracy drops to **~70%** — see the
[unseen-generator section](docs/EVALUATION.md#unseen-generator-evaluation-2026-08-14-the-number-that-decides-viability)
for the full breakdown. Browser-vs-Python agreement was checked
separately (`tools/parity/`) and holds to within a couple of points — see
[docs/EVALUATION.md](docs/EVALUATION.md#browser-parity-and-one-bug-it-caught),
which also explains why swapping in FP32 SigLIP2 is the first improvement worth
making.

## Build

Requires Node 22+.

```bash
npm ci
npm run fetch:models
npm run build
```

`fetch:models` is the only step that touches the network. It downloads both
models from Hugging Face, each pinned to an immutable commit SHA and verified by
SHA-256. No weights are committed and none remain in git history, so a clone is
under 1 MB (measured: 392 KB). After `fetch:models` the extension is fully
offline — you can disconnect and everything below still works.

`npm run build` writes `dist/` (~472 MB, almost all model weights).

## Install

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked** and choose the `dist/` directory
4. Visit any page with images

To analyse local pages opened as `file://`, also switch on **Allow access to
file URLs** on the extension's card. Chrome guards that setting against
scripted changes, so it is a manual step and the `file://` path is the one part
of the extension not covered by the automated live test.

The popup shows the backend in use, how many images have been analysed, and
lets you change the threshold. Loading and scoring are verified live in
Chrome 136 — see [Verify](#verify).

> Chrome removed support for the `--load-extension` command-line flag in M137,
> so the extension must be loaded through the UI as above. This affects
> automated harnesses only, not normal installation.

## Verify

Four layers, each proving something the others do not. They are listed in
increasing order of how much they actually tell you.

**1 — Unit tests.** Pure logic: view geometry, the size gate's invariants,
message validation, badge rendering against a DOM.

```bash
npm run verify    # typecheck + 97 tests + build
```

**2 — Offline accuracy.** The numbers in
[docs/EVALUATION.md](docs/EVALUATION.md#reproducing), reproduced from the
corpus with the Python harness in `tools/eval/`.

**3 — Browser parity.** Runs the *shipped* modules against the *shipped*
weights in a real browser and diffs per-image scores against the Python
reference, because canvas and Pillow are different resamplers.

```bash
node tools/parity/build-parity.mjs /tmp/fl-parity
cd /tmp/fl-parity && python3 -m http.server 8801   # then open localhost:8801
```

**4 — Live end-to-end.** Loads the packed `dist/` into a real Chrome, opens a
page of labelled images, and checks that badges appear in the page's DOM with
scores on them. This is the only layer that tests the extension rather than the
model.

```bash
PER_SOURCE=2 node tools/e2e/make-testpage.mjs /tmp/fl-testpage
cd /tmp/fl-testpage && python3 -m http.server 8799 &
CHROME_BIN="/path/to/Chrome for Testing" node tools/e2e/live-test.mjs
```

`CHROME_BIN` must be Chrome **136 or earlier** — M137 removed the
`--load-extension` flag the harness drives. That constraint is on the
*automation* only; installing by hand works on current Chrome.

### What layer 4 found

Layers 1–3 all passed while the extension was completely broken — not
degraded, but never once having worked. Two defects, both fatal, neither
reachable without a browser:

- The manifest declared `worker-src 'self' blob:`. Chrome rejects `blob:` in
  that directive and refused to load the extension **on every version**.
- Once installable, ONNX Runtime's WebGPU provider blocked the offscreen
  document's thread synchronously at 100% CPU and never returned, so the
  engine never became ready and nothing was ever scored.

Both are fixed and described in
[docs/EVALUATION.md](docs/EVALUATION.md#live-browser-verification-and-two-bugs-only-it-could-find).
Current live result on Chrome 136:

```
engine   ready · backend wasm · 2 models
page     56 images, 20 within the viewport margin → 20 scored, 20 badges
badge    "AI 38% · below 65%"
scores   AI recall 69.2% (9/13) · specificity 85.7% (6/7) · bACC@0.65 77.5%
```

Twenty of fifty-six is the intended behaviour, not a shortfall: discovery is
viewport-gated, so a page of 5,000 thumbnails costs 5,000 cheap DOM checks
rather than 5,000 inference runs. Scroll and the rest are scored.

The 77.5% is twenty images — far too small to be an accuracy measurement. It is
here to show the shipped pipeline produces sane scores end-to-end, and that it
lands near the 78.6% offline figure rather than somewhere unrelated. The real
numbers are in [docs/EVALUATION.md](docs/EVALUATION.md).

## Privacy

- No image data or telemetry is transmitted anywhere.
- **Images the page lets us read cost no network request at all.** The content
  script reads the decoded pixels and passes them inline as a lossless PNG
  `data:` URL. Verified: on a 56-image page the server logged exactly one GET
  per image — the page's own load — and none from the extension.
- **Cross-origin images without CORS are the exception.** Canvas tainting
  blocks reading their pixels (tested, not assumed —
  `docs/review/taint-test.mjs`), so those fall back to one GET to the image's
  own origin, with `credentials: 'omit'`. The image's host can observe that
  request; no third party is ever contacted. On a typical page whose images
  come from a CDN, expect this path; on a same-origin page, expect none.
  An earlier README called that refetch "normally a cache hit" and later
  "unavoidable"; both were wrong.
- `chrome.tabs.captureVisibleTab` would also avoid the request, and is still
  rejected: a capture returns pixels at **display** resolution, so a 1024px
  photo shown as a 200px thumbnail comes back as 200px of rescaled pixels,
  destroying the native detail the fingerprint detector reads in exactly the
  regime where accuracy is already worst.
- Consequence: **analysis works offline** for same-origin and CORS images.
  Cross-origin images without CORS cannot be scored with networking disabled.
- At build time, `fetch:models` downloads one pinned, hash-verified model
  artifact. Nothing else is fetched, ever.
- Nothing is written to disk beyond your settings in `chrome.storage.local`.

The `connect-src` entry in the extension CSP allows `https:`/`http:` solely so
the offscreen document can re-fetch image URLs the page already loaded; there
is no other outbound request in the codebase.

## Reproducibility

- Model weights are pinned by commit SHA and SHA-256; a mismatch fails the build.
- The SigLIP2 ONNX export is deterministic — regenerating it with
  `scripts/convert-siglip.py` produces the identical SHA-256 recorded in
  `models/registry.json` (verified by re-running the conversion from scratch).
- `dist/BUILD.json` records the digests that went into a given build.

## Licence

MIT — see [LICENSE](LICENSE). Bundled model weights keep their upstream licences
(MIT and Apache-2.0 respectively); attribution is recorded in
`models/registry.json`.
