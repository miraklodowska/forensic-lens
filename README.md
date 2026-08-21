# Forensic Lens

A Manifest V3 Chrome extension that scores the images on any page for AI
generation, running entirely on your own machine. No cloud inference, no API
calls, no local server — the models are packed into the extension and no image
data or telemetry is sent anywhere. The one network cost is stated plainly:
each analysed image is re-fetched once from its own origin (see
[Privacy](#privacy) for why).

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

Images are fetched by URL in the offscreen document rather than read off the
page, because a cross-origin image taints any canvas it is drawn into, and
extension messaging is JSON, so decoded pixels cannot be transferred out of the
content script. (`chrome.tabs.captureVisibleTab` could sidestep both — see
[Privacy](#privacy) for why it is the wrong trade here.) This refetch is **not** a
cache hit — Chrome partitions its HTTP cache by top-level site, and the
extension's offscreen document is never the page's site — so **every analysed
image costs one outbound GET to its own origin**. To keep that honest and
minimal, discovery analyses `currentSrc`, the exact variant the page already
displayed, never a larger srcset candidate the page didn't load. The request
carries `credentials: 'omit'`, and no image data leaves the browser.

### Detection

Two complementary detectors, fused with a size gate:

| Model | Role | License |
|---|---|---|
| [Community Forensics DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (ViT-S/16 @384) | per-pixel generator fingerprints | MIT |
| [SigLIP2 AI-vs-human](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (@224) | semantic “does this look generated” | Apache-2.0 |

They fail in different places, and predictably so: the fingerprint reader needs
native-resolution pixels and goes blind on small images, while the semantic one
is size-insensitive but fires on real faces and paintings. Image size is free at
inference time, so it gates how far the first model is trusted. Full method,
measured numbers and known weaknesses are in **[docs/EVALUATION.md](docs/EVALUATION.md)**.

Measured on a 3,559-image corpus of real and AI images from public datasets:
**75.9% balanced accuracy** at the 0.65 threshold, **76.0%** under
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

`fetch:models` is the only step that touches the network. It downloads one
artifact from Hugging Face, pinned to an immutable commit SHA and verified by
SHA-256, and verifies the SHA-256 of the second model, which ships in the repo.
After this the extension is fully offline — you can disconnect and everything
below still works.

`npm run build` writes `dist/` (~215 MB, almost all model weights).

## Install

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked** and choose the `dist/` directory
4. Visit any page with images

The popup shows the backend in use, how many images have been analysed, and
lets you change the threshold.

Verified live in Chrome 136 (`tools/e2e/live-test.mjs`): extension loads,
service worker starts, offscreen document builds both sessions, and badges
render in the page with scores — 77.5% balanced accuracy on a 20-image
end-to-end sample, consistent with the 75.9% offline figure.

> Chrome removed support for the `--load-extension` command-line flag in M137,
> so the extension must be loaded through the UI as above. This affects
> automated harnesses only, not normal installation.

## Verify

```bash
npm run typecheck
npm test          # 97 unit tests
npm run verify    # typecheck + test + build
```

To reproduce the accuracy numbers, see
[docs/EVALUATION.md](docs/EVALUATION.md#reproducing). To check the browser
pipeline against the Python reference:

```bash
node tools/parity/build-parity.mjs /tmp/fl-parity
cd /tmp/fl-parity && python3 -m http.server 8801
# open http://localhost:8801 — runs the shipped modules on real weights
```

## Privacy

- No image data or telemetry is transmitted anywhere.
- **At analysis time, each analysed image is re-fetched once from its own
  origin** (the same `currentSrc` URL the page already displayed, with
  `credentials: 'omit'`). The image's host can observe that second request;
  no third party is contacted. Chrome's partitioned HTTP cache means it really
  does hit the network, not the cache. An earlier version of this README
  claimed the refetch was "normally a cache hit"; that was wrong.
- Why not read the pixels off the page instead? For cross-origin images that
  does not work: canvas tainting blocks the readback and extension messaging
  is JSON, so it cannot carry an `ImageBitmap`. Both were tested rather than
  assumed — see `docs/review/taint-test.mjs`.
- One alternative *would* avoid the request: `chrome.tabs.captureVisibleTab`,
  cropping the image's rectangle out of a viewport capture. It is rejected on
  a measurement argument, not an impossibility one. A capture returns pixels at
  **display** resolution, so a 1024px photo shown as a 200px thumbnail comes
  back as 200px of rescaled pixels — destroying exactly the native-resolution
  detail the fingerprint detector reads, in precisely the size regime where
  accuracy is already worst. It trades a network request for the signal itself.
- Consequence: **analysis needs the network.** Offline, pages render their
  cached images but no badges appear.
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
