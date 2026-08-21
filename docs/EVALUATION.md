# Evaluation

Everything here is reproducible from `tools/eval/`. The corpus is built from
public Hugging Face datasets; no benchmark images, hashes or lookup tables are
embedded anywhere in the extension.

## The corpus

3,559 images: **1,719 real** and **1,840 AI-generated**, across 27 source
families. 1,236 of them are under 256px (636 real, 600 AI), added after review
showed the original corpus had almost nothing in that regime and so could not
see the size-gate failure described below. Sampled from random offsets of each dataset rather than the first N
rows, so a sorted or clustered dataset does not turn into a single scene.

| Class | Sources |
|---|---|
| Real | MS-COCO (val + train), Open Images v7 (256px and full), Megalith/Flickr, CelebA-HQ, LFW, BitMind web-photo scrape, DTD textures, Caltech-256, WikiArt paintings, mugshots |
| AI | Gemini 2.5 Flash Image (“Nano Banana”), Midjourney (JourneyDB + GenImage), FLUX.1-dev, SDXL, RealVisXL, SD 1.4, SD 2.1, DeepFloyd IF, Wukong, Kling, Meta Imagine, Aura |

Real sources deliberately include the things that make AI detectors fire on
innocent images: paintings, textures, studio portraits and mugshots. Excluding
them would have produced a much prettier number and a worse detector.

## Method

Two things are being decided, and they are independent:

**Which crops the model sees.** These detectors read per-pixel generator
traces, which live in the high frequencies that downscaling removes. Measured on
this corpus, moving Community Forensics from its documented recipe (shortest edge
→ 440, centre crop 384) to a native-resolution crop raised AUC from 0.891 to
0.909 and moved Midjourney's median score from 0.85 to 1.00. The shipped recipe
averages three views spanning both regimes. Padding and tiling small images
instead of upscaling were both tested and were *worse* (AI/real median separation
5.14 for upscale vs 3.19 reflect-pad, 4.94 tile), so upscaling stays.

**Where the threshold falls.** Community Forensics ranks well but is not
calibrated: it puts nearly every real image at ~0.000 and spreads generated ones
from 0.02 to 0.99. At its raw output, a 0.65 cut gives **AUC 0.82 but only 29%
recall** — it rejects most of the AI images it has already ranked above every
real one. Ranking quality and calibration are separate properties and a fixed
threshold consumes only the second.

So the shipped constants come from a **class-balanced logistic fit** over the two
models' aggregated logits, with the intercept shifted by `logit(0.65)` so the
fitted decision boundary lands exactly on the required threshold. The fit is
class-balanced because balanced accuracy is the target and the corpus classes are
not equal. Being monotone in each model's logit, the map cannot change either
model's ranking — it only moves the line.

### Why two models, gated by size

They fail in different places, predictably:

- **Community Forensics** (ViT-S/16 @384, fingerprint-based) has excellent
  specificity but needs native-resolution pixels. On 256px FLUX samples it
  scored **1.2% recall** — it is blind once an image is too small to crop 384px
  without upscaling.
- **SigLIP2** (semantic, @224) does not care about size and caught those same
  FLUX images at ~0.999, but fires on real content: 74% of CelebA-HQ faces and
  56% of mugshots are flagged at its raw output.

Image size is known for free at inference time, so it gates how far the
size-sensitive model is trusted (`gate.a` / `gate.t` in `models/pipeline.json`).

There is deliberately **no additive size feature**. Given one, the fit learns
"large images are real" — worth +1.2 points in-sample and pure corpus artifact.
Size may modulate trust in a model; it must never vote on its own.

**The first implementation of that principle violated it.** The gate was
additive — `w_cf + cfBig·big + cfSmall·small` — which expresses "trust CF less"
only while CF's logit carries information. Below ~256px CF saturates near −10.5
for every image, at which point `cfSmall · cf · small` stops being an
interaction and becomes a constant **+1.78 per octave** toward "AI": precisely
the standalone size prior the design excluded. The effective weight crossed zero
at ~160px and went negative, so a confident "real" from CF actively voted "AI".
A 128px image both detectors called real scored **0.766** and was flagged.

The gate is now multiplicative, `cfMax · σ(a·(octaves − t))`, bounded in
`[0, cfMax]` by construction. Losing confidence in a model can only discount it
toward silence, never invert it. `tests/unit/fusion.test.ts` asserts
non-negativity and monotonicity at every size, and that a double-"real" image is
never flagged at any resolution — the assertion that would have caught this
originally.

The lesson generalises: checking that a covariate has no *standalone* term is not
enough. An interaction with a saturating variable is a standalone term wearing a
disguise.

## Results

At the required **0.65** threshold, on the full 3,559-image corpus:

| Configuration | Balanced accuracy | LOFO mean |
|---|---|---|
| Old additive gate (the first shipped attempt) | 72.9% | — |
| **Ensemble, multiplicative gate (current)** | **75.9%** | **76.0%** |

**Leave-one-family-out** holds out each family in turn, refits the calibration
without it, and scores the held-out family. Derived small-image sets inherit
their parent's family so a downscaled twin is held out alongside its original.

### An earlier version of this document claimed 79.5%. That number was wrong.

It was measured on a 2,439-image corpus containing **zero AI images below 256px
and only 116 real ones**. The size regime where the fusion misbehaved was simply
absent, so the measurement could not see it. Re-scored on a corpus that includes
that regime, the configuration behind the 79.5% claim gets **72.9%** — below the
bar it was reported as clearing.

The two configurations on identical data:

| | balanced accuracy | real FP <160px | AI recall <160px |
|---|---|---|---|
| old additive gate | 72.9% | **99.6%** | 100% |
| new multiplicative gate | **75.9%** | 35.5% | 66.6% |

The old gate flagged essentially every real image under 160px. Its apparent
accuracy came from a corpus that never asked it about one.

Note this is not a trade: the corrected gate is better on both axes at once. The
earlier number was not a real result that we gave up to fix a bug — it was an
artifact of the evaluation set.

### Quantization cost

Re-measured 2026-08-14 on the current 3,559-image corpus and multiplicative
gate (FP32 logits: `logits_siglip.json`, refit with `finalize_v2.py`):

| SigLIP2 build | in-sample | LOFO | real FP <160px | AI recall <160px |
|---|---|---|---|---|
| INT8 (shipped) | 75.9% | 76.0% | 35.5% | 66.6% |
| FP32 | **78.6%** | **78.5%** | **25.2%** | 59.8%* |

Dynamic INT8 costs **2.7 points in-sample / 2.5 LOFO** in exchange for
343 MB → 87 MB. (An earlier version of this section carried 82.0→79.5 figures
from the old 2,439-image corpus; magnitude now confirmed on current data.)
*The FP32 fit chooses a steeper gate (a=3.0, t=−0.5) that trusts CF less on
small images, trading small-image recall for far fewer false positives there.
The loss is real and matches what the Community Forensics authors document for
their own INT8 export. It is taken deliberately: a 430 MB unpacked extension is
not something anyone will install. To trade back, run
`scripts/convert-siglip.py`, point `models/pipeline.json` at `model.onnx`, and
refit with `tools/eval/finalize_v2.py`.

Static (calibrated) INT8 should recover most of the gap and is the obvious next
improvement; the attempt here exhausted memory on a 343 MB graph.

## Unseen-generator evaluation (2026-08-14): the number that decides viability

Everything above is substantially an **in-distribution** measurement:
Community Forensics' training data (arXiv:2411.04125, collected mid-2024)
includes FLUX.1, Midjourney V5/V6, DALL-E 2/3, Ideogram V1/V2, Imagen 3,
DeepFloyd, the SD family and ~4,800 community diffusion models, plus real
images from COCO and CelebA — most of this corpus. So a second corpus was
built (`tools/eval/fetch_unseen.py`): **1,082 images from 10 generator
families that shipped after CF's data collection** — 782 native-resolution
plus 300 downscaled/recompressed by the same small-image protocol as the main
corpus. Sources and provenance audits are recorded in each family's
`_manifest.json` (Rapidata preference sets are filtered on their explicit
`model1`/`model2` columns and content-hash-deduped). Scored with the **frozen
shipped calibration — no refit** — after the scorer first reproduced the
main-corpus 75.89% from cached logits as a self-check (`score_unseen.py`).

| family | n | recall (shipped INT8) | recall (FP32 stack) |
|---|---:|---:|---:|
| hunyuanimage-2.1 | 97 | 85.6% | 92.8% |
| hidream-i1 | 110 | 82.7% | 90.0% |
| imagen-4-ultra | 110 | 69.1% | 77.3% |
| seedream-4.5 | 110 | 69.1% | 63.6% |
| xai-aurora | 110 | 69.1% | 74.5% |
| flux.2-dev | 130 | 67.7% | 66.9% |
| seedream-3 | 85 | 58.8% | 62.4% |
| **gpt-image-1 (4o)** | 110 | **34.5%** | **50.0%** |
| **qwen-image** | 110 | **33.6%** | **33.6%** |
| **z-image-turbo** | 110 | **20.0%** | **21.8%** |
| **pooled** | 1,082 | **58.9%** | **63.0%** |

Paired with this corpus's real side (which is unchanged — its specificity is
81.7% shipped / 83.9% FP32 over 1,719 images), a benchmark whose AI side were
all unseen generators scores:

| | balanced accuracy |
|---|---:|
| shipped INT8 stack | **70.3%** |
| FP32 stack (the planned upgrade) | **73.5%** |
| shipped, real side restricted to non-CF-training real families (n=1,160) | 68.2% |

**The failure is structural, not calibration.** On the three missed families
both detectors' median logits are firmly "real" (CF / SigLIP2:
gpt-image-1 −4.9 / −4.9, qwen-image −2.9 / −7.8, z-image −5.5 / −8.1, on
native ≥1024px images — CF's best size regime). No monotone fusion or
threshold can recover a signal neither model produces. The detected families
ride almost entirely on CF still firing (flux.2 +2.4 while SigLIP says −7.4).

**Known gaps that make these numbers optimistic, not pessimistic:** no public
image set exists for Midjourney v7 or Ideogram v3, so they are absent here —
independent benchmarks (arXiv:2602.07814; DailyBench, arXiv:2607.24016) put
MJ-v7-class detection at ~20–30% across all methods, and find that no
detector, of 23 tested, exceeds 75% balanced accuracy on modern commercial
generators. Caveat in the other direction: Rapidata-sourced images arrive
JPEG-re-encoded by the HF datasets-server (one extra web-typical
recompression, the same protocol that built the main corpus's HF families).

**Conclusion: on any benchmark whose AI side reflects current generators,
this extension does not clear 75% balanced accuracy at the 0.65 threshold —
~70% shipped, ~73.5% with the FP32 upgrade. The 75.9% headline above is a
real measurement of 2024-era generators, and its 0.9-point margin over the
bar does not survive generator novelty.**

## Browser parity, and one bug it caught

Offline numbers only count if the browser reproduces them. Pillow and canvas are
different resamplers, and these detectors read exactly the frequencies a
resampler touches — so `tools/parity/` runs the *shipped* modules against the
*shipped* weights in a real browser and diffs the result against Python.

| | shipped INT8 (44 images) | FP32 SigLIP2 (28 images) |
|---|---|---|
| mean \|Δ Community Forensics logit\| | 0.434 | 0.536 |
| mean \|Δ SigLIP2 logit\| | 1.84 | 1.40 |
| **max** \|Δ SigLIP2 logit\| | **16.6** | **6.6** |
| mean \|Δ probability\| | 0.040 | 0.030 |
| **verdict agreement at 0.65** | **95.5%** (42/44) | **92.9%** (26/28) |

Two things to read carefully here.

**These samples are small and the disagreements sit near the threshold.** An
early run over real images only showed 100% agreement; adding AI images dropped
it to 95.5%. Treat the offline balanced accuracy as transferring to within a
couple of points, not exactly.

**INT8 amplifies runtime divergence.** The worst single SigLIP2 logit
disagreement is **16.6 under INT8 versus 6.6 under FP32** — on one 256px FLUX
image, Python returned +6.4 (confidently generated) and the browser −10.6
(confidently real). The input files are byte-identical and both are plain RGB
PNGs, so this is not a decoding difference: ONNX Runtime's Python CPU INT8
kernels and ORT-web's WASM INT8 kernels do not agree, and quantization makes the
model brittle to that. This is a second, separate cost of INT8 beyond the 2.5
accuracy points — the FP32 number is the more trustworthy one in a browser.

FP32 SigLIP2 also loads on **WebGPU**, where the INT8 model cannot (see below),
putting both detectors on the GPU.

### Known mismatch: the harness double-counts views the runtime dedupes

`planViews` in `src/core/views.ts` drops views that resolve to the same source
rectangle, so the model is never run twice on identical pixels and no view gets
silently double weight. The Python harness does **not** do this.

It bites on square images. SigLIP2's `sq` (squash to 224) and `ar` (shortest edge
→ 224, centre crop) are the same operation when width equals height, and the
cached logits show them agreeing to the last digit:

```
ai_flux_h4_00000.png  256x256   sq +6.3934   ar +6.3934   nc +5.4442
```

Python averages all three, weighting that view 2/3; the browser averages two,
weighting it 1/2. The effect is small — about 0.16 of a logit on the example
above, nowhere near the 17 that sent me looking — but it means the shipped
calibration was fitted under a slightly different view weighting than the one
the extension applies.

The correct fix is to make the harness mirror `planViews` and refit, not to
remove the dedupe: double-counting a view is the actual bug. It is left as-is
here because any move to FP32 SigLIP2 requires a refit anyway, and both changes
should land in the same pass.

This harness earned its keep immediately. On the first run, SigLIP2's logit came
back as the *identical constant* `0.3457602187991142` for every image, while
Community Forensics varied correctly. ONNX Runtime's **WebGPU backend does not
implement this build's INT8 operators and silently returns a constant instead of
failing** — a detector that looks alive and has stopped discriminating entirely.
Nothing in the Python evaluation could have seen it, because Python never runs
the WebGPU path.

The fix is per-model execution providers in `models/pipeline.json`: FP32
Community Forensics runs on WebGPU, the INT8 model pins to WASM. Forcing both to
WASM (`?ep=wasm`) reproduces the correct varying logits, which is how the
diagnosis was confirmed.

## Performance

Roughly **4.4 s per image** on an M-series Mac with the page focused (three
384px views on WebGPU plus three 224px views on single-threaded WASM). In a
background tab, Chrome's throttling pushes that to ~35 s.

Single-threaded WASM is the bottleneck: extension pages are not cross-origin
isolated, so `SharedArrayBuffer` is unavailable and ORT cannot use threads. The
three routes to improving it, roughly in order of payoff:

1. **Ship FP32 SigLIP2 on WebGPU.** Verified to load and run there, putting both
   detectors on the GPU and removing the WASM bottleneck entirely — and it is
   worth +2.5 accuracy points and far better browser/Python agreement besides.
   The cost is 343 MB instead of 87 MB (a ~490 MB extension). On the evidence
   collected here this is the right trade, and it is the first change to make.
2. Set `cross_origin_embedder_policy` / `cross_origin_opener_policy` in the
   manifest to unlock threaded WASM — but COEP also blocks the cross-origin image
   fetches the offscreen document depends on, so image bytes would have to be
   routed through the service worker instead.
3. Drop SigLIP2 to its single `sq` view — 3× less WASM work for about 3 points
   (80.2% → 77.2% in the FP32 ensemble).

Steady-state per-image throughput for the FP32/WebGPU configuration was not
cleanly measured (the runs include one-off shader compilation), so it is quoted
as "removes the bottleneck in principle" rather than with a number.

## Known weaknesses

These are real and are not hidden by the headline number.

**Generators newer than the training data** are the dominant weakness — see
the unseen-generator section above: 20–35% recall on GPT-image-1, Qwen-Image
and Z-Image, with both detectors confidently voting "real".

**AI images we miss** (held-out recall): Aura 24%, FLUX.1-dev 46%, RealVisXL
53%, Gemini 2.5 Flash Image 56%. The small-image blind spot is the dominant
factor — every high-resolution family scores 82–100%.

**Real images we wrongly flag** (held-out specificity): paintings 44%,
Caltech objects 53%, CelebA-HQ faces 55%, mugshots 59%. All driven by SigLIP2,
whose errors are hard: it outputs 0.000 or 1.000, so no monotone calibration can
soften them. Non-photorealistic real art is the single worst case, because
"not photographic" and "AI-generated" are genuinely entangled for a semantic
model.

## Reproducing

```bash
python3 tools/eval/fetch_corpus.py      # builds the corpus from public datasets
python3 tools/eval/cache_logits.py      # Community Forensics, all crops
python3 tools/eval/siglip_eval.py       # SigLIP2, all crops
python3 tools/eval/finalize.py          # fits and prints the shipped constants
```

`finalize.py` writes exactly the `fusion` block that appears in
`models/pipeline.json`. Corpus construction samples at a fixed seed (20260813),
so a re-run selects the same images.
