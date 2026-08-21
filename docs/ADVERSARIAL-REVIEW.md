# Adversarial review ledger

Goal: find a concrete reason this submission would FAIL the poidh bounty.
Each entry: surface attacked, what was done, verdict, severity, repro.
Verdicts: CONFIRMED / REFUTED / INCONCLUSIVE. Severity: fatal / serious / cosmetic.

---

## FINAL VERDICT (loop closed after 7 iterations, all listed surfaces attacked)

**Would this clear a 75% balanced-accuracy bar on an unseen benchmark: NO —
more likely than not it falls short.** Converging fully-held-out estimates:
family-level 2-fold CV mean **74.3%** (52% of folds < 75%); shared-source
grouping **71.8%** (66% < 75%); genuinely-unseen generators (the ones
Community Forensics never trained on — the definition of an unseen
benchmark) mean **73.1%**, as low as 52.7% per family. The published 78.8%
LOFO is half in-sample by construction (77.8% when scored honestly) and is
measured mostly on generators and real sources inside the fingerprint
model's training set. Add browser-runtime noise (their own parity: 95.5%
verdict agreement, INT8-brittle) and the sub-200px chance-level regime, and
75% requires a favourably-mixed benchmark: SD-era generators, large images,
photographic real content. A 2025-era benchmark (FLUX/nano-banana-class
generators, thumbnails, art) lands in the 60s.

**Single biggest risk:** the headline number is substantially an
in-distribution measurement — CF trained on 10 of the eval's 14 AI
generator families and on COCO/CelebA real images, so the eval cannot see
how the ensemble does on new generators; where the corpus does contain
truly-unseen ones, held-out accuracy is 52.7–90.3%, mean 73.1%.

**Beyond the accuracy bar, two claims are demonstrably false as shipped:**
(1) "the runtime never opens a network connection" — every analysed image
is re-fetched over the network (cache-partition miss, empirically
demonstrated; srcset picking fetches hi-res variants the page never
loaded); with internet off, the extension goes inert. (2) The end-to-end
extension has never been observed working in a live browser — the author's
own e2e artifact scored 0/84, and no 2026 browser on this machine will
load it via the provided harness. What IS solid: the numbers reproduce from
cached logits to the digit, the clean-room build is byte-identical, 90/90
tests pass, the perf claim is honest, and the documented INT8-on-WebGPU
constant bug is real and correctly mitigated.

**Most damaging single finding for everyday use (iteration 2):** the size
gate degenerates into a "small ⇒ AI" vote — at the default 128px minimum,
**every real photo thumbnail at 128–160px tested was flagged as AI at
p≈0.7–0.9** (8/8 at both sizes), a regime the corpus cannot see because it
contains zero AI images below 256px.

---

## Iteration 1 — 2026-08-13 — Accuracy claim: leakage in the "honest" LOFO number

**Surface:** ACCURACY IS FAKE / CALIBRATION OVERFIT. Does the claimed 79.5%
in-sample / 78.8% LOFO survive a genuinely held-out evaluation?

**What was done:**

1. Re-ran `tools/eval/finalize.py` against the cached logits in
   `~/aidetect-data/` (corpus + logits present on disk, 2,439 rows).
2. Verified the shipped `models/pipeline.json` fusion block byte-matches
   `~/aidetect-data/calibration_int8.json` (INT8 fit: in-sample 79.52%,
   LOFO 78.75%) and that the default finalize.py run (FP32 logits) reproduces
   the doc's FP32 row (82.03% / 80.94%). **The published numbers reproduce
   exactly from the cached logits — no fabrication.**
3. Wrote `docs/review/attack_holdout.py`, which mirrors finalize.py's
   features/fitter/threshold bit-for-bit (validated: reproduces the shipped
   LOFO protocol at exactly 78.75%), then re-scores under honest protocols.

**Findings:**

- **CONFIRMED (serious): the LOFO protocol is half in-sample.** In
  `finalize.py`, when an AI family is held out, recall is computed on the
  held-out family but specificity on ALL real images — which were in the
  refit's training set (and symmetrically for real families). See
  `tools/eval/finalize.py` lines 87–89. A fully-out-of-sample pairwise LOFO
  (fit without one AI and one REAL family, score only on the two held-out
  families, all 182 pairs) gives **77.79% vs the claimed 78.75%** — ~1 point
  of inflation in the number the docs call "the honest estimate".

- **CONFIRMED (serious, fatal *for the 75%-bar question*): performance on an
  unseen family mix straddles the bar and is mix-dominated.** Repeated
  family-level 2-fold CV (400 folds, both classes fully unseen in the test
  fold): mean bACC **74.28%**, median 74.57%, 5th pct 63.5%, **52% of folds
  below 75%**. Grouping families that share an underlying dataset/prompt set
  (coco×3, openimages×2, ELSA_D3×4, GenImage×2, bitmind-whitepaper×3):
  mean **71.80%**, **66% of folds below 75%**. Caveat, stated honestly:
  2-fold halves the training set, which is somewhat pessimistic for the
  shipped fit (trained on all 27 families). The sharper result is the
  per-family holdout spread from the pairwise protocol: held-out
  **aura 52.7%, flux.1-dev 54.0%, paintings 56.1%, objects 61.1%,
  realvisxl 62.1%** — an unseen benchmark weighted toward recent generators
  and non-photographic real content lands well under 75%; one weighted
  toward SD-era generators and COCO-like photos lands well over 80%.

- **REFUTED: "the numbers are fabricated / don't reproduce".** They reproduce
  to the digit from the cached logits, and shipped constants match the fit.

- **INCONCLUSIVE (cosmetic): two configured real families are silently
  missing.** `fetch_corpus.py` lists `real_bdd` (dashcam, 70) and `real_sky`
  (sky-timelapse, 60); both corpus dirs exist and are empty, no logits were
  ever cached for them, and EVALUATION.md's source table omits them. Looks
  like a silent fetch failure, not score-then-drop (no evidence any score was
  ever produced). Dashcam-style real photos (blur, heavy compression) are
  false-positive bait, so their absence plausibly flatters specificity, but
  this is untested.

**Repro:**

```bash
cd ~/forensic-lens-ensemble
python3 tools/eval/finalize.py                 # 82.03 / 80.94 (FP32 default paths)
python3 -W ignore docs/review/attack_holdout.py  # [A] 78.75 exact repro of shipped LOFO
                                                 # [B] 77.79 fully-held-out pairwise
                                                 # [C] 74.28 family 2-fold
                                                 # [D] 71.80 shared-source-group 2-fold
```

**Not yet explored (for later iterations):** SigLIP2 training-set overlap with
eval sources (needs model-card research); silent-wrongness on degenerate
inputs (1x1, CMYK, animated, EXIF-rotated, 16-bit); network-request audit of
dist/ in a live browser; live extension smoke test; clean-room build
reproducibility; performance measurement on real pages.

---

## Iteration 2 — 2026-08-13 — Silent wrongness: the size gate is a "small ⇒ AI" vote

**Surface:** SILENT WRONGNESS. Ran the *shipped* modules (`planViews`,
`renderView`, `fuse`) with the *shipped* weights in a real Chromium
(Playwright, cf on WebGPU / sg on WASM — same providers as the extension) via
a modified `tools/parity` harness, on degenerate inputs and on real/AI
thumbnails. Repro scripts: `docs/review/run-degen.mjs` + scratchpad image
generator (documented below).

**Findings:**

- **CONFIRMED (fatal for everyday-browsing usefulness): every real photo
  thumbnail at 128–160px is flagged as AI with high confidence.** 24 real
  photos (COCO, Flickr/Megalith, BitMind web photos) downscaled to common
  thumbnail sizes and run through the browser pipeline: **128px: 8/8 flagged
  (mean p=0.80); 160px: 8/8 flagged (mean p=0.70); 192px: 3/8 flagged.**
  AI thumbnails are also all flagged, i.e. below ~200px the detector outputs
  "AI" for essentially everything — balanced accuracy in this band collapses
  to ~50% (chance) while showing users confident 70–90% badges on ordinary
  photos. The extension's default `minImageSize` is **128**, so this band is
  scored by default on every gallery/avatar/product-grid page.
  **Mechanism** (verified against `src/core/fusion.ts` + shipped constants):
  CF goes blind on small images and pins at logit ≈ −10.5 for *all* content;
  the gate term `cfSmall·cf·small ≈ (−0.169)·(−10.5)·small ≈ +1.78 per octave
  below 384px` becomes a constant-coefficient **standalone size feature voting
  "AI"** — precisely the artifact the docs claim was engineered out ("size may
  modulate trust in a model; it must never vote on its own"). At 128px the
  term contributes +2.81 log-odds; a real COCO photo that scores p=0.18 at
  full size scores p≈0.79 at 128px with unchanged semantics.
  **Why the corpus hid it:** the [128,256) band contains **116 real and 0 AI
  images** (mostly 250px LFW faces, where `small`=0.62 keeps the term mild),
  so neither the in-sample fit nor LOFO ever measured this regime. Extreme
  case: a 1×1 image scores **p=0.99** (min-size filter blocks it in the
  extension, but it shows the extrapolation direction).

- **CONFIRMED (serious for the parity claim, cosmetic for users): alpha is
  ignored and read as black.** Two files with identical visible content but
  different RGB stored under transparent pixels score **bit-identically** in
  the browser (transparent → premultiplied black), while Pillow's
  `.convert("RGB")` in `tools/eval/evalkit.py` keeps the stored RGB (mean
  pixel diff 95.6/255 on the test pair). So for any transparent PNG the
  Python reference and the browser score *different images*; the browser also
  scores a black background the page never displays (vs white-composited
  version: Δsg 0.72). No verdict flip demonstrated on my pair; the divergence
  mechanism is proven.

- **CONFIRMED (cosmetic): SVG images error out silently.**
  `isAnalyzableSource` accepts any https URL, so SVG `<img>`s ≥128px are
  discovered and sent to the offscreen document, where `createImageBitmap`
  rejects them ("The source image could not be decoded"); content.ts drops
  non-ok replies without a badge. SVGs are simply never analysed — no crash,
  no retry loop, but a silent coverage gap the docs do not mention.

- **REFUTED: broad decode wrongness.** Grayscale, 16-bit PNG (truncated to
  8-bit identically), palette PNG, CMYK JPEG, 8000×6000 JPEG, animated GIF,
  and APNG all decode and score sanely (no constants, no NaN). EXIF
  orientation **is** honoured by `createImageBitmap` (rotated dims observed).
  Animated images are scored on frame 1 only — defensible, undocumented.

**Repro:**

```bash
node tools/parity/build-parity.mjs <dir>   # then drop test images in <dir>/img,
                                           # rewrite <dir>/images.json to point at them
python3 -m http.server 8802 --directory <dir> --bind 127.0.0.1
node docs/review/run-degen.mjs             # Playwright; dumps window.__parity rows
# thumbnail test: real corpus photos resized to min-edge 128/160/192 (LANCZOS)
```

---

## Iteration 3 — 2026-08-13 — It never ran live, and inference is not network-silent

**Surface:** IT NEVER RAN + RULE COMPLIANCE.

**What was done:**

1. Built the repo's own e2e test page (`tools/e2e/make-testpage.mjs`, 28
   images, served on 127.0.0.1:8799 with request logging) and ran
   `tools/e2e/run-chrome.mjs` against every Chromium on this machine.
2. Wrote `docs/review/diag-ext.mjs` to read `chrome://extensions-internals`
   and see whether `dist/` actually installs.
3. Wrote `docs/review/partition-test.mjs` + a Cache-Control-serving logged
   HTTP server to test the offscreen document's "normally a cache hit" fetch
   assumption under Chrome's HTTP cache partitioning.

**Findings:**

- **CONFIRMED (serious): the extension has never scored a single image in a
  live browser, and on current Chrome it cannot even be loaded by the
  submission's own harness.** Evidence: (a) `/tmp/fl-e2e-results.json`,
  written by the author's earlier harness run today at 18:47, records
  **`scored: []` — 0 of 84 figures done**; (b) my runs: Playwright Chrome for
  Testing 151 and unbranded Chromium 141 both refuse `--load-extension`
  (extensions-internals shows only the two component extensions; no service
  worker ever appears); (c) no other artifact of a live run exists in the
  repo. The numeric core (decode→views→ONNX→fuse) IS verified in-browser by
  the parity harness, but the extension glue — discovery, service-worker
  routing, offscreen lifecycle, badge painting — is verified only by unit
  tests with mocked Chrome APIs. The README's manual-UI install path is
  plausible but unverified by anyone, including the author.

- **CONFIRMED (serious): "the runtime never opens a network connection" is
  false at inference time.** The offscreen document re-fetches every image
  URL (`src/offscreen/offscreen.ts` `loadBitmap`, `cache: 'force-cache'`),
  betting on a browser-cache hit. Chrome partitions its HTTP cache by
  top-level site (default since M85); a `chrome-extension://` offscreen
  document is never the page's site, so the bet loses every time. Empirical
  demonstration with the exact fetch options (`credentials:'omit',
  cache:'force-cache'`): an image loaded by a page on `127.0.0.1:8807` and
  fresh in cache (max-age 3600) is **re-requested from the server** when
  fetched from a `localhost:8807` top-level page (different partition), while
  the same-partition refetch produces **no request** (server log: 1 baseline
  + 1 page load + 1 cross-partition = 3 GETs; same-partition fetch added
  none). Consequences: **one outbound GET to the image host per analysed
  image** (the extension CSP's `connect-src https: http:` exists precisely
  because the runtime does open network connections); and **with internet
  disabled the extension goes inert** — the page renders its cached images
  but the offscreen partition is cold, `fetch` rejects, and no badge ever
  appears, directly contradicting "you can disconnect and everything below
  still works". *Caveat:* demonstrated cross-site rather than from a real
  `chrome-extension://` context, because the extension cannot be loaded (see
  above); the partition mechanism is the same, but a live-extension
  confirmation is queued for the next iteration via an older Chromium.

**Repro:**

```bash
PER_SOURCE=1 node tools/e2e/make-testpage.mjs <dir> && python3 -m http.server 8799 --directory <dir>
CHROME_BIN=<any 2026 Chrome/Chromium> node tools/e2e/run-chrome.mjs   # -> NO service worker
node docs/review/diag-ext.mjs                                         # -> extension absent
node docs/review/partition-test.mjs                                   # -> 3rd GET appears cross-partition
python3 -c "import json;d=json.load(open('/tmp/fl-e2e-results.json'));print(d['scored'])"  # []
```

---

## Iteration 4 — 2026-08-13 — Live-load exhausted; clean-room build reproduces

**Surface:** IT NEVER RAN (live-load escalation) + REPRODUCIBILITY.

**Findings:**

- **INCONCLUSIVE (but damning for verifiability): there is no way to run this
  extension in an automated live browser on a 2026 macOS.** Five escalating
  attempts: (1) Playwright Chrome for Testing 151 — `--load-extension`
  refused; (2) unbranded Chromium 141 — refused (extensions-internals shows
  only component extensions); (3) Chrome for Testing 136 (pre-M137, flag
  should work), windowed — process starts but hangs before CDP comes up,
  twice, ≥5 min; (4) CfT 136 `--headless=new` — starts fine but silently
  ignores `--load-extension` (no extension target, no error logged); (5)
  Chromium 130 snapshot (r1356025) — runs `--version` from CLI but dies
  instantly when launched windowed on this macOS. Consequence: the
  submission's e2e harness is unrunnable here by anyone, the author's own
  artifact scored 0/84 (iteration 3), and the only remaining install path —
  manual `chrome://extensions` UI loading — has no recorded verification.

- **REFUTED: "the build does not reproduce".** Simulated a fresh checkout
  (rsync of the project minus `.gitignore`d paths — `node_modules/`, `dist/`,
  `models/weights/siglip/` — and minus `cf-vit.onnx` to force the download
  path). `npm ci` → `npm run fetch:models` (downloads cf-vit.onnx from the
  pinned immutable HF revision, SHA-256 verified; verifies the bundled INT8
  SigLIP against the registry) → `npm run verify`: typecheck passes, **90/90
  unit tests pass**, build completes, and the resulting `dist/BUILD.json`
  digests are **byte-identical** to the shipped build.

- **Noted (cosmetic, packaging): the submission is not a git repository.**
  There is no `.git` here; "clone and build" cannot literally be exercised,
  and the 83 MB bundled INT8 model's presence in whatever repo is eventually
  pushed is unverified. Also unverified (toolchain absent): the claim that
  `scripts/convert-siglip.py` regenerates the INT8 file to the identical
  SHA-256 — taken on faith; the shipped file does match the registry hash.

**Repro:**

```bash
rsync -a --exclude node_modules --exclude dist --exclude 'models/weights/siglip' \
  --exclude 'models/weights/cf-vit.onnx' ~/forensic-lens-ensemble/ /tmp/clean/
cd /tmp/clean && npm ci && npm run fetch:models && npm run verify   # all green
diff <(python3 -m json.tool dist/BUILD.json) <(python3 -m json.tool ~/forensic-lens-ensemble/dist/BUILD.json)
```

---

## Iteration 5 — 2026-08-13 — Detector training data overlaps the eval corpus; 0.65 does not transfer

**Surface:** ACCURACY IS FAKE (training-set overlap) + CALIBRATION OVERFIT
(operating point).

**What was done:** Fetched the model cards / paper for both detectors;
cross-referenced the documented training data of Community Forensics
(arXiv:2411.04125, CVPR 2025) against the eval corpus families; quantified
the held-out gap between generators CF trained on and generators it never
saw, using the fully-held-out pairwise protocol from iteration 1; measured
out-of-sample threshold optimality on 60 family-level 2-fold splits.

**Findings:**

- **CONFIRMED (serious): the eval corpus is substantially in-distribution
  for the fingerprint detector.** Community Forensics' documented training
  data includes real images from **COCO and CelebA** (both eval real
  sources — explaining the suspicious 99–100% LOFO specificity on
  coco/coco-train) and fake images from **Midjourney V5/V6, FLUX.1, SDXL,
  SD 1.x/2.x, DeepFloyd** plus ~4,763 systematically-sampled HF diffusion
  models (which plausibly cover RealVisXL and Wukong too). That makes **10
  of the 14 eval AI families generators CF trained on**. Fully-held-out
  bACC by coverage: **CF-seen generators mean 79.7%; genuinely unseen
  generators (gemini-2.5-flash-image, kling, meta-imagine, aura) mean
  73.1%** — below the 75% bar. An unseen benchmark is, by construction, all
  unseen generators; 73.1% (with per-family spread 52.7–90.3%) is the best
  available estimate of that setting from this corpus.

- **INCONCLUSIVE: SigLIP2 detector training overlap.** The
  Ateeqq/ai-vs-human-image-detector card discloses only "60,000 AI + 60,000
  human images" with **no sources named**, and itself notes "some users
  reported overfitting issues". Overlap with the eval families can be
  neither proven nor excluded — which for a bounty reviewer is itself a
  problem: the semantic half of the ensemble has an unauditable training
  set.

- **CONFIRMED (serious): the 0.65 operating point does not transfer out of
  sample.** The fit pins the decision boundary to 0.65 by construction
  (in-sample). On 60 family-level 2-fold splits with both classes held out,
  the bACC-optimal threshold has **median 0.71, IQR [0.49, 0.85]**, and
  scoring at 0.65 sacrifices **mean 3.5 bACC points (max 14.3)** versus the
  fold-optimal threshold. The required threshold is fixed by the bounty, so
  this is not cheating — but it means the "calibrated so 0.65 is the
  boundary" property is a corpus artifact, not a stable property of the
  detector.

**Repro:** iteration-5 code block in `docs/review/` history (inline script:
reuses `attack_holdout.py` machinery; seen/unseen sets per the CF paper's
dataset section; threshold sweep np.linspace(0.05,0.95,181)).

---

## Iteration 6 — 2026-08-13 — Performance measured; srcset compounds the network finding

**Surface:** PERFORMANCE AS DISQUALIFIER + residual review of the
never-live-tested glue (`image-key.ts`, `queue.ts`, `service-worker.ts`).

**Findings:**

- **REFUTED (performance as a *wrong claim*): the ~4.4 s/image number is
  honest — conservative, even.** Measured the shipped pipeline
  (cf:webgpu + sg:wasm, same providers as the extension) on 28 realistic
  corpus images in Chromium via the parity harness with 100 ms polling:
  **steady-state mean 1.73 s/image (median 1.47, p90 2.17)**, 2.5 s for the
  first image including model load and shader compile, 49.3 s total for 28
  images (`docs/review/perf-run.mjs`). The doc's 4.4 s claim presumably
  includes extension round-trip overhead on other hardware; either way it is
  not understated. The usefulness arithmetic still stings — a 30-image page
  takes ~1–2 min to fully badge serially, a 200-image board 5–15 min, and
  the docs themselves admit ~35 s/image in background tabs — but that is a
  disclosed limitation, not a false claim.

- **CONFIRMED (serious, compounds iteration 3): on responsive pages the
  extension downloads image variants the page never loaded.**
  `pickSrcFromSrcset` (src/core/image-key.ts:86) deliberately selects the
  **highest-resolution** srcset candidate ("more pixels means a more
  reliable forensic signal"), and discovery prefers srcset over `src`
  (`rawSource`, src/content/discovery.ts:35). A news site serving 400 px
  layout images with 2–4 K srcset variants will have the offscreen document
  fetch the multi-megabyte hi-res variant of *every* analysed image over
  the network — a guaranteed cache miss independent of cache partitioning,
  multiplying inference-time bandwidth (hundreds of MB on a big gallery)
  on top of the already-confirmed per-image network request.

- **INCONCLUSIVE (code-level hunches, unprovable without a live browser —
  clearly labelled as such):** (a) `offscreenReady`
  (src/background/service-worker.ts:27) is never invalidated after success;
  if Chrome reclaims the offscreen document, every inference fails
  ("Receiving end does not exist") until the idle service worker itself is
  torn down and restarted (~30 s+ window of dead badges). (b) All popup
  stats (`analyzed`, `flagged`, backend) and the 600-entry score cache are
  module state in an MV3 service worker and silently reset on every ~30 s
  idle teardown — the popup can show near-zero counts on a long-lived page.
  Queue and LRU logic read correct; unit tests cover them.

**Repro:**

```bash
# perf: images.json in the parity dir pointed at 28 corpus images, then
node docs/review/perf-run.mjs      # backend cf:webgpu sg:wasm — 1.73s/img steady state
```

---

## Iteration 7 — 2026-08-13 — Residual sweep: nothing new; loop closed

**Surface:** verification of the remaining load-bearing claim (INT8-on-WebGPU
constant output) + survey of any unexplored residue.

**Findings:**

- **REFUTED (their claim verified): the documented INT8-on-WebGPU silent
  constant is real and still present.** Forcing both models onto WebGPU
  (`?ep=webgpu` on the parity harness) returns the **identical sg logit
  0.3458 for all 28 images** (matching the documented 0.34576…) while cf
  varies normally. The shipped per-model provider pin (INT8 → WASM) is the
  correct mitigation and is what `models/pipeline.json` does.

- No new attack surfaces remain from the brief: accuracy/leakage (iters 1,
  5), calibration/threshold (iters 1, 5), silent wrongness (iters 2, 7),
  rule compliance (iter 3), live-run status (iters 3, 4), reproducibility
  (iter 4), performance (iter 6) are all logged with repros. Unverifiable
  on this machine and left open: `convert-siglip.py` determinism (needs a
  torch toolchain), live-extension confirmation of the partition-miss
  refetch, and SigLIP2 training-data overlap (training set undisclosed
  upstream).

Loop stop criterion reached: this iteration produced nothing new and every
listed surface is exhausted. Final verdict at the top of this ledger.
