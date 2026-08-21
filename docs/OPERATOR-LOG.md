# Operator log

One entry per work session. Read this first; do not repeat completed work.
Numbers are never quoted without what they were measured on and how many samples.

---

## 2026-08-14 — Iteration 1

**State at pickup:** builds clean, 101 unit tests, 75.9% balanced accuracy
in-sample / 76.0% LOFO on the 3,559-image corpus (frozen calibration in
`models/pipeline.json`, reproduced from cached logits to the third decimal
before any new work: 75.892% == calibration_v2.json). Not submitted, not in git.

### Item 1 (GO/NO-GO) — DONE. VERDICT: NO-GO.

**Final numbers** (frozen shipped calibration, scorer validated by exact
reproduction of the main-corpus 75.892% before scoring anything new):

- In-distribution (main corpus, n=3,559): **75.9%** in-sample / **76.0%** LOFO.
- Unseen generators (10 post-CF-training families, n=1,082 = 782 native +
  300 small-protocol variants): pooled recall **58.9%**, all-unseen balanced
  accuracy **70.3%** (real side n=1,719 at 81.7% specificity); **68.2%**
  against non-CF-training real families only (n=1,160).
- Per family: hunyuan 85.6, hidream 82.7, imagen4/seedream45/aurora 69.1,
  flux.2 67.7, seedream3 58.8, **gpt-image-1 34.5, qwen-image 33.6,
  z-image-turbo 20.0**. Spot-checked images from the missed families: clean,
  photorealistic generations — the data is sound, the detectors are blind.
- Escape hatch tested and closed: the FP32 SigLIP stack (item 4's winner,
  +2.7 in-distribution) reaches only **73.5%** all-unseen (pooled recall
  63.0%, spec 83.9%) — still under the bar, at 490 MB.
- Structural cause: on the missed families BOTH detectors' median logits vote
  "real" (CF −2.9..−5.5, SigLIP −4.9..−8.1) on native ≥1024px images. No
  fusion/threshold change can help.
- Coverage gaps recorded: Midjourney v7 and Ideogram v3 have no public data;
  external benchmarks (2602.07814, 2607.24016) put that class at ~20–30%
  detection, so the unseen number is optimistic, not pessimistic.
- Docs updated: EVALUATION.md gained the unseen-generator section (tables,
  caveats, conclusion); README headline now carries the ~70% unseen number
  next to the 75.9%; Known Weaknesses leads with generator novelty.

### Item 1 log (how it was measured)

Goal: measure the shipped, frozen calibration on generators that postdate
Community Forensics' training data (arXiv:2411.04125, collected mid-2024).

- Corroborated the "no detector clears 75% on modern generators" claim with a
  second source: DailyBench (arXiv:2607.24016) reports 18–30% average
  detection accuracy on Flux Dev / Firefly v4 / Midjourney v7 across methods,
  and finds training-data alignment outweighs detector architecture.
  First source remains arXiv:2602.07814 (CF best open detector at 78.0% mean;
  Flux Dev 21%, MJ v7 24%, Imagen 4 19%).
- Prior art in-repo: ADVERSARIAL-REVIEW.md iteration 5 estimated 73.1% mean
  (52.7–90.3% per family) on the four then-unseen families — but against the
  OLD additive-gate calibration on the OLD 2,439-image corpus. This iteration
  measures the CURRENT shipped config on TRULY post-training generators.
- Built `tools/eval/fetch_unseen.py`: 10 unseen families —
  gpt-image-1(4o 26.3.25), seedream-3, seedream-4.5, hunyuanimage-2.1,
  imagen-4-ultra, hidream-i1, xai-aurora (all via Rapidata preference sets,
  filtered by their explicit model1/model2 columns, content-hash deduped),
  qwen-image (Ayush-Singh GenAI-bench outputs), z-image-turbo (lrzjason),
  flux.2-dev (Sarim-Hash raw PNGs, seed-42 1024px).
  **Documented gaps: Midjourney v7 (no public dataset exists — MJ has no API)
  and Ideogram v3 (bitmind/ideogram-27k is a single 37 GB zip; ckoh04's
  "ideogram-4" folders have unverifiable provenance). The unseen number
  therefore does NOT cover two of the strongest 2025 commercial generators,
  and both benchmark papers above put MJ-class detection in the ~20s.**
- Known caveat, recorded: Rapidata images arrive via HF datasets-server as
  JPEG re-encodes (same protocol that built the main corpus's HF-sourced
  families, so comparable); flux.2 arrives as raw PNG.
- Built `tools/eval/score_unseen.py`: scores the unseen corpus with the FROZEN
  pipeline.json constants (no refit — that is the point), after first
  reproducing the main-corpus in-sample number from cached logits as a
  self-check (passes: 75.892% exact).
- Fetch in progress (Rapidata row scans are slow). un_gptimage complete (80).

### Item 2 (network honesty) — DONE (code + docs + empirical closure)

- `src/content/discovery.ts`: discovery now analyses `img.currentSrc` (the
  variant the page actually loaded) instead of picking the largest srcset
  candidate; attribute-swap lazy loaders are re-checked on 'load' because
  currentSrc only updates after the new resource loads. `pickSrcFromSrcset`
  removed from `src/core/image-key.ts` (and its 5 tests); 1 new test for the
  currentSrc preference, 1 for the src-attribute fallback. 97/97 tests pass,
  typecheck + build clean.
- README: removed the false "normally a cache hit" and "runtime never opens a
  network connection" claims. Privacy section now states plainly: one outbound
  GET per analysed image to its own origin (partitioned-cache miss, empirically
  shown in ADVERSARIAL-REVIEW iteration 3), credentials omitted, offline ⇒ no
  badges. Also fixed stale corpus size (2,439 → 3,559) and test count.
- `docs/review/taint-test.mjs` (new, Playwright, Chromium): settles the
  ImageBitmap question empirically —
  cross-origin no-CORS: getImageData SecurityError; createImageBitmap OK but
  readback of the bitmap also SecurityError (taint propagates);
  JSON.stringify(ImageBitmap) === "{}" so chrome.runtime messaging (JSON)
  cannot carry pixels; CORS+crossorigin=anonymous reads OK but pages don't set
  it and re-loading with crossorigin is itself a network fetch.
  ⇒ The offscreen refetch is the only general path; honesty, not avoidance,
  was the right fix. (Live-extension confirmation of the messaging half
  belongs to item 3.)

### Item 4 (INT8-vs-FP32 re-measure) — DONE

- Completed FP32 SigLIP logits over the 1,120 corpus images added since the
  old measurement (`logits_siglip.json` now covers all 3,559; resumable run,
  84 s). Refit with `finalize_v2.py` (SG_LOGITS=fp32, OUT=calibration_v2_fp32):
  **FP32 78.60% in-sample / 78.49% LOFO vs INT8 75.89 / 76.00** → INT8 costs
  2.7 / 2.5 points on the CURRENT corpus and gate (old stale figure said 2.5 on
  the old corpus — direction and magnitude confirmed). FP32's best gate is
  steeper (a=3.0, t=−0.5): real FP <160px drops 35.5% → 25.2%, at the cost of
  small-image AI recall (66.6% → 59.8%). docs/EVALUATION.md updated, stale
  figures replaced, old-claim correction noted inline.
- Bonus finding while in the worker: `offscreenReady` was never invalidated, so
  a reclaimed offscreen document meant dead badges until the worker itself was
  torn down; now reset on sendMessage failure (untested live — item 3 must
  exercise it). Also removed the worker header's false "no network request of
  its own, ever" comment.

### Item 3 — blocked on browser this session

- claude-in-chrome extension not connected in the running Chrome; manual
  chrome://extensions load remains the only path and needs a human or a
  connected browser session.

### Final verdict for this iteration

**Does this clear 75% on unseen generators: NO** (70.3% shipped, 73.5% with
the best available upgrade; both detectors structurally blind to
GPT-image-1 / Qwen-Image / Z-Image-class models). **Should it be submitted:
NO** — the in-distribution margin is 0.9 points, the unseen deficit is ~5
points, and two independent published benchmarks agree no current detector
clears this bar on modern generators. A pass would require the maintainers'
benchmark to be dominated by 2024-era generators, i.e. the pass would be the
contamination, not the detector.

### Unknowns carried forward

- Live extension behaviour: still never observed in a real browser (item 3,
  blocked: claude-in-chrome not connected; manual chrome://extensions load
  needs a human). The offscreenReady-invalidation fix is live-untested.
- SigLIP2 training-data overlap: still unauditable upstream (unchanged).
- What the maintainers' private benchmark actually contains: unknown; the
  verdict above assumes it is not composed purely of 2024-era generators.
- Rapidata-sourced unseen images carry one datasets-server JPEG re-encode
  (same as main corpus's HF families); flux.2 family is raw PNG.
