# Evaluation harness

Reproduces every number in [../../docs/EVALUATION.md](../../docs/EVALUATION.md).
Needs Python 3.11+ with `onnxruntime numpy pillow`.

```bash
python3 -m venv .venv && .venv/bin/pip install onnxruntime numpy pillow
.venv/bin/python tools/eval/fetch_corpus.py    # ~2,400 images from public HF datasets
.venv/bin/python tools/eval/cache_logits.py    # Community Forensics, 9 crops per image
.venv/bin/python tools/eval/siglip_eval.py     # SigLIP2, 3 crops per image
.venv/bin/python tools/eval/finalize.py        # fits + prints the shipped constants
```

`finalize.py` emits exactly the `fusion` block in `models/pipeline.json`.

Other scripts:

- `ensemble2.py` — compares feature sets for the fusion (this is where the
  standalone-size-feature leak was found and rejected).
- `pad_test.py` — pad vs tile vs upscale for images smaller than the crop.
- `evalkit.py` — shared preprocessing/metrics; mirrors `src/core/views.ts`.

Paths default to `~/aidetect-data/`; override with the `CORPUS`/`MODELS` env
vars or edit the constants at the top of each script.
