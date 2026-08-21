#!/usr/bin/env python3
"""Regenerate models/weights/siglip-detector-int8.onnx from upstream weights.

The extension ships this file because upstream publishes safetensors only and
there is no permissively-licensed ONNX export of it on the Hub. Running this
script reproduces the checked-in artifact byte-for-byte: the SHA-256 it prints
should match `sha256` for that file in models/registry.json.

    pip install torch transformers onnx onnxruntime
    python3 scripts/convert-siglip.py

Nothing in the extension depends on Python — this is a one-off provenance tool
so the bundled binary is auditable rather than opaque.
"""

import hashlib
import json
import os
import sys

import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "weights")
WORK = os.path.join(WEIGHTS, "siglip")
FINAL = os.path.join(WEIGHTS, "siglip-detector-int8.onnx")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Wrapped(torch.nn.Module):
    """Exposes just the logits, so the ONNX graph has one clean output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    os.makedirs(WORK, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID).eval()

    size = processor.size
    height = size.get("height") or size.get("shortest_edge")
    width = size.get("width") or size.get("shortest_edge")
    print(f"id2label={model.config.id2label}  input={height}x{width}")
    print(f"image_mean={processor.image_mean} image_std={processor.image_std}")

    fp32 = os.path.join(WORK, "model.onnx")
    if not os.path.exists(fp32):
        print("exporting fp32 ONNX ...")
        torch.onnx.export(
            Wrapped(model),
            (torch.randn(1, 3, height, width),),
            fp32,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"fp32: {os.path.getsize(fp32) / 1e6:.1f} MB")

    print("quantizing to dynamic INT8 ...")
    quantize_dynamic(
        fp32, FINAL, weight_type=QuantType.QInt8, extra_options={"MatMulConstBOnly": True}
    )

    digest = sha256(FINAL)
    print(f"\nwrote {FINAL}")
    print(f"  {os.path.getsize(FINAL) / 1e6:.1f} MB")
    print(f"  sha256 {digest}")

    registry = json.load(open(os.path.join(ROOT, "models", "registry.json")))
    expected = next(
        (a["sha256"] for a in registry.get("bundled", []) if a["file"] == os.path.basename(FINAL)),
        None,
    )
    if expected is None:
        print("  (no registry entry to compare against)")
    elif expected == digest:
        print("  ✓ matches models/registry.json")
    else:
        print(f"  ✗ registry expects {expected}")
        sys.exit(1)


if __name__ == "__main__":
    main()
