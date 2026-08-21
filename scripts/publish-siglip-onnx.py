#!/usr/bin/env python3
"""Publish the FP32 SigLIP ONNX export to the Hugging Face Hub.

The repository cannot carry this file: at 343 MB it is past GitHub's 100 MB
per-file hard limit, and Git LFS would put a ~1 GB/month bandwidth quota between
the repo and anyone cloning it. So the weights live on the Hub and
`npm run fetch:models` downloads them pinned to a commit and verified by
SHA-256 — the same mechanism the Community Forensics model already uses.

Run once. Needs a Hugging Face token with write scope:

    hf auth login          # or: export HF_TOKEN=...
    python3 scripts/publish-siglip-onnx.py

Prints the commit sha to pin in models/registry.json.
"""

import hashlib
import os
import sys

from huggingface_hub import HfApi

STAGING = os.environ.get("STAGING", "/tmp/hf-upload")
REPO = os.environ.get("HF_REPO", "siglip-ai-vs-human-onnx")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    api = HfApi()
    try:
        me = api.whoami()
    except Exception as exc:
        print(f"not authenticated: {exc}\n\nRun `hf auth login` first.", file=sys.stderr)
        sys.exit(1)

    user = me["name"]
    repo_id = f"{user}/{REPO}"
    model = os.path.join(STAGING, "model.onnx")
    if not os.path.exists(model):
        print(f"missing {model} — run scripts/convert-siglip.py first", file=sys.stderr)
        sys.exit(1)

    digest = sha256(model)
    size = os.path.getsize(model)
    print(f"user      {user}")
    print(f"repo      {repo_id}  (public)")
    print(f"model     {size / 1e6:.1f} MB")
    print(f"sha256    {digest}\n")

    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    print("uploading (343 MB, this takes a few minutes)…")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=STAGING,
        repo_type="model",
        commit_message="ONNX FP32 export of Ateeqq/ai-vs-human-image-detector",
    )

    info = api.model_info(repo_id)
    print("\ndone.")
    print(f"  url       https://huggingface.co/{repo_id}")
    print(f"  revision  {info.sha}")
    print(f"  sha256    {digest}")
    print(f"  sizeBytes {size}")
    print("\nPin these in models/registry.json under `artifacts`.")


if __name__ == "__main__":
    main()
