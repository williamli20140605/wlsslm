#!/usr/bin/env python3
"""Fetch and vendor Qwen3 tokenizer files into assets/tokenizer/qwen3/.

This script downloads tokenizer artifacts only (no model weights).
It tries Qwen/Qwen3-0.6B first, then falls back to Qwen/Qwen3-1.7B.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

PRIMARY_MODEL_ID = "Qwen/Qwen3-0.6B"
FALLBACK_MODEL_ID = "Qwen/Qwen3-1.7B"
TARGET_DIR = Path("assets/tokenizer/qwen3")
METADATA_PATH = TARGET_DIR / "source_model.json"

# Keep this list focused on tokenizer artifacts only.
ALLOW_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "tokenizer.model",
]


def model_exists(model_id: str) -> bool:
    api = HfApi()
    try:
        api.model_info(model_id)
        return True
    except HfHubHTTPError:
        return False


def resolve_model_id() -> str:
    if model_exists(PRIMARY_MODEL_ID):
        return PRIMARY_MODEL_ID
    if model_exists(FALLBACK_MODEL_ID):
        return FALLBACK_MODEL_ID
    raise RuntimeError(
        f"Neither tokenizer source model is available: {PRIMARY_MODEL_ID}, {FALLBACK_MODEL_ID}"
    )


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    chosen_model = resolve_model_id()

    snapshot_download(
        repo_id=chosen_model,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
    )

    METADATA_PATH.write_text(
        json.dumps(
            {
                "chosen_model_id": chosen_model,
                "primary_model_id": PRIMARY_MODEL_ID,
                "fallback_model_id": FALLBACK_MODEL_ID,
                "allow_patterns": ALLOW_PATTERNS,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Tokenizer files vendored from {chosen_model} into {TARGET_DIR}")


if __name__ == "__main__":
    main()
