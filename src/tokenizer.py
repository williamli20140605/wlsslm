#!/usr/bin/env python3
"""Load Qwen3 tokenizer from local vendored assets only."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

TOKENIZER_DIR = Path("assets/tokenizer/qwen3")


def load_local_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR) -> PreTrainedTokenizerBase:
    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {tokenizer_dir}. "
            "Run `python3 src/tokenizer_fetch.py` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        use_fast=True,
    )
    return tokenizer


def main() -> None:
    tokenizer = load_local_tokenizer()
    sample = "Hello 你好, tokenizer check."
    token_ids = tokenizer.encode(sample, add_special_tokens=True)
    print(f"Loaded tokenizer from {TOKENIZER_DIR}")
    print(f"Sample length (chars): {len(sample)}")
    print(f"Token count: {len(token_ids)}")


if __name__ == "__main__":
    main()
