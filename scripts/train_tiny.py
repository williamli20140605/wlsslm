#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer import load_local_tokenizer
from src.train_core import train_core


def main() -> None:
    bin_path = ROOT / "data" / "sample_mixed.bin"
    meta_path = bin_path.with_suffix(".meta.json")
    out_ckpt = ROOT / "checkpoints" / "tiny.pt"

    if not bin_path.exists():
        raise FileNotFoundError(f"Missing token bin: {bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    tokenizer = load_local_tokenizer()
    meta = json.loads(meta_path.read_text())
    eos_token_id = int(meta["eos_token_id"])
    vocab_size = max(int(tokenizer.vocab_size), eos_token_id + 1)

    result = train_core(
        bin_path=str(bin_path),
        vocab_size=vocab_size,
        out_ckpt=str(out_ckpt),
        block_size=128,
        batch_size=8,
        n_layer=2,
        n_head=4,
        d_model=256,
        steps=100,
        learning_rate=3e-4,
        weight_decay=0.0,
    )

    print("train_tiny complete")
    for key in ["checkpoint_path", "device", "steps", "loss_start", "loss_end"]:
        print(f"{key}={result[key]}")


if __name__ == "__main__":
    main()
