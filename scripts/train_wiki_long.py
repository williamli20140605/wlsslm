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
    bin_path = ROOT / "data" / "train_wiki_hf_mix.bin"
    meta_path = bin_path.with_suffix(".meta.json")
    out_ckpt = ROOT / "checkpoints" / "wiki_long.pt"

    if not bin_path.exists():
        raise FileNotFoundError(f"Missing token bin: {bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    tokenizer = load_local_tokenizer()
    meta = json.loads(meta_path.read_text())
    eos_token_id = int(meta["eos_token_id"])
    vocab_size = max(int(tokenizer.vocab_size), eos_token_id + 1)

    # "Long" (chunked) for this bootstrap corpus.
    # Run in small chunks to reduce OOM risk; save frequent step checkpoints.
    from scripts.ckpt_utils import find_latest_step_ckpt

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = "wiki_long"

    # Safer defaults for 16GB unified memory
    block_size = 256
    batch_size = 2
    n_layer = 4
    n_head = 8
    d_model = 512

    total_steps = 2000
    chunk_steps = 50

    latest_ckpt, latest_step = find_latest_step_ckpt(ckpt_dir, run_name)
    resume = str(latest_ckpt) if latest_ckpt else None

    remaining = total_steps - latest_step
    if remaining <= 0:
        print(f"Nothing to do: already have {run_name} at step {latest_step}")
        return

    print(f"Resuming from step {latest_step} ({'none' if not resume else resume})")

    steps_to_run = min(chunk_steps, remaining)

    result = train_core(
        bin_path=str(bin_path),
        vocab_size=vocab_size,
        out_ckpt=str(out_ckpt),
        block_size=block_size,
        batch_size=batch_size,
        n_layer=n_layer,
        n_head=n_head,
        d_model=d_model,
        steps=steps_to_run,
        learning_rate=3e-4,
        weight_decay=0.01,
        log_every=10,
        resume_ckpt=resume,
        save_every=10,
        run_name=run_name,
        save_dir=str(ckpt_dir),
    )

    print("train_wiki_long complete")
    for key in ["checkpoint_path", "device", "steps", "loss_start", "loss_end"]:
        print(f"{key}={result[key]}")


if __name__ == "__main__":
    main()
