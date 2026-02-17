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
from scripts.ckpt_utils import find_latest_step_ckpt


def main() -> None:
    index_path = ROOT / "data" / "shards" / "mix50m.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing shard index: {index_path}")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_name = "mix50m_long"
    out_ckpt = ckpt_dir / f"{run_name}.pt"

    tokenizer = load_local_tokenizer()
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = max(int(tokenizer.vocab_size), eos_token_id + 1)

    j = json.loads(index_path.read_text(encoding="utf-8"))
    first_bin = j["shards"][0]["bin"]
    first_bin_path = (index_path.parent / first_bin)

    latest_ckpt, latest_step = find_latest_step_ckpt(ckpt_dir, run_name)
    resume = str(latest_ckpt) if latest_ckpt else None

    # Forever mode (A): run in small chunks. You can just re-run this command.
    chunk_steps = 50

    print(f"Resuming from step {latest_step} ({'none' if not resume else resume})")

    # Safer defaults for 16GB unified memory
    result = train_core(
        bin_path=str(first_bin_path),
        vocab_size=vocab_size,
        out_ckpt=str(out_ckpt),
        block_size=256,
        batch_size=2,
        n_layer=4,
        n_head=8,
        d_model=512,
        steps=chunk_steps,
        learning_rate=3e-4,
        weight_decay=0.01,
        log_every=10,
        resume_ckpt=resume,
        save_every=500,
        keep_last_k=3,
        run_name=run_name,
        save_dir=str(ckpt_dir),
    )

    print("train_mix50m_long chunk complete")
    for key in ["checkpoint_path", "device", "global_step", "loss_start", "loss_end"]:
        print(f"{key}={result.get(key)}")


if __name__ == "__main__":
    main()
