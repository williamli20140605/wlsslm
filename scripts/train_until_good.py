#!/usr/bin/env python3
"""Train with both:
- fixed budget: stop at max_global_steps
- early stopping: stop if val_loss doesn't improve for `patience` evals

Designed for WL workflow:
- long jobs in foreground
- frequent step checkpoints + resume

Data:
- train shards: all but last `val_shards`
- val shards: last `val_shards`

Usage:
  source .venv/bin/activate
  python scripts/train_until_good.py \
    --index data/shards/mix50m.index.json \
    --run-name mix50m_es \
    --max-global-steps 20000 \
    --chunk-steps 50 \
    --eval-every-chunks 10 \
    --patience 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ckpt_utils import find_latest_step_ckpt
from src.data_loader import TokenBatchLoader
from src.model import TransformerConfig, TransformerLM
from src.tokenizer import load_local_tokenizer
from src.train_core import train_core


def build_model_from_ckpt(ckpt_path: Path, device: str) -> TransformerLM:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = ckpt["model_config"]
    cfg = TransformerConfig(**cfg_dict)
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def eval_loss(model: TransformerLM, loader: TokenBatchLoader, batches: int = 200) -> float:
    losses = []
    for _ in range(batches):
        x, y = loader.get_batch()
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/shards/mix50m.index.json")
    ap.add_argument("--run-name", default="mix50m_es")
    ap.add_argument("--val-shards", type=int, default=2)

    ap.add_argument("--max-global-steps", type=int, default=20000)
    ap.add_argument("--chunk-steps", type=int, default=50)

    ap.add_argument("--eval-every-chunks", type=int, default=10)
    ap.add_argument("--eval-batches", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3)

    # training config
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    ap.add_argument("--shard-rotate-every-batches", type=int, default=50)
    args = ap.parse_args()

    index_path = ROOT / args.index
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = index["shards"]
    if len(shards) < (args.val_shards + 1):
        raise ValueError("Not enough shards for train/val split")

    shard_bins = [index_path.parent / s["bin"] for s in shards]
    train_bins = shard_bins[: -args.val_shards]
    val_bins = shard_bins[-args.val_shards :]

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Find resume
    latest_ckpt, latest_step = find_latest_step_ckpt(ckpt_dir, args.run_name)
    resume = str(latest_ckpt) if latest_ckpt else None

    save_every = 500
    keep_last_k = 3

    print(f"[train_until_good] run={args.run_name}")
    print(f"[train_until_good] max_global_steps={args.max_global_steps} chunk_steps={args.chunk_steps}")
    print(f"[train_until_good] train_shards={len(train_bins)} val_shards={len(val_bins)}")
    print(f"[train_until_good] resume_step={latest_step} resume={resume or 'none'}")
    print(f"[train_until_good] save_every={save_every} keep_last_k={keep_last_k}")

    # tokenizer/vocab
    tokenizer = load_local_tokenizer()
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = max(int(tokenizer.vocab_size), eos_token_id + 1)

    # device selection mirrors train_core
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    best_val = math.inf
    best_step = 0
    bad_evals = 0

    # We'll run chunk by chunk until one of stop conditions triggers.
    chunk_i = 0
    while True:
        if latest_step >= args.max_global_steps:
            print(f"[train_until_good] STOP budget reached: step={latest_step}")
            break

        steps_to_run = min(args.chunk_steps, args.max_global_steps - latest_step)

        # Train one chunk. We point bin_path at the first train shard;
        # train_core will detect index json if present in the same dir only for that shard.
        # To ensure train uses ONLY train shards, we pass a synthetic loader by writing a tiny temp index.
        # Simpler: use first train shard path but rely on loader rotation among full index would include val.
        # So here we avoid train_core's auto-index and instead train on a concatenated list via loader path list.
        # Practical compromise: train on all shards (including val) is not acceptable.
        # Therefore, we create a temporary index file listing only train shards.
        tmp_index = index_path.parent / f".{args.run_name}.train.index.json"
        tmp_payload = dict(index)
        tmp_payload["shards"] = [{"bin": p.name, "token_count": 0} for p in train_bins]
        tmp_index.write_text(json.dumps(tmp_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Use first train shard but in a dir that has our temp index name pattern.
        # train_core detects <stem>.index.json; easiest is to pass bin_path with stem matching tmp index.
        # We therefore symlink/copy is overkill; instead we bypass train_core auto-index by calling
        # train_core with a bin that does NOT trigger index detection, and we monkeypatch by temporarily
        # placing a matching index file.
        # We'll place: <bin_stem_prefix>.index.json next to the bin.
        first_train = train_bins[0]
        index_candidate = first_train.parent / (first_train.stem.split("_shard")[0] + ".index.json")
        backup = None
        if index_candidate.exists():
            backup = index_candidate.read_bytes()
        index_candidate.write_text(tmp_index.read_text(encoding="utf-8"), encoding="utf-8")

        try:
            result = train_core(
                bin_path=str(first_train),
                vocab_size=vocab_size,
                out_ckpt=str(ckpt_dir / f"{args.run_name}.pt"),
                block_size=args.block_size,
                batch_size=args.batch_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                d_model=args.d_model,
                steps=steps_to_run,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                log_every=10,
                resume_ckpt=resume,
                save_every=save_every,
                keep_last_k=keep_last_k,
                run_name=args.run_name,
                save_dir=str(ckpt_dir),
            )
        finally:
            # restore index candidate
            if backup is None:
                try:
                    index_candidate.unlink()
                except Exception:
                    pass
            else:
                index_candidate.write_bytes(backup)
            try:
                tmp_index.unlink()
            except Exception:
                pass

        latest_step = int(result["global_step"])
        latest_ckpt, _ = find_latest_step_ckpt(ckpt_dir, args.run_name)
        resume = str(latest_ckpt) if latest_ckpt else str(ckpt_dir / f"{args.run_name}.pt")

        chunk_i += 1

        do_eval = (chunk_i % int(args.eval_every_chunks) == 0) or (latest_step >= args.max_global_steps)
        if do_eval:
            # Val loader: deterministic rotation for coverage
            val_loader = TokenBatchLoader.from_shard_paths(
                [str(p) for p in val_bins],
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=device,
                shard_rotate_every_batches=args.shard_rotate_every_batches,
            )
            model = build_model_from_ckpt(Path(resume), device)
            v = eval_loss(model, val_loader, batches=int(args.eval_batches))
            print(f"[train_until_good] EVAL step={latest_step} val_loss={v:.4f}")

            if v < best_val - 1e-4:
                best_val = v
                best_step = latest_step
                bad_evals = 0
                print(f"[train_until_good] new best val_loss={best_val:.4f} at step={best_step}")
            else:
                bad_evals += 1
                print(f"[train_until_good] no improvement (bad_evals={bad_evals}/{args.patience})")
                if bad_evals >= int(args.patience):
                    print(f"[train_until_good] STOP early stopping: best_step={best_step} best_val={best_val:.4f}")
                    break

    print("[train_until_good] DONE")


if __name__ == "__main__":
    main()
