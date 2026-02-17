from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.data_loader import TokenBatchLoader
from src.model import TransformerConfig, TransformerLM


def train_core(
    *,
    bin_path: str,
    vocab_size: int,
    out_ckpt: str,
    block_size: int = 128,
    batch_size: int = 8,
    n_layer: int = 2,
    n_head: int = 4,
    d_model: int = 256,
    ff_mult: float = 8.0 / 3.0,
    dropout: float = 0.0,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    steps: int = 100,
    seed: int = 1337,
    device: str | None = None,
    log_every: int = 10,
    resume_ckpt: str | None = None,
    save_every: int = 0,
    keep_last_k: int = 0,
    run_name: str = "run",
    save_dir: str = "checkpoints",
) -> dict[str, Any]:
    torch.manual_seed(seed)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Support shard index: if a sibling *.index.json exists, prefer shard mode.
    bin_p = Path(bin_path)
    index_candidate = bin_p.parent / (bin_p.stem.split("_shard")[0] + ".index.json")
    if index_candidate.exists():
        loader = TokenBatchLoader.from_shard_index(
            index_path=index_candidate,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
        )
    else:
        loader = TokenBatchLoader.from_bin(
            bin_path=bin_path,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
        )

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        d_model=d_model,
        ff_mult=ff_mult,
        dropout=dropout,
    )
    model = TransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_step = 0
    losses: list[float] = []

    # Resume (model + optimizer + losses + step) if provided.
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=True)
            opt_state = ckpt.get("optimizer_state")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
            start_step = int(ckpt.get("global_step", 0))
            prev_losses = ckpt.get("losses")
            if isinstance(prev_losses, list):
                losses = [float(x) for x in prev_losses]

    model.train()

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: Path, global_step: int, final: bool = False):
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(cfg),
            "train_config": {
                "bin_path": bin_path,
                "block_size": block_size,
                "batch_size": batch_size,
                "steps": steps,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "device": device,
                "run_name": run_name,
            },
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "global_step": global_step,
            "final": final,
        }
        # Atomic write: write to tmp then rename.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)

        # Auto-prune old step checkpoints for this run.
        if keep_last_k and not final:
            try:
                step_ckpts = sorted(
                    [p for p in save_dir_path.glob(f"{run_name}_step*.pt") if not p.name.endswith('.tmp')],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for old in step_ckpts[keep_last_k:]:
                    old.unlink(missing_ok=True)
            except Exception:
                pass

    for i in range(1, steps + 1):
        global_step = start_step + i

        x, y = loader.get_batch()
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)

        if i % log_every == 0 or i == 1 or i == steps:
            print(f"step={global_step}/{start_step + steps} loss={loss_value:.4f}")

        if save_every and (i % save_every == 0):
            save_checkpoint(save_dir_path / f"{run_name}_step{global_step:05d}.pt", global_step)

        # MPS: best-effort cache release to reduce OOM risk
        if device == "mps" and (i % 50 == 0):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_path, start_step + steps, final=True)

    return {
        "checkpoint_path": str(out_path),
        "device": device,
        "steps": steps,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "global_step": start_step + steps,
    }
