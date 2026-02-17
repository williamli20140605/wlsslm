#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_STEP_RE = re.compile(r"^(?P<name>.+)_step(?P<step>\d{5})\.pt$")


def find_latest_step_ckpt(ckpt_dir: Path, run_name: str) -> tuple[Optional[Path], int]:
    """Return (path, step) for the latest *loadable* step checkpoint.

    Some checkpoints can be corrupted if the disk was full mid-write.
    We iterate from newest step to oldest and try a lightweight torch.load.
    """
    if not ckpt_dir.exists():
        return (None, 0)

    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob(f"{run_name}_step*.pt"):
        m = _STEP_RE.match(p.name)
        if not m:
            continue
        step = int(m.group("step"))
        candidates.append((step, p))

    if not candidates:
        return (None, 0)

    candidates.sort(key=lambda x: x[0], reverse=True)

    try:
        import torch
    except Exception:
        # If torch isn't available, fall back to purely lexical choice.
        step, p = candidates[0]
        return (p, step)

    for step, p in candidates:
        try:
            ckpt = torch.load(str(p), map_location="cpu")
            # basic sanity
            if isinstance(ckpt, dict) and int(ckpt.get("global_step", step)) >= step:
                return (p, step)
        except Exception:
            continue

    return (None, 0)
