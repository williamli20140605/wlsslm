#!/usr/bin/env python3
"""Sample/generate text from a saved wlsslm checkpoint.

This loads:
- local Qwen3 tokenizer (assets/tokenizer/qwen3)
- checkpoint produced by src/train_core.py
- TransformerLM from src/model.py

Example:
  source .venv/bin/activate
  python scripts/sample_from_ckpt.py \
    --ckpt checkpoints/mix50m_epoch1.pt \
    --prompt "Hi" \
    --max-new 200 \
    --temperature 0.9 \
    --top-k 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import TransformerConfig, TransformerLM
from src.tokenizer import load_local_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to checkpoint .pt")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    tok = load_local_tokenizer()

    # Choose device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = TransformerConfig(**ckpt["model_config"])
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Encode prompt
    ids = tok.encode(args.prompt, add_special_tokens=False)
    if not ids:
        raise SystemExit("Prompt encoded to empty token list")

    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        x,
        max_new_tokens=int(args.max_new),
        temperature=float(args.temperature),
        top_k=int(args.top_k) if args.top_k > 0 else None,
    )

    out_ids = out[0].tolist()
    text = tok.decode(out_ids)

    print("=== prompt ===")
    print(args.prompt)
    print("=== completion ===")
    # print full decoded output (includes prompt)
    print(text)


if __name__ == "__main__":
    main()
