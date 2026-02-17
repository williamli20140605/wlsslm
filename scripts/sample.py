#!/usr/bin/env python3
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample text from a trained tiny checkpoint")
    p.add_argument("--ckpt", type=str, default="checkpoints/tiny.pt")
    p.add_argument("--prompt", type=str, default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def choose_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt_path = ROOT / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = TransformerConfig(**checkpoint["model_config"])

    model = TransformerLM(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    tokenizer = load_local_tokenizer()
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError("Prompt produced empty token list. Use a non-empty prompt.")

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    out_ids = out[0].tolist()
    text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(f"device={device}")
    print(f"prompt={args.prompt!r}")
    print("sample_text:")
    print(text)


if __name__ == "__main__":
    main()
