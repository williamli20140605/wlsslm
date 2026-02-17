#!/usr/bin/env python3
"""Inspect a packed uint32 token .bin and decode a preview."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tokenizer import load_local_tokenizer  # noqa: E402


def read_u32_bin(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"Invalid uint32 binary size: {len(data)} bytes")
    return [value for (value,) in struct.iter_unpack("<I", data)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_path", type=Path, help="Path to .bin file")
    parser.add_argument("--n", type=int, default=24, help="Show first N token ids")
    parser.add_argument("--k", type=int, default=64, help="Decode first K tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.bin_path.with_suffix(".meta.json")

    tokens = read_u32_bin(args.bin_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    tokenizer = load_local_tokenizer()
    first_n = tokens[: args.n]
    preview_tokens = tokens[: args.k]
    preview_text = tokenizer.decode(preview_tokens, skip_special_tokens=False)

    print(f"bin={args.bin_path}")
    print(f"meta={meta_path}")
    print(f"meta_token_count={meta.get('token_count')}")
    print(f"token_count_read={len(tokens)}")
    print(f"first_{args.n}_token_ids={first_n}")
    print(f"decoded_first_{args.k}_tokens={preview_text}")


if __name__ == "__main__":
    main()
