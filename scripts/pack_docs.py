#!/usr/bin/env python3
"""Pack text docs into a uint32 token .bin with EOS between docs."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tokenizer import load_local_tokenizer  # noqa: E402

SOURCE_MODEL_PATH = REPO_ROOT / "assets/tokenizer/qwen3/source_model.json"


def split_docs(text: str, mode: str) -> list[str]:
    if mode == "lines":
        return [line for line in text.splitlines() if line.strip()]

    docs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
            continue
        if current:
            docs.append("\n".join(current))
            current = []

    if current:
        docs.append("\n".join(current))
    return docs


def tokenize_docs(docs: Iterable[str], eos_token_id: int) -> list[int]:
    tokenizer = load_local_tokenizer()
    token_ids: list[int] = []
    for doc in docs:
        token_ids.extend(tokenizer.encode(doc, add_special_tokens=False))
        token_ids.append(eos_token_id)
    return token_ids


def write_u32_bin(out_path: Path, token_ids: list[int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        if token_ids:
            f.write(struct.pack(f"<{len(token_ids)}I", *token_ids))


def write_meta(out_path: Path, token_ids: list[int], eos_token_id: int) -> Path:
    meta_path = out_path.with_suffix(".meta.json")
    tokenizer_source = json.loads(SOURCE_MODEL_PATH.read_text(encoding="utf-8"))
    payload = {
        "dtype": "uint32",
        "token_count": len(token_ids),
        "eos_token_id": eos_token_id,
        "tokenizer_source": tokenizer_source,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input text file")
    parser.add_argument(
        "--split",
        choices=("lines", "blanklines"),
        default="lines",
        help="Document split mode (default: lines)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/sample_mixed.bin"),
        help="Output token .bin path (default: data/sample_mixed.bin)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")

    tokenizer = load_local_tokenizer()
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    docs = split_docs(text, args.split)
    token_ids = tokenize_docs(docs, eos_token_id)

    write_u32_bin(args.out, token_ids)
    meta_path = write_meta(args.out, token_ids, eos_token_id)

    print(f"input={args.input}")
    print(f"split={args.split}")
    print(f"docs={len(docs)}")
    print(f"out_bin={args.out}")
    print(f"out_meta={meta_path}")
    print(f"token_count={len(token_ids)}")
    print(f"eos_token_id={eos_token_id}")


if __name__ == "__main__":
    main()
