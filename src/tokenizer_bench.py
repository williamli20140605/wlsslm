#!/usr/bin/env python3
"""Benchmark local tokenizer on eval/sample_mixed.txt."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tokenizer import load_local_tokenizer

INPUT_PATH = Path("eval/sample_mixed.txt")
OUTPUT_PATH = Path("docs/tokenizer_bench.json")
ENCODE_REPEATS = 200
DECODE_REPEATS = 200


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark input file: {INPUT_PATH}")

    text = INPUT_PATH.read_text(encoding="utf-8")
    tokenizer = load_local_tokenizer()

    warmup_ids = tokenizer.encode(text, add_special_tokens=True)

    t0 = time.perf_counter()
    for _ in range(ENCODE_REPEATS):
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(DECODE_REPEATS):
        tokenizer.decode(warmup_ids, skip_special_tokens=False)
    t3 = time.perf_counter()

    encode_seconds = t1 - t0
    decode_seconds = t3 - t2
    chars = len(text)
    tokens = len(token_ids)

    result = {
        "input_path": str(INPUT_PATH),
        "chars": chars,
        "tokens": tokens,
        "encode_repeats": ENCODE_REPEATS,
        "decode_repeats": DECODE_REPEATS,
        "encode_seconds_total": encode_seconds,
        "decode_seconds_total": decode_seconds,
        "encode_tokens_per_second": (tokens * ENCODE_REPEATS) / encode_seconds
        if encode_seconds > 0
        else None,
        "decode_tokens_per_second": (tokens * DECODE_REPEATS) / decode_seconds
        if decode_seconds > 0
        else None,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote benchmark results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
