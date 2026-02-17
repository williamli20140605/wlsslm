#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", help="path to shard index json")
    args = ap.parse_args()

    p = Path(args.index)
    j = json.loads(p.read_text(encoding="utf-8"))
    shards = j.get("shards") or []

    print("name", j.get("name"))
    print("target_tokens", j.get("target_tokens"))
    print("total_tokens", j.get("total_tokens"))
    print("total_docs", j.get("total_docs"))
    print("shards", len(shards))

    if shards:
        print("first", shards[0])
        print("last", shards[-1])


if __name__ == "__main__":
    main()
