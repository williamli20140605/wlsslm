#!/usr/bin/env python3
"""Build a large token corpus as uint32 token shards (no huge raw txt).

Goal: reach a target token budget (default 50M) using HF datasets streaming.
Mix (by tokens, best-effort round-robin):
- Wikipedia en
- Wikipedia zh
- FineWeb (English web)

Outputs:
- data/shards/<name>_shard0001.bin (uint32 token ids)
- data/shards/<name>_shard0001.meta.json (counts + tokenizer info)
- data/shards/<name>.index.json (overall progress + list of shards)

Notes:
- This is designed for long foreground runs. Use `caffeinate` and `tee`.
- Streaming datasets are not deterministic; resume continues by creating new shards.
- Tokens are stored as: doc tokens + EOS token between docs.

Example:
  source .venv/bin/activate
  HF_TOKEN=... python scripts/build_corpus_50m.py --target-tokens 50000000 --name mix50m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_tokenizer():
    from src.tokenizer import load_local_tokenizer

    return load_local_tokenizer()


def _safe_json_write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def iter_wikipedia(cfg: str) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if t:
            yield t


def iter_fineweb(dataset_id: str, subset: Optional[str]) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    if subset:
        ds = load_dataset(dataset_id, subset, **kwargs)
    else:
        ds = load_dataset(dataset_id, **kwargs)

    for ex in ds:
        # most web datasets use `text`
        t = (ex.get("text") or "").strip()
        if t:
            yield t


@dataclass
class ShardInfo:
    shard_id: int
    bin_path: str
    meta_path: str
    token_count: int
    doc_count: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mix50m")
    ap.add_argument("--target-tokens", type=int, default=50_000_000)
    ap.add_argument("--shard-tokens", type=int, default=5_000_000, help="tokens per shard (approx)")
    ap.add_argument("--min-doc-chars", type=int, default=400)
    ap.add_argument("--max-doc-chars", type=int, default=20_000)
    ap.add_argument("--sleep", type=float, default=0.0)

    ap.add_argument("--wiki-en", default="20231101.en")
    ap.add_argument("--wiki-zh", default="20231101.zh")

    # FineWeb defaults: may vary by availability. Override if needed.
    ap.add_argument("--fineweb-id", default="HuggingFaceFW/fineweb")
    ap.add_argument("--fineweb-subset", default="sample-10BT")

    ap.add_argument("--out-dir", default="data/shards")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("Tokenizer has no eos_token_id")

    # Load/Resume index
    index_path = out_dir / f"{args.name}.index.json"
    index: dict[str, Any]
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        # Allow extending the build by rerunning with a higher target.
        index["target_tokens"] = int(args.target_tokens)
        index["shard_tokens"] = int(args.shard_tokens)
        if "sources" not in index:
            index["sources"] = {}
        index["sources"].update(
            {
                "wiki_en": args.wiki_en,
                "wiki_zh": args.wiki_zh,
                "fineweb_id": args.fineweb_id,
                "fineweb_subset": args.fineweb_subset,
            }
        )
    else:
        index = {
            "name": args.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_tokens": args.target_tokens,
            "shard_tokens": args.shard_tokens,
            "tokenizer": {
                "local": "assets/tokenizer/qwen3",
                "eos_token_id": int(eos),
                "vocab_size": int(getattr(tok, "vocab_size", 0)),
            },
            "sources": {
                "wiki_en": args.wiki_en,
                "wiki_zh": args.wiki_zh,
                "fineweb_id": args.fineweb_id,
                "fineweb_subset": args.fineweb_subset,
            },
            "total_tokens": 0,
            "total_docs": 0,
            "shards": [],
            "updated_at": None,
        }

    total_tokens = int(index.get("total_tokens", 0))
    total_docs = int(index.get("total_docs", 0))

    # Next shard id
    shards_list = index.get("shards") or []
    shard_id = int(shards_list[-1]["shard_id"]) + 1 if shards_list else 1

    print(f"Starting build: name={args.name} target_tokens={args.target_tokens} already={total_tokens} next_shard={shard_id}")
    print("If FineWeb config fails, rerun with --fineweb-id/--fineweb-subset overrides.")

    it_en = iter_wikipedia(args.wiki_en)
    it_zh = iter_wikipedia(args.wiki_zh)

    # FineWeb is optional; if it fails we fall back to wiki-only.
    fineweb_ok = True
    try:
        it_fw = iter_fineweb(args.fineweb_id, args.fineweb_subset)
        # quick test pull
        _ = next(it_fw)
        # re-create iterator after consuming one
        it_fw = iter_fineweb(args.fineweb_id, args.fineweb_subset)
    except Exception as e:
        fineweb_ok = False
        print(f"WARN: FineWeb unavailable ({e}). Falling back to wikipedia-only.")
        it_fw = iter(())

    def clean_text(t: str) -> str:
        t = t.strip()
        if len(t) > args.max_doc_chars:
            t = t[: args.max_doc_chars]
        return t

    def next_doc(source: str) -> Optional[str]:
        try:
            if source == "en":
                return clean_text(next(it_en))
            if source == "zh":
                return clean_text(next(it_zh))
            if source == "fw":
                return clean_text(next(it_fw))
            raise ValueError(source)
        except StopIteration:
            return None

    # token budget mix order: en, zh, fw (approx 25/25/50 by frequency)
    schedule = ["en", "zh", "fw", "fw"] if fineweb_ok else ["en", "zh"]
    sched_i = 0

    while total_tokens < args.target_tokens:
        shard_tokens = 0
        shard_docs = 0

        bin_name = f"{args.name}_shard{shard_id:04d}.bin"
        meta_name = f"{args.name}_shard{shard_id:04d}.meta.json"
        bin_path = out_dir / bin_name
        meta_path = out_dir / meta_name

        # If shard already exists, skip it (resume mode)
        if bin_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            shard_tokens = int(meta.get("token_count", 0))
            shard_docs = int(meta.get("doc_count", 0))
            total_tokens += shard_tokens
            total_docs += shard_docs
            print(f"SKIP existing shard {bin_name} tokens={shard_tokens} docs={shard_docs}")
            shard_id += 1
            continue

        # Write shard incrementally
        with open(bin_path, "wb") as f:
            buf = array("I")

            def flush():
                nonlocal buf
                if not buf:
                    return
                buf.tofile(f)
                buf = array("I")

            last_report = time.time()

            while shard_tokens < args.shard_tokens and total_tokens < args.target_tokens:
                source = schedule[sched_i % len(schedule)]
                sched_i += 1

                doc = next_doc(source)
                if doc is None:
                    # if one source exhausts, continue others
                    continue
                if len(doc) < args.min_doc_chars:
                    continue

                ids = tok.encode(doc, add_special_tokens=False)
                if len(ids) < 64:
                    continue

                # append doc tokens + EOS
                buf.extend(ids)
                buf.append(int(eos))

                n = len(ids) + 1
                shard_tokens += n
                total_tokens += n
                shard_docs += 1
                total_docs += 1

                # flush periodically
                if len(buf) >= 200_000:
                    flush()

                now = time.time()
                if now - last_report > 5:
                    print(
                        f"shard={shard_id:04d} shard_tokens~={shard_tokens} total_tokens~={total_tokens}/{args.target_tokens} docs={total_docs}",
                        flush=True,
                    )
                    last_report = now

                if args.sleep:
                    time.sleep(args.sleep)

            flush()

        shard_meta = {
            "name": args.name,
            "shard_id": shard_id,
            "bin": str(bin_path),
            "token_count": shard_tokens,
            "doc_count": shard_docs,
            "eos_token_id": int(eos),
            "sources": {
                "wiki_en": args.wiki_en,
                "wiki_zh": args.wiki_zh,
                "fineweb_id": args.fineweb_id,
                "fineweb_subset": args.fineweb_subset,
                "fineweb_ok": fineweb_ok,
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _safe_json_write(meta_path, shard_meta)

        shard_info = {
            "shard_id": shard_id,
            "bin": bin_name,
            "meta": meta_name,
            "token_count": shard_tokens,
            "doc_count": shard_docs,
        }
        index["shards"].append(shard_info)
        index["total_tokens"] = int(total_tokens)
        index["total_docs"] = int(total_docs)
        index["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _safe_json_write(index_path, index)

        print(f"WROTE shard {shard_id:04d} tokens={shard_tokens} total={total_tokens}")
        shard_id += 1

    print("DONE")
    print(f"total_tokens={total_tokens}")
    print(f"index={index_path}")


if __name__ == "__main__":
    # Avoid printing tokens/secrets; HF_TOKEN is read from env automatically by huggingface_hub.
    main()
