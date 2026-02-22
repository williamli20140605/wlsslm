#!/usr/bin/env python3
"""Build an English-only Wikipedia token corpus as uint32 token shards.

Purpose:
- Produce a *cleaner, article-like* continuation-pretrain dataset than general web.
- Keep code identical across Mac mini + Windows (WSL); data will differ.

Outputs (default):
- data/shards/<name>_shard0001.bin (uint32 token ids)
- data/shards/<name>_shard0001.meta.json
- data/shards/<name>.index.json

Notes:
- Uses HuggingFace datasets streaming.
- Optionally rejects any doc containing CJK characters (for "no Chinese" preference).

Example:
  source .venv/bin/activate
  python scripts/build_wiki_en_only.py \
    --name wiki_en_only_50m \
    --target-tokens 50000000 \
    --shard-tokens 5000000 \
    --reject-cjk

Then train with:
  python scripts/train_until_good.py \
    --index data/shards/wiki_en_only_50m.index.json \
    --run-name wiki_en_cont_v1 \
    --init-ckpt checkpoints/fineweb1b_es.pt \
    --val-shards 2 \
    --block-size 256 --batch-size 8 --n-layer 6 --n-head 4 --d-model 256
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Iterator

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


# Broad CJK detection: Han + Extensions + Kana + Hangul (covers "no Chinese" hardline)
_CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7AF]"
)


def contains_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wiki_en_only_50m")
    ap.add_argument("--target-tokens", type=int, default=50_000_000)
    ap.add_argument("--shard-tokens", type=int, default=5_000_000, help="tokens per shard (approx)")
    ap.add_argument("--min-doc-chars", type=int, default=400)
    ap.add_argument("--max-doc-chars", type=int, default=20_000)
    ap.add_argument("--sleep", type=float, default=0.0)

    ap.add_argument("--wiki-en", default="20231101.en")
    ap.add_argument("--reject-cjk", action="store_true", help="Drop any doc containing CJK chars")

    ap.add_argument("--out-dir", default="data/shards")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("Tokenizer has no eos_token_id")

    vocab_size = max(int(getattr(tok, "vocab_size", 0) or 0), int(eos) + 1)

    index_path = out_dir / f"{args.name}.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["target_tokens"] = int(args.target_tokens)
        index["shard_tokens"] = int(args.shard_tokens)
        index.setdefault("sources", {})
        index["sources"].update({"wiki_en": args.wiki_en, "reject_cjk": bool(args.reject_cjk)})
        # keep tokenizer info but ensure it's sane
        index.setdefault("tokenizer", {})
        index["tokenizer"].update({"eos_token_id": int(eos), "vocab_size": int(vocab_size)})
    else:
        index = {
            "name": args.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_tokens": int(args.target_tokens),
            "shard_tokens": int(args.shard_tokens),
            "tokenizer": {
                "local": "assets/tokenizer/qwen3",
                "eos_token_id": int(eos),
                "vocab_size": int(vocab_size),
            },
            "sources": {
                "wiki_en": args.wiki_en,
                "reject_cjk": bool(args.reject_cjk),
            },
            "total_tokens": 0,
            "total_docs": 0,
            "dropped_cjk_docs": 0,
            "shards": [],
            "updated_at": None,
        }

    total_tokens = int(index.get("total_tokens", 0))
    total_docs = int(index.get("total_docs", 0))
    dropped_cjk = int(index.get("dropped_cjk_docs", 0))

    shards_list = index.get("shards") or []
    shard_id = int(shards_list[-1]["shard_id"]) + 1 if shards_list else 1

    it_en = iter_wikipedia(args.wiki_en)

    def clean_text(t: str) -> str:
        t = t.strip()
        if len(t) > args.max_doc_chars:
            t = t[: args.max_doc_chars]
        return t

    print(
        f"Starting wiki build: name={args.name} target_tokens={args.target_tokens} already={total_tokens} next_shard={shard_id} reject_cjk={bool(args.reject_cjk)}"
    )

    while total_tokens < args.target_tokens:
        shard_tokens = 0
        shard_docs = 0

        bin_name = f"{args.name}_shard{shard_id:04d}.bin"
        meta_name = f"{args.name}_shard{shard_id:04d}.meta.json"
        bin_path = out_dir / bin_name
        meta_path = out_dir / meta_name

        if bin_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            shard_tokens = int(meta.get("token_count", 0))
            shard_docs = int(meta.get("doc_count", 0))
            total_tokens += shard_tokens
            total_docs += shard_docs
            print(f"SKIP existing shard {bin_name} tokens={shard_tokens} docs={shard_docs}")
            shard_id += 1
            continue

        with open(bin_path, "wb") as f:
            buf = array("I")

            def flush() -> None:
                nonlocal buf
                if not buf:
                    return
                buf.tofile(f)
                buf = array("I")

            last_report = time.time()

            while shard_tokens < args.shard_tokens and total_tokens < args.target_tokens:
                try:
                    doc = clean_text(next(it_en))
                except StopIteration:
                    break

                if len(doc) < args.min_doc_chars:
                    continue

                if args.reject_cjk and contains_cjk(doc):
                    dropped_cjk += 1
                    continue

                ids = tok.encode(doc, add_special_tokens=False)
                if len(ids) < 64:
                    continue

                buf.extend(ids)
                buf.append(int(eos))

                n = len(ids) + 1
                shard_tokens += n
                total_tokens += n
                shard_docs += 1
                total_docs += 1

                if len(buf) >= 200_000:
                    flush()

                now = time.time()
                if now - last_report > 5:
                    print(
                        f"shard={shard_id:04d} shard_tokens~={shard_tokens} total_tokens~={total_tokens}/{args.target_tokens} docs={total_docs} dropped_cjk={dropped_cjk}",
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
            "vocab_size": int(vocab_size),
            "sources": {"wiki_en": args.wiki_en, "reject_cjk": bool(args.reject_cjk)},
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
        index["dropped_cjk_docs"] = int(dropped_cjk)
        index["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _safe_json_write(index_path, index)

        print(f"WROTE shard {shard_id:04d} tokens={shard_tokens} total={total_tokens} dropped_cjk={dropped_cjk}")
        shard_id += 1

    print("DONE")
    print(f"total_tokens={total_tokens}")
    print(f"total_docs={total_docs}")
    print(f"dropped_cjk_docs={dropped_cjk}")
    print(f"index={index_path}")


if __name__ == "__main__":
    main()
