#!/usr/bin/env python3
"""FineWeb-only corpus builder (uint32 token shards).

- No Wikipedia.
- Default rejects CJK chars.

Outputs:
- data/shards/<name>_shard0001.bin (uint32 token ids)
- data/shards/<name>_shard0001.meta.json
- data/shards/<name>.index.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Basic CJK ranges (Han/Hiragana/Katakana/Hangul)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")


def load_tokenizer():
    from src.tokenizer import load_local_tokenizer

    return load_local_tokenizer()


def _safe_json_write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def iter_fineweb(dataset_id: str, subset: Optional[str]) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    ds = load_dataset(dataset_id, subset, **kwargs) if subset else load_dataset(dataset_id, **kwargs)

    for ex in ds:
        t = (ex.get("text") or "").strip()
        if t:
            yield t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fineweb_en_only")
    ap.add_argument("--target-tokens", type=int, default=50_000_000)
    ap.add_argument("--shard-tokens", type=int, default=5_000_000)
    ap.add_argument("--min-doc-chars", type=int, default=400)
    ap.add_argument("--max-doc-chars", type=int, default=20_000)
    ap.add_argument("--sleep", type=float, default=0.0)

    ap.add_argument("--allow-cjk", action="store_true", help="Allow CJK characters (default: reject).")

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

    index_path = out_dir / f"{args.name}.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["target_tokens"] = int(args.target_tokens)
        index["shard_tokens"] = int(args.shard_tokens)
        index.setdefault("sources", {})
        index["sources"].update(
            {
                "fineweb_id": args.fineweb_id,
                "fineweb_subset": args.fineweb_subset,
                "allow_cjk": bool(args.allow_cjk),
            }
        )
    else:
        index = {
            "name": args.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_tokens": int(args.target_tokens),
            "shard_tokens": int(args.shard_tokens),
            "tokenizer": {
                "local": "assets/tokenizer/qwen3",
                "eos_token_id": int(eos),
                "vocab_size": int(getattr(tok, "vocab_size", 0)),
            },
            "sources": {
                "fineweb_id": args.fineweb_id,
                "fineweb_subset": args.fineweb_subset,
                "allow_cjk": bool(args.allow_cjk),
            },
            "total_tokens": 0,
            "total_docs": 0,
            "shards": [],
            "updated_at": None,
        }

    total_tokens = int(index.get("total_tokens", 0))
    total_docs = int(index.get("total_docs", 0))

    shards_list = index.get("shards") or []
    shard_id = int(shards_list[-1]["shard_id"]) + 1 if shards_list else 1

    print(
        f"Starting FineWeb-only build: name={args.name} target_tokens={args.target_tokens} already={total_tokens} next_shard={shard_id}"
    )
    print("If FineWeb config fails, rerun with --fineweb-id/--fineweb-subset overrides.")

    try:
        it_fw = iter_fineweb(args.fineweb_id, args.fineweb_subset)
        _ = next(it_fw)  # quick test pull
        it_fw = iter_fineweb(args.fineweb_id, args.fineweb_subset)
    except Exception as e:
        raise SystemExit(f"ERROR: FineWeb unavailable ({e}).")

    def clean_text(t: str) -> str:
        t = t.strip()
        if len(t) > args.max_doc_chars:
            t = t[: args.max_doc_chars]
        if (not args.allow_cjk) and _CJK_RE.search(t):
            return ""
        return t

    def next_doc() -> Optional[str]:
        try:
            return clean_text(next(it_fw))
        except StopIteration:
            return None

    while total_tokens < args.target_tokens:
        shard_tokens = 0
        shard_docs = 0

        bin_name = f"{args.name}_shard{shard_id:04d}.bin"
        meta_name = f"{args.name}_shard{shard_id:04d}.meta.json"
        bin_path = out_dir / bin_name
        meta_path = out_dir / meta_name

        # Resume mode: skip existing shards
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

            def flush():
                nonlocal buf
                if buf:
                    buf.tofile(f)
                    buf = array("I")

            last_report = time.time()

            while shard_tokens < args.shard_tokens and total_tokens < args.target_tokens:
                doc = next_doc()
                if doc is None:
                    raise SystemExit("FineWeb stream ended unexpectedly.")
                if len(doc) < args.min_doc_chars:
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
                "fineweb_id": args.fineweb_id,
                "fineweb_subset": args.fineweb_subset,
                "allow_cjk": bool(args.allow_cjk),
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _safe_json_write(meta_path, shard_meta)

        index["shards"].append(
            {
                "shard_id": shard_id,
                "bin": bin_name,
                "meta": meta_name,
                "token_count": shard_tokens,
                "doc_count": shard_docs,
            }
        )
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
    main()
