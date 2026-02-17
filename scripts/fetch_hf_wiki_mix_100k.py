#!/usr/bin/env python3
"""Build a mixed zh/en text corpus (>= target tokens) from HuggingFace Wikipedia dataset.

Approach:
- Stream samples from Wikipedia (en + zh)
- For each doc, estimate tokens using local Qwen3 tokenizer
- Stop when we reach target token budget (default 120k tokens)
- Write blankline-separated docs to an output .txt

Then you can pack with:
  source .venv/bin/activate
  python scripts/pack_docs.py data/raw/wiki_hf_mix.txt --split blanklines --out data/train_wiki_hf_mix.bin

Notes:
- Uses HF datasets streaming to avoid large local downloads.
- If streaming is unavailable (older datasets version), remove streaming=True and keep small limits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_tokenizer():
    # local import without requiring package layout
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from tokenizer import load_local_tokenizer  # noqa

    return load_local_tokenizer()


def iter_wiki(lang_cfg: str):
    from datasets import load_dataset

    # wikimedia/wikipedia provides configs like 20231101.en / 20231101.zh
    ds = load_dataset("wikimedia/wikipedia", lang_cfg, split="train", streaming=True)
    for ex in ds:
        # fields vary; commonly 'text' exists
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/wiki_hf_mix.txt")
    ap.add_argument("--target-tokens", type=int, default=120_000)
    ap.add_argument("--max-docs-per-lang", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--en", default="20231101.en")
    ap.add_argument("--zh", default="20231101.zh")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("Tokenizer has no eos_token_id")

    docs: list[str] = []
    total_tokens = 0

    def add_doc(t: str):
        nonlocal total_tokens
        # keep docs not too huge
        t = t.strip()
        if len(t) < 200:
            return
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) < 50:
            return
        total_tokens += len(ids) + 1  # + eos
        docs.append(t)

    # Round-robin between en and zh to approximate 50/50 tokens.
    it_en = iter_wiki(args.en)
    it_zh = iter_wiki(args.zh)

    en_docs = 0
    zh_docs = 0

    while total_tokens < args.target_tokens:
        if en_docs < args.max_docs_per_lang:
            try:
                add_doc(next(it_en))
                en_docs += 1
            except StopIteration:
                en_docs = args.max_docs_per_lang
        if total_tokens >= args.target_tokens:
            break
        if zh_docs < args.max_docs_per_lang:
            try:
                add_doc(next(it_zh))
                zh_docs += 1
            except StopIteration:
                zh_docs = args.max_docs_per_lang

        if (en_docs + zh_docs) % 50 == 0:
            print(f"docs={len(docs)} tokens~={total_tokens} (en_docs={en_docs} zh_docs={zh_docs})")

        if en_docs >= args.max_docs_per_lang and zh_docs >= args.max_docs_per_lang:
            break

    # Shuffle docs lightly
    random.shuffle(docs)

    out_path.write_text("\n\n\n".join(docs) + "\n", encoding="utf-8")

    meta = {
        "out": str(out_path),
        "target_tokens": args.target_tokens,
        "approx_tokens": total_tokens,
        "docs": len(docs),
        "en_cfg": args.en,
        "zh_cfg": args.zh,
        "tokenizer": "assets/tokenizer/qwen3",
        "eos_token_id": eos,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("DONE")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
