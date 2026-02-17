#!/usr/bin/env python3
"""Fetch a small mixed zh/en corpus from Wikipedia using MediaWiki API.

- No extra dependencies (stdlib only).
- Writes blankline-separated docs suitable for scripts/pack_docs.py --split blanklines.

Usage:
  python3 scripts/fetch_wiki_mix.py --out data/raw/wiki_mix.txt --zh 120 --en 120

Notes:
- This is for bootstrapping (100k+ tokens), not for full-scale pretraining.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def http_get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "wlsslm/0.0 (fetch_wiki_mix)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data.decode("utf-8"))


def fetch_random_extract(lang: str, max_chars: int = 6000) -> str:
    # Random page
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "format": "json",
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 1,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "exlimit": 1,
        "exintro": 0,
        "exchars": max_chars,
        "redirects": 1,
    }
    url = api + "?" + urllib.parse.urlencode(params)
    j = http_get_json(url)
    pages = (((j.get("query") or {}).get("pages")) or {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    title = (page.get("title") or "").strip()
    extract = (page.get("extract") or "").strip()
    if not extract:
        return ""

    # Basic cleanup: remove excessive blank lines
    lines = [ln.rstrip() for ln in extract.splitlines()]
    cleaned = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
            continue
        blank_run = 0
        cleaned.append(ln)
    text = "\n".join(cleaned).strip()

    # Prefix title for context
    if title and title not in text[:200]:
        text = f"Title: {title}\n\n" + text
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/wiki_mix.txt")
    ap.add_argument("--zh", type=int, default=120)
    ap.add_argument("--en", type=int, default=120)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--max-chars", type=int, default=6000)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = [("zh", args.zh), ("en", args.en)]
    docs: list[str] = []

    for lang, n in total:
        ok = 0
        attempts = 0
        while ok < n and attempts < n * 12:
            attempts += 1
            try:
                text = fetch_random_extract(lang, max_chars=args.max_chars)
            except Exception as e:
                # backoff on network flakiness
                print(f"[{lang}] fetch error: {e}", file=sys.stderr)
                time.sleep(max(args.sleep, 1.0))
                continue
            if len(text) < 300:
                time.sleep(args.sleep)
                continue
            docs.append(text)
            ok += 1
            if ok % 10 == 0:
                print(f"[{lang}] {ok}/{n}")
            time.sleep(args.sleep)

        if ok < n:
            print(f"[{lang}] warning: only got {ok}/{n}", file=sys.stderr)

    # Shuffle to mix languages
    random.shuffle(docs)

    # blankline-separated docs
    out_path.write_text("\n\n\n".join(docs) + "\n", encoding="utf-8")
    print(f"Wrote {len(docs)} docs to {out_path} (bytes={out_path.stat().st_size})")


if __name__ == "__main__":
    main()
