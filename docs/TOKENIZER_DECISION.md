# Tokenizer Decision (Phase 1 Part 1)

## Goal

Vendor a Qwen3 tokenizer locally for offline loading in this repo.

## Source Strategy

1. Primary source model id: `Qwen/Qwen3-0.6B`
2. Fallback source model id: `Qwen/Qwen3-1.7B`

Current chosen source: `Qwen/Qwen3-0.6B` (resolved by tokenizer_fetch.py; see assets/tokenizer/qwen3/source_model.json).

## Notes

- Only tokenizer-related artifacts are downloaded.
- Files are stored in `assets/tokenizer/qwen3/`.
- Runtime loading in `src/tokenizer.py` uses local files only (no network).
