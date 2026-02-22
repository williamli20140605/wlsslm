# wlsslm

Bootstrap for a from-scratch pretraining harness.

This repo started with tokenizer vendoring and local evaluation, and now includes:
- Sharded corpus building (HF datasets streaming → uint32 token shards)
- Minimal Transformer training (MPS-friendly)
- Budget + early-stopping training loop

## Quick Start (current workflow)

- End-to-end usage: `docs/USAGE.md`

## Tokenizer

- Qwen3 tokenizer is vendored into `assets/tokenizer/qwen3/`

## Dependencies (baseline)

- `transformers`
- `huggingface_hub`
- `tokenizers`
- `torch`
- `datasets` (for streaming corpus building)

## Scripts (high level)

- Corpus builder (sharded): `scripts/build_corpus_50m.py`
- Inspect shards: `scripts/shards_inspect.py`
- Training:
  - Budget + early stopping: `scripts/train_until_good.py`
  - Foreground helper: `scripts/run_train_until_good.sh`

