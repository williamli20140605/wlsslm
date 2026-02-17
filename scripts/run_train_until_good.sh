#!/usr/bin/env bash
set -euo pipefail

# Foreground long training with BOTH:
# - budget stop (max_global_steps)
# - early stopping on val_loss
#
# Stop anytime with Ctrl+C.

cd "$(dirname "$0")/.."

source .venv/bin/activate
mkdir -p logs

# Default knobs (override via env vars)
RUN_NAME="${RUN_NAME:-mix50m_es}"
MAX_GLOBAL_STEPS="${MAX_GLOBAL_STEPS:-195000}"  # ~1 epoch on 100M tokens @ 512 tokens/step
CHUNK_STEPS="${CHUNK_STEPS:-50}"
EVAL_EVERY_CHUNKS="${EVAL_EVERY_CHUNKS:-100}"     # 100 chunks * 50 steps = 5000 steps per eval
PATIENCE="${PATIENCE:-3}"
VAL_SHARDS="${VAL_SHARDS:-2}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"

echo "[run_train_until_good] run=$RUN_NAME max_global_steps=$MAX_GLOBAL_STEPS chunk_steps=$CHUNK_STEPS eval_every_chunks=$EVAL_EVERY_CHUNKS patience=$PATIENCE val_shards=$VAL_SHARDS"

PYTHONUNBUFFERED=1 python scripts/train_until_good.py \
  --index data/shards/mix50m.index.json \
  --run-name "$RUN_NAME" \
  --max-global-steps "$MAX_GLOBAL_STEPS" \
  --chunk-steps "$CHUNK_STEPS" \
  --eval-every-chunks "$EVAL_EVERY_CHUNKS" \
  --patience "$PATIENCE" \
  --val-shards "$VAL_SHARDS" \
  --eval-batches "$EVAL_BATCHES" \
  2>&1 | tee -a "logs/train_${RUN_NAME}.log"
