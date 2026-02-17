#!/usr/bin/env bash
set -euo pipefail

# Foreground infinite training loop (WL rule: long jobs stay in foreground).
# Runs one 50-step chunk each iteration (python scripts/train_mix50m_long.py),
# then sleeps briefly to give MPS + system a breather.
# Stop anytime with Ctrl+C.

SLEEP_SECS="${SLEEP_SECS:-5}"

cd "$(dirname "$0")/.."

mkdir -p logs

echo "[run_train_forever] started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[run_train_forever] sleep=${SLEEP_SECS}s between chunks"

i=0
while true; do
  i=$((i+1))
  echo "[run_train_forever] chunk #$i at $(date '+%Y-%m-%d %H:%M:%S')"
  # Append logs; training script handles resume + checkpoints.
  python scripts/train_mix50m_long.py 2>&1 | tee -a logs/train_mix50m.log
  echo "[run_train_forever] chunk #$i done; sleeping ${SLEEP_SECS}s"
  sleep "$SLEEP_SECS"
done
