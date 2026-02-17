# wlsslm Usage

This doc explains how to run wlsslm end-to-end:
- build/extend the token corpus (sharded)
- train (budget + early stop)
- use the trained model (final checkpoint)

## 0) Setup

```bash
cd /Users/william/Documents/wlsslm
source .venv/bin/activate
```

## 1) Corpus (tokens)

Corpus is stored as uint32 token shards:
- Index: `data/shards/mix50m.index.json`
- Shards: `data/shards/mix50m_shard*.bin`

### HF token (recommended)

Create a HuggingFace token with **Read** scope and store it in `.env` (do not commit):

```bash
cd /Users/william/Documents/wlsslm
cat > .env <<'EOF'
HF_TOKEN=hf_xxx
EOF

echo ".env" >> .gitignore
```

Load it into your current shell:

```bash
set -a; source .env; set +a
```

### Build / extend shards

```bash
caffeinate -dimsu python scripts/build_corpus_50m.py \
  --name mix50m \
  --target-tokens 100000000 \
  --shard-tokens 5000000
```

## 2) Train (run-to-good)

Recommended: run the budget + early-stopping loop in the foreground.

```bash
caffeinate -dimsu bash scripts/run_train_until_good.sh
```

Key defaults (see script):
- `MAX_GLOBAL_STEPS=195000` (~1 epoch on 100M tokens @ 512 tokens/step)
- `EVAL_EVERY_CHUNKS=100` (5000 steps per eval)
- `PATIENCE=3`

Outputs:
- Final checkpoint: `checkpoints/<RUN_NAME>.pt`
- Step checkpoints: `checkpoints/<RUN_NAME>_stepXXXXX.pt` (saved every 500 steps, keep last 3)

## 3) Use the trained model (final checkpoint)

### Inspect checkpoint metadata

```bash
python - <<'PY'
import torch
p='checkpoints/mix50m_epoch1.pt'  # change RUN_NAME
ckpt=torch.load(p, map_location='cpu')
print('global_step', ckpt.get('global_step'))
print('final', ckpt.get('final'))
print('model_config', ckpt.get('model_config'))
PY
```

### Sample / generate text

```bash
python scripts/sample_from_ckpt.py \
  --ckpt checkpoints/mix50m_epoch1.pt \
  --prompt "Hi" \
  --max-new 200 \
  --temperature 0.9 \
  --top-k 50
```

Notes:
- This is a continuation model (pretraining). Assistant-style chat requires an SFT phase.
