# Progress Log

## 2026-02-16 - Phase 1 Part 1 Plan (First Entry)

Planned files to create:
- `README.md`
- `docs/TOKENIZER_DECISION.md`
- `eval/sample_mixed.txt`
- `src/tokenizer_fetch.py`
- `src/tokenizer.py`
- `src/tokenizer_bench.py`
- `harness/progress.md`
- `harness/features.json`

Minimal pip dependencies:
- `transformers`
- `huggingface_hub`
- `tokenizers`

## 2026-02-16 - Phase 1 Part 2 Execution (Blocked by Offline Dependency Install)

Environment setup and validation:
- Created local venv at `.venv/`.
- Activated `.venv` and checked versions:
  - `Python 3.14.3`
  - `pip 26.0` from `.venv/lib/python3.14/site-packages/pip`

Commands run:
- `. .venv/bin/activate && python -m pip install --upgrade pip`
- `. .venv/bin/activate && pip install transformers huggingface_hub tokenizers`
- `. .venv/bin/activate && python src/tokenizer_fetch.py`
- `. .venv/bin/activate && python src/tokenizer.py`
- `. .venv/bin/activate && python src/tokenizer_bench.py`

Key outputs:
- `pip install ...` failed due offline/DNS resolution errors while contacting PyPI.
- `python src/tokenizer_fetch.py` failed with `ModuleNotFoundError: No module named 'huggingface_hub'`.
- `python src/tokenizer.py` failed with `ModuleNotFoundError: No module named 'transformers'`.
- `python src/tokenizer_bench.py` failed with `ModuleNotFoundError: No module named 'transformers'`.

Notes:
- `assets/tokenizer/qwen3/source_model.json` was not generated because fetch dependencies could not be installed.

## 2026-02-16 - Phase 1 Part 2 (venv + tokenizer fetch/bench)

Ran inside local venv `.venv`:
- Installed deps: transformers, huggingface_hub, tokenizers
- Vendored tokenizer: `python src/tokenizer_fetch.py`
  - chosen_model_id: Qwen/Qwen3-0.6B
- Local load smoke test: `python src/tokenizer.py`
- Bench: `python src/tokenizer_bench.py` -> docs/tokenizer_bench.json

Key outputs:
- assets/tokenizer/qwen3/source_model.json created
- docs/tokenizer_bench.json created

## 2026-02-16 - Phase 2 Step 1 (Data pack/shards, Strategy B)

Implemented:
- `scripts/pack_docs.py`
  - Splits docs by non-empty lines by default.
  - Supports `--split blanklines` for paragraph splitting.
  - Tokenizes each doc with local Qwen3 tokenizer (`src/tokenizer.py` loader).
  - Appends EOS token after each doc.
  - Writes uint32 binary token file and adjacent `.meta.json`.
- `scripts/inspect_bin.py`
  - Reads `.bin` + `.meta.json`.
  - Prints first N token IDs and decoded preview of first K tokens.

Commands run:
- `. .venv/bin/activate && python scripts/pack_docs.py eval/sample_mixed.txt`
- `. .venv/bin/activate && python scripts/pack_docs.py eval/sample_mixed.txt --split blanklines --out data/sample_mixed_blanklines.bin`
- `. .venv/bin/activate && python scripts/inspect_bin.py data/sample_mixed.bin --n 24 --k 48`
- `ls -lh data/sample_mixed.bin data/sample_mixed.meta.json`
- `cat data/sample_mixed.meta.json`

Key outputs:
- Default split run:
  - `split=lines`
  - `docs=3`
  - `out_bin=data/sample_mixed.bin`
  - `token_count=47`
  - `eos_token_id=151645`
- Blankline split run:
  - `split=blanklines`
  - `docs=1`
  - `out_bin=data/sample_mixed_blanklines.bin`
  - `token_count=45`
- Inspect run:
  - `meta_token_count=47`
  - `token_count_read=47`
  - `first_24_token_ids=[9707, 1879, 0, 1096, 374, 264, 45958, 46842, 6077, 13, 151645, 114854, 81705, 46944, 15946, 82847, 105063, 108704, 3837, 100751, 101098, 109371, 81705, 1773]`
  - `decoded_first_48_tokens=Hello world! This is a tokenizer sanity sample.<|im_end|>今天我们测试一个中英混合文本，用于快速基准测试。<|im_end|>Numbers: 12345, punctuation: !?., and symbols: @#&*<|im_end|>`
- File checks:
  - `data/sample_mixed.bin` size: `188B`
  - `data/sample_mixed.meta.json` size: `460B`
  - Meta includes: `dtype=uint32`, `token_count=47`, `eos_token_id=151645`, `tokenizer_source.chosen_model_id=Qwen/Qwen3-0.6B`
## 2026-02-16 - Phase 2 Step 2 (Minimal Transformer from scratch)

Additional progress (later on 2026-02-16):
- Built 100M-token sharded corpus: `data/shards/mix50m.index.json` (18 shards).
- Added shard-index loader + deterministic shard rotation.
- Added budget + early-stopping trainer: `scripts/train_until_good.py`.
- Hardened checkpoints: atomic write (`*.tmp` → rename), save_every=500, keep_last_k=3.
- Documented workflow in `CLAWREAD.md`.

Implemented core code:
- `src/model.py`
  - Decoder-only Transformer LM with:
    - token embedding
    - tied output projection (`F.linear(..., tok_emb.weight)`)
    - `RMSNorm`
    - RoPE (applied to Q/K)
    - SwiGLU FFN
    - causal self-attention mask
    - autoregressive `generate(...)`
- `src/data_loader.py`
  - `TokenBatchLoader` loads uint32 `.bin` with `np.memmap`
  - yields random `(x, y)` next-token batches for `block_size` / `batch_size`
  - supports tiny streams by wrapping indices modulo stream length
- `src/train_core.py`
  - `train_core(...)` minimal train loop (no CLI)
  - trains for `N` steps
  - saves checkpoint with model state/config/losses

Implemented scripts:
- `scripts/train_tiny.py`
  - trains on `data/sample_mixed.bin`
  - tiny config: `n_layer=2`, `n_head=4`, `d_model=256`, `block_size=128`, `batch_size=8`, `steps=100`
  - writes `checkpoints/tiny.pt`
- `scripts/sample.py`
  - loads checkpoint + local tokenizer
  - samples from prompt with temperature + top-k

Dependency/install notes (.venv):
- Checked torch: missing initially
- Attempted install: `. .venv/bin/activate && pip install torch` (offline index retries)
- Final resolved runtime torch version present in `.venv`:
  - `torch==2.10.0`

Commands run:
- `. .venv/bin/activate && python -c "import torch; print(torch.__version__)"`
- `. .venv/bin/activate && pip install torch`
- `. .venv/bin/activate && python scripts/train_tiny.py`
- `. .venv/bin/activate && python scripts/sample.py --ckpt checkpoints/tiny.pt --prompt "Hello" --max-new-tokens 40 --temperature 0.8 --top-k 20`

Key outputs:
- Training (CPU):
  - `step=1/100 loss=11.9144`
  - `step=100/100 loss=0.3404`
  - `checkpoint_path=/Users/william/Documents/wlsslm/checkpoints/tiny.pt`
  - `loss_start=11.914421081542969`
  - `loss_end=0.3404248356819153`
- Sample snippet:
  - `Hello world! This is a tokenizer sanity sample.<|im_end|>今天我们测试一个中英混合文本，用于快速基准测试。<|im_end|>Numbers: 12345, punctuation: !?., and symbols`

Fixes during run:
- Updated `src/model.py` loss flattening from `.view(...)` to `.reshape(...)` to avoid non-contiguous tensor runtime error.


## 2026-02-19 — Windows OpenSSH + WSL FineWeb-only 1B build interruptions

- Windows host 192.168.110.78: SSH stopped working (TCP/22 timeout). Root cause was sshd not running + firewall/profile mismatch (WLAN Public).
- Fixed by running Admin PowerShell:
  - Start-Service sshd (listening on 0.0.0.0:22 and ::22)
  - Firewall OpenSSH rule set to Profile Any
  - WLAN profile switched to Private
- WSL build (FineWeb-only 1B shards): `scripts/build_fineweb_only.py`
  - Progress reached ~353.85M tokens (shard0071) and later ~370.87M tokens (shard0075).
  - Build crashed at one point due to missing Python dependency: `ModuleNotFoundError: transformers` (running outside `.venv`).
  - SSH disconnects/timeouts occurred when VPN/network was toggled on the Windows host; tmux socket missing indicates WSL restart/stop.

Next actions:
- Ensure builder always runs via `./.venv/bin/python` and dependencies are installed inside `.venv`.
- Keep Windows network stable (avoid toggling VPN during long builds) or expect reconnect + resume.
