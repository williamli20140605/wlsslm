from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class TokenBatchLoader:
    """Random next-token batches from a uint32 token stream.

    Supports:
    - single .bin file (uint32 tokens)
    - shard index json produced by scripts/build_corpus_50m.py
    """

    data: np.memmap
    block_size: int
    batch_size: int
    device: torch.device

    # Optional shard mode
    shard_paths: list[Path] | None = None
    shard_token_counts: list[int] | None = None
    shard_i: int = 0
    shard_rotate_every_batches: int = 0  # 0 = disabled
    _batch_counter: int = 0

    @classmethod
    def from_bin(
        cls,
        bin_path: str | Path,
        block_size: int,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> "TokenBatchLoader":
        path = Path(bin_path)
        if not path.exists():
            raise FileNotFoundError(f"Token bin not found: {path}")

        data = np.memmap(path, mode="r", dtype=np.uint32)
        if data.size < 2:
            raise ValueError("Token stream must contain at least 2 tokens")

        return cls(
            data=data,
            block_size=block_size,
            batch_size=batch_size,
            device=torch.device(device),
        )

    @classmethod
    def from_shard_index(
        cls,
        index_path: str | Path,
        *,
        block_size: int,
        batch_size: int,
        device: str | torch.device = "cpu",
        base_dir: str | Path | None = None,
        shard_rotate_every_batches: int = 50,
    ) -> "TokenBatchLoader":
        """Load a shard index json and start from shard 0.

        Index format is the one produced by scripts/build_corpus_50m.py:
          {"shards": [{"bin": "mix50m_shard0001.bin", "token_count": ...}, ...]}

        base_dir defaults to index file directory.
        """
        import json

        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        if base_dir is None:
            base_dir = index_path.parent
        base_dir = Path(base_dir)

        j: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
        shards = j.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("Index has no shards")

        shard_paths: list[Path] = []
        shard_token_counts: list[int] = []
        for s in shards:
            if not isinstance(s, dict):
                continue
            bin_name = s.get("bin")
            if not isinstance(bin_name, str):
                continue
            p = base_dir / bin_name
            if not p.exists():
                raise FileNotFoundError(f"Shard bin missing: {p}")
            shard_paths.append(p)
            shard_token_counts.append(int(s.get("token_count", 0)))

        if not shard_paths:
            raise ValueError("No valid shard paths")

        data = np.memmap(shard_paths[0], mode="r", dtype=np.uint32)
        if data.size < 2:
            raise ValueError("Shard token stream must contain at least 2 tokens")

        return cls(
            data=data,
            block_size=block_size,
            batch_size=batch_size,
            device=torch.device(device),
            shard_paths=shard_paths,
            shard_token_counts=shard_token_counts,
            shard_i=0,
            shard_rotate_every_batches=int(shard_rotate_every_batches),
            _batch_counter=0,
        )

    @classmethod
    def from_shard_paths(
        cls,
        shard_paths: list[str | Path],
        *,
        block_size: int,
        batch_size: int,
        device: str | torch.device = "cpu",
        shard_rotate_every_batches: int = 50,
    ) -> "TokenBatchLoader":
        """Create a shard loader from an explicit list of shard .bin paths."""
        paths = [Path(p) for p in shard_paths]
        if not paths:
            raise ValueError("shard_paths is empty")
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Shard bin missing: {p}")

        data = np.memmap(paths[0], mode="r", dtype=np.uint32)
        if data.size < 2:
            raise ValueError("Shard token stream must contain at least 2 tokens")

        return cls(
            data=data,
            block_size=block_size,
            batch_size=batch_size,
            device=torch.device(device),
            shard_paths=paths,
            shard_token_counts=None,
            shard_i=0,
            shard_rotate_every_batches=int(shard_rotate_every_batches),
            _batch_counter=0,
        )

    @property
    def token_count(self) -> int:
        return int(self.data.size)

    def _maybe_rotate_shard(self) -> None:
        if not self.shard_paths:
            return
        assert self.shard_i is not None
        # move to next shard (cyclic)
        self.shard_i = (self.shard_i + 1) % len(self.shard_paths)
        self.data = np.memmap(self.shard_paths[self.shard_i], mode="r", dtype=np.uint32)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get one random batch.

        Shard mode: we sample within the current shard only; occasionally rotate shards.
        This keeps implementation simple and avoids cross-shard indexing.
        """
        # Supports tiny files by wrapping indices modulo stream length.
        n = self.token_count

        # If shard is too small for block_size, rotate until we find a usable one.
        if n < (self.block_size + 2) and self.shard_paths:
            for _ in range(len(self.shard_paths)):
                self._maybe_rotate_shard()
                n = self.token_count
                if n >= (self.block_size + 2):
                    break
            if n < (self.block_size + 2):
                raise ValueError("All shards are too small for block_size")

        # Deterministic shard rotation (mode 1): rotate every N batches.
        if self.shard_paths is not None and self.shard_rotate_every_batches:
            if self._batch_counter > 0 and (self._batch_counter % self.shard_rotate_every_batches == 0):
                self._maybe_rotate_shard()
                n = self.token_count

        starts = np.random.randint(0, n, size=self.batch_size, dtype=np.int64)

        seq_len = self.block_size + 1
        offsets = np.arange(seq_len, dtype=np.int64)
        idx = (starts[:, None] + offsets[None, :]) % n

        seq = np.asarray(self.data[idx], dtype=np.int64)
        x_np = seq[:, :-1]
        y_np = seq[:, 1:]

        x = torch.from_numpy(x_np).to(self.device, dtype=torch.long)
        y = torch.from_numpy(y_np).to(self.device, dtype=torch.long)
        self._batch_counter += 1
        return x, y
