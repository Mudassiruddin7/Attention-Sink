"""Real text, as bytes, with no API and no tokenizer to download.

The corpus is fetched once, concatenated into a flat byte file and read through a
memmap, so a 12-hour session spends its time training rather than parsing. Byte
level keeps the vocabulary at 256 plus a handful of task symbols, which is what
makes a from-scratch study affordable on one GPU.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Byte values 0..255 are text. Everything above is a task symbol.
BOS = 256
QRY = 257
SEP = 258
KEY_BASE = 259
N_KEY = 64
VAL_BASE = KEY_BASE + N_KEY
N_VAL = 64
VOCAB_SIZE = VAL_BASE + N_VAL                      # 387

KEY_SYMBOLS = np.arange(KEY_BASE, KEY_BASE + N_KEY, dtype=np.int64)
VAL_SYMBOLS = np.arange(VAL_BASE, VAL_BASE + N_VAL, dtype=np.int64)

# Parquet shards on the hub, reachable over plain HTTPS without a token.
SOURCES: Dict[str, List[str]] = {
    "wikitext103": [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1/train-00001-of-00002.parquet",
    ],
}


@dataclass
class CorpusConfig:
    name: str = "wikitext103"
    cache: str = "corpus"
    shards: int = 1                  # how many parquet shards to take
    val_bytes: int = 4_000_000
    synthetic_bytes: int = 8_000_000  # only used when nothing can be downloaded

    def to_dict(self) -> Dict:
        return {"name": self.name, "shards": self.shards, "val_bytes": self.val_bytes}


def _read_parquet_text(blob: bytes) -> str:
    """Pull the text column out of a parquet shard."""
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(blob))
    column = "text" if "text" in table.column_names else table.column_names[0]
    parts = [s for s in table.column(column).to_pylist() if s]
    return "\n".join(parts)


def _synthetic_text(n_bytes: int, seed: int = 0) -> str:
    """A stand-in with sentence structure and repetition, for offline testing only."""
    rng = np.random.default_rng(seed)
    words = ["the", "model", "layer", "token", "memory", "context", "window", "value",
             "recall", "state", "signal", "depth", "order", "binding", "sequence"]
    out: List[str] = []
    size = 0
    while size < n_bytes:
        n = int(rng.integers(6, 18))
        line = " ".join(words[int(i)] for i in rng.integers(0, len(words), size=n))
        if rng.random() < 0.3 and out:                    # repeat an earlier line
            line = out[int(rng.integers(0, len(out)))]
        out.append(line + ".")
        size += len(line) + 2
    return "\n".join(out)


def ensure_corpus(cfg: CorpusConfig, verbose: bool = True) -> Dict:
    """Download once, then reuse. Returns the paths and a record of what was used."""
    cache = Path(cfg.cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / f"{cfg.name}.json"
    train_path = cache / f"{cfg.name}.train.bin"
    val_path = cache / f"{cfg.name}.val.bin"
    if meta_path.exists() and train_path.exists() and val_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    text, source = "", "synthetic"
    for url in SOURCES.get(cfg.name, [])[: max(1, cfg.shards)]:
        try:
            if verbose:
                print(f"fetching {url.rsplit('/', 1)[-1]}", flush=True)
            with urllib.request.urlopen(url, timeout=180) as fh:
                blob = fh.read()
            text += _read_parquet_text(blob)
            source = cfg.name
        except Exception as exc:                            # offline, or no pyarrow
            if verbose:
                print(f"  could not use that shard: {type(exc).__name__}: {exc}", flush=True)
    if not text:
        if verbose:
            print("falling back to synthetic text; results are for plumbing only", flush=True)
        text = _synthetic_text(cfg.synthetic_bytes)

    raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
    if raw.size < cfg.val_bytes * 3:
        cfg = CorpusConfig(**{**cfg.__dict__, "val_bytes": max(1024, raw.size // 8)})
    raw[: raw.size - cfg.val_bytes].tofile(train_path)
    raw[raw.size - cfg.val_bytes:].tofile(val_path)
    meta = {"name": cfg.name, "source": source, "train_bytes": int(raw.size - cfg.val_bytes),
            "val_bytes": int(cfg.val_bytes), "train": str(train_path), "val": str(val_path),
            "sha1_head": hashlib.sha1(raw[:1_000_000].tobytes()).hexdigest()[:12]}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"corpus ready: {meta['train_bytes'] / 1e6:.1f}MB train, "
              f"{meta['val_bytes'] / 1e6:.1f}MB val, source {source}", flush=True)
    return meta


class ByteWindows:
    """Random windows of text, as byte tokens."""

    def __init__(self, path: str, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.data.size)

    def window(self, length: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or self.rng
        start = int(rng.integers(0, max(1, self.data.size - length - 1)))
        return np.asarray(self.data[start:start + length], dtype=np.int64)

    def batch(self, batch_size: int, length: int,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return np.stack([self.window(length, rng) for _ in range(batch_size)])
