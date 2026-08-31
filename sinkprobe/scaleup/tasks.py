"""Long-context tasks over real text.

Six skills, not one. Each item is a window of real text with task symbols spliced
in, so a model has to read natural context and still do the lookup. Answers come
from a symbol alphabet disjoint from text bytes, which is what makes a target
identifiable at a distance the model does not know in advance: the earlier version
of this study hid the value among filler drawn from the same vocabulary, and no
model could have solved it.

Every task exposes the same item shape, so one trainer and one scorer serve all of
them: tokens, a type per position (text, answer, ignored) and the positions whose
prediction is scored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .corpus import (BOS, KEY_SYMBOLS, N_KEY, N_VAL, QRY, SEP, VAL_SYMBOLS, VOCAB_SIZE,
                     ByteWindows)

IGNORE, REST, ANSWER = 0, 1, 2

KINDS = ("kv_vargap", "kv_multikey", "multi_hop", "freq_sym", "span_copy", "lm")


@dataclass
class TaskConfig:
    kind: str = "kv_vargap"
    n_pairs: int = 8             # real key-value pairs in the haystack
    n_decoys: int = 4            # extra pairs that are never queried
    gap_min: int = 1             # text tokens between a key and its value
    gap_max: int = 16
    n_queries: int = 1           # 1 keeps one scored trial per item
    hops: int = 3                # multi_hop chain length
    n_syms: int = 6              # freq_sym candidates
    span: int = 4                # span_copy span length
    min_text: int = 24           # text tokens between spliced blocks

    def to_dict(self) -> Dict:
        return asdict(self)


def _chunk_sizes(total: int, n_chunks: int, min_each: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Split a text budget into n_chunks pieces, each at least min_each long."""
    if total < n_chunks * min_each:
        raise ValueError(f"text budget {total} too small for {n_chunks} chunks")
    free = total - n_chunks * min_each
    cuts = np.sort(rng.integers(0, free + 1, size=max(0, n_chunks - 1)))
    sizes = np.diff(np.concatenate([[0], cuts, [free]])) + min_each
    return sizes.astype(np.int64)


def _splice(text: np.ndarray, blocks: Sequence[np.ndarray], body_len: int,
            min_text: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[int]]:
    """Interleave blocks into text, keeping the body exactly body_len tokens."""
    used = int(sum(len(b) for b in blocks))
    budget = body_len - used
    sizes = _chunk_sizes(budget, len(blocks) + 1, min_text, rng)
    out: List[np.ndarray] = []
    positions: List[int] = []
    cursor, taken = 0, 0
    for i, block in enumerate(blocks):
        out.append(text[taken:taken + sizes[i]])
        cursor += int(sizes[i])
        taken += int(sizes[i])
        positions.append(cursor)
        out.append(block)
        cursor += len(block)
    out.append(text[taken:taken + sizes[-1]])
    body = np.concatenate(out)
    assert body.size == body_len, (body.size, body_len)
    return body, positions


class LongContextTask:
    """One task kind, built on windows of real text."""

    def __init__(self, cfg: TaskConfig, windows: ByteWindows):
        self.cfg = cfg
        self.windows = windows

    # -- vocabulary bookkeeping -------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def chance(self) -> float:
        """Accuracy of guessing uniformly inside the answer alphabet."""
        if self.cfg.kind in ("kv_vargap", "kv_multikey", "multi_hop", "span_copy"):
            return 1.0 / N_VAL
        if self.cfg.kind == "freq_sym":
            return 1.0 / max(2, self.cfg.n_syms)
        return 0.0

    def query_block(self) -> int:
        """Tokens appended per query, including the answer."""
        if self.cfg.kind == "freq_sym":
            return 2                          # marker, answer
        if self.cfg.kind == "span_copy":
            return 3                          # marker, cue symbol, answer
        return 3                              # marker, key symbol, answer

    def n_scored(self) -> int:
        """Queries actually written into an item, which is not always what was asked for."""
        c = self.cfg
        if c.kind in ("kv_vargap", "kv_multikey"):
            return max(1, min(c.n_queries, c.n_pairs))
        if c.kind == "lm":
            return 0
        return 1

    def required_length(self) -> int:
        """Shortest input this configuration always fits in."""
        c = self.cfg
        if c.kind == "lm":
            return 16
        if c.kind == "multi_hop":
            blocks = c.hops + c.n_decoys
            widest = 3
        elif c.kind == "freq_sym":
            blocks = (c.n_syms * (c.n_syms + 3)) // 2      # counts run 2..n_syms+1
            widest = 1
        elif c.kind == "span_copy":
            blocks = 2
            widest = c.span
        else:
            blocks = c.n_pairs + c.n_decoys
            widest = 1 + c.gap_max + 1
        body = blocks * widest + (blocks + 1) * c.min_text
        return 1 + body + self.n_scored() * self.query_block()

    # -- item construction -----------------------------------------------------
    def _text(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.windows.window(n, rng)

    def _kv(self, body_len: int, rng: np.random.Generator) -> Dict:
        """Key-value pairs at a distance the model is not told."""
        c = self.cfg
        n_total = c.n_pairs + c.n_decoys
        keys = rng.choice(KEY_SYMBOLS, size=n_total, replace=False)
        values = rng.choice(VAL_SYMBOLS, size=n_total, replace=False)
        gaps = rng.integers(c.gap_min, c.gap_max + 1, size=n_total)
        text = self._text(body_len, rng)
        blocks, cursor = [], 0
        for i in range(n_total):
            fill = text[cursor:cursor + int(gaps[i])]
            cursor += int(gaps[i])
            blocks.append(np.concatenate([[keys[i]], fill, [values[i]]]).astype(np.int64))
        order = rng.permutation(n_total)
        body, positions = _splice(text, [blocks[i] for i in order], body_len, c.min_text, rng)
        where = {int(order[j]): positions[j] for j in range(n_total)}
        asked = rng.choice(c.n_pairs, size=self.n_scored(), replace=False)
        return {"body": body, "keys": keys, "values": values, "gaps": gaps,
                "asked": asked, "where": where}

    def _multi_hop(self, body_len: int, rng: np.random.Generator) -> Dict:
        """x1 holds a value, x2 points at x1, x3 points at x2: answer needs the chain."""
        c = self.cfg
        n_names = c.hops + c.n_decoys
        names = rng.choice(KEY_SYMBOLS, size=n_names, replace=False)
        values = rng.choice(VAL_SYMBOLS, size=1 + c.n_decoys, replace=False)
        links = [np.array([names[0], SEP, values[0]], dtype=np.int64)]
        for h in range(1, c.hops):
            links.append(np.array([names[h], SEP, names[h - 1]], dtype=np.int64))
        for d in range(c.n_decoys):
            links.append(np.array([names[c.hops + d], SEP, values[1 + d]], dtype=np.int64))
        text = self._text(body_len, rng)
        order = rng.permutation(len(links))
        body, _ = _splice(text, [links[i] for i in order], body_len, c.min_text, rng)
        return {"body": body, "query": names[c.hops - 1], "answer": values[0]}

    def _freq_sym(self, body_len: int, rng: np.random.Generator) -> Dict:
        """Which symbol occurs most often: an answer no single position carries."""
        c = self.cfg
        syms = rng.choice(KEY_SYMBOLS, size=c.n_syms, replace=False)
        counts = rng.permutation(np.arange(1, c.n_syms + 1)) + 1
        winner = int(np.argmax(counts))
        blocks = [np.array([syms[i]], dtype=np.int64)
                  for i in range(c.n_syms) for _ in range(int(counts[i]))]
        text = self._text(body_len, rng)
        order = rng.permutation(len(blocks))
        body, _ = _splice(text, [blocks[i] for i in order], body_len, c.min_text, rng)
        return {"body": body, "answer": syms[winner], "counts": counts.tolist()}

    def _span_copy(self, body_len: int, rng: np.random.Generator) -> Dict:
        """A span appears once; its first symbol reappears and the rest must follow."""
        c = self.cfg
        span = rng.choice(VAL_SYMBOLS, size=c.span, replace=False)
        text = self._text(body_len, rng)
        blocks = [span.astype(np.int64), np.array([SEP], dtype=np.int64)]
        body, _ = _splice(text, blocks, body_len, c.min_text, rng)
        return {"body": body, "cue": span[0], "answer": span[1]}

    def _one(self, length: int, rng: np.random.Generator):
        c = self.cfg
        need = self.required_length()
        if length < need:
            raise ValueError(f"length {length} too short for {c.kind}: needs {need}")
        if c.kind == "lm":
            seq = np.concatenate([[BOS], self._text(length - 1, rng)]).astype(np.int64)
            types = np.full(length, REST, dtype=np.int64)
            types[0] = IGNORE
            return seq, types, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), \
                np.zeros(0, dtype=np.float32)

        qb = self.query_block()
        n_q = self.n_scored()
        body_len = length - 1 - n_q * qb
        if c.kind in ("kv_vargap", "kv_multikey"):
            item = self._kv(body_len, rng)
            queries = [(item["keys"][p], item["values"][p], int(item["gaps"][p]),
                        item["where"][int(p)]) for p in item["asked"]]
        elif c.kind == "multi_hop":
            item = self._multi_hop(body_len, rng)
            queries = [(item["query"], item["answer"], c.hops, body_len // 2)]
        elif c.kind == "freq_sym":
            item = self._freq_sym(body_len, rng)
            queries = [(None, item["answer"], 0, body_len // 2)]
        elif c.kind == "span_copy":
            item = self._span_copy(body_len, rng)
            queries = [(item["cue"], item["answer"], c.span, body_len // 2)]
        else:
            raise ValueError(f"unknown task kind {c.kind}")

        seq = np.zeros(length, dtype=np.int64)
        types = np.zeros(length, dtype=np.int64)
        seq[0] = BOS
        seq[1:1 + body_len] = item["body"]
        types[:body_len] = REST                      # predict the next body token
        ans_pos = np.zeros(len(queries), dtype=np.int64)
        gaps = np.zeros(len(queries), dtype=np.int64)
        depth = np.zeros(len(queries), dtype=np.float32)
        base = 1 + body_len
        for qi, (key, answer, gap, where) in enumerate(queries):
            o = base + qi * qb
            seq[o] = QRY
            if key is None:
                seq[o + 1] = answer
                ans_pos[qi] = o
            else:
                seq[o + 1] = key
                seq[o + 2] = answer
                ans_pos[qi] = o + 1
            types[ans_pos[qi]] = ANSWER
            gaps[qi] = gap
            depth[qi] = where / max(1, body_len)
        return seq, types, ans_pos, gaps, depth

    def build(self, batch_size: int, length: int,
              rng: Optional[np.random.Generator] = None) -> Dict[str, torch.Tensor]:
        rng = rng or np.random.default_rng()
        out = [self._one(length, rng) for _ in range(batch_size)]
        seq, types, ans, gap, dep = (np.stack(z) for z in zip(*out))
        return {"tokens": torch.from_numpy(seq), "types": torch.from_numpy(types),
                "answer_pos": torch.from_numpy(ans), "answer_gap": torch.from_numpy(gap),
                "answer_depth": torch.from_numpy(dep)}


def easy_config(kind: str) -> TaskConfig:
    """The rung a working model must clear quickly. Used by the learnability gate."""
    if kind in ("kv_vargap", "kv_multikey"):
        return TaskConfig(kind=kind, n_pairs=4, n_decoys=0, gap_min=1, gap_max=1,
                          n_queries=4, min_text=8)
    if kind == "multi_hop":
        return TaskConfig(kind=kind, hops=1, n_decoys=1, min_text=8)
    if kind == "freq_sym":
        return TaskConfig(kind=kind, n_syms=2, min_text=8)
    if kind == "span_copy":
        return TaskConfig(kind=kind, span=2, min_text=8)
    return TaskConfig(kind=kind)


def study_config(kind: str, gap_max: int = 16) -> TaskConfig:
    """The setting the paper reports."""
    if kind == "kv_vargap":
        return TaskConfig(kind=kind, n_pairs=8, n_decoys=4, gap_min=1, gap_max=gap_max)
    if kind == "kv_multikey":
        return TaskConfig(kind=kind, n_pairs=8, n_decoys=10, gap_min=1, gap_max=gap_max)
    if kind == "multi_hop":
        return TaskConfig(kind=kind, hops=3, n_decoys=6)
    if kind == "freq_sym":
        return TaskConfig(kind=kind, n_syms=6)
    if kind == "span_copy":
        return TaskConfig(kind=kind, span=6)
    return TaskConfig(kind=kind)
