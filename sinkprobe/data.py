"""Synthetic long context data with two independent knobs.

A sequence is

    BOS  [haystack with key value pairs]  QRY k v  QRY k v ...

The haystack is a run of filler segments of random length. A segment is
either fresh random tokens or a verbatim copy of an earlier segment. Key value
pairs sit at segment boundaries, so they never break a copy. The query block
at the end asks for the values of several of those keys.

Every position is labelled by what it has to predict.

    NOOP    the next token cannot be predicted from anything in the context
            (fresh filler, the first token of a copy, a key or value on its
            first appearance, the key named by a query)
    COPY    the next token continues a copied segment, so reading the right
            earlier position predicts it
    ANSWER  the next token is the value of a queried key
    OTHER   positions that are neither (BOS target bookkeeping, query markers)

Knob one, p_noop, is the share of segments that are fresh random tokens. It
sets how many query positions have nothing worth reading, and nothing else.

Knob two, query_skew, tilts which pairs are queried toward the end
(positive) or the start (negative) of the context, with probability
proportional to exp(query_skew * depth). It sets where retrieval is demanded
during training, and nothing else. Evaluation always queries uniformly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

OTHER, NOOP, COPY, ANSWER = 0, 1, 2, 3
TYPE_NAMES = {OTHER: "other", NOOP: "noop", COPY: "copy", ANSWER: "answer"}


@dataclass
class TaskConfig:
    n_keys: int = 64
    n_values: int = 64
    n_filler: int = 256
    n_pairs: int = 16
    n_queries: int = 8
    p_noop: float = 0.5
    seg_min: int = 8
    seg_max: int = 24
    query_skew: float = 0.0
    shared_vocab: bool = True

    bos: int = 0
    qry: int = 1

    # With shared_vocab (the default) keys and values are ordinary filler
    # tokens, as a needle in real text is made of ordinary words. The keys of a
    # sequence are removed from its filler, so each key occurs once before its
    # query and retrieval is the same copy operation the filler rewards.
    @property
    def filler_base(self) -> int:
        return 2 if self.shared_vocab else 2 + self.n_keys + self.n_values

    @property
    def key_base(self) -> int:
        return self.filler_base if self.shared_vocab else 2

    @property
    def value_base(self) -> int:
        return self.filler_base if self.shared_vocab else 2 + self.n_keys

    @property
    def key_range(self) -> int:
        return self.n_filler if self.shared_vocab else self.n_keys

    @property
    def value_range(self) -> int:
        return self.n_filler if self.shared_vocab else self.n_values

    @property
    def vocab_size(self) -> int:
        return self.filler_base + self.n_filler

    def to_dict(self):
        d = asdict(self)
        d["vocab_size"] = self.vocab_size
        return d


class HaystackTask:
    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg

    def _one(self, seq_len: int, rng: np.random.Generator, skew: float):
        c = self.cfg
        body_len = seq_len - 1 - 3 * c.n_queries
        n_fill = body_len - 2 * c.n_pairs
        if n_fill < c.seg_max:
            raise ValueError("sequence too short for this task configuration")

        keys = rng.choice(c.key_range, size=c.n_pairs, replace=False) + c.key_base
        allowed = np.arange(c.filler_base, c.vocab_size)
        if c.shared_vocab:
            # Neither filler nor values may repeat a key of this sequence.
            allowed = np.setdiff1d(allowed, keys)
            vals = allowed[rng.integers(0, len(allowed), size=c.n_pairs)]
        else:
            vals = rng.integers(0, c.value_range, size=c.n_pairs) + c.value_base

        # Filler segments. pred marks tokens that can be predicted from context.
        segs, seg_pred = [], []
        filled = 0
        while filled < n_fill:
            room = n_fill - filled
            if segs and rng.random() >= c.p_noop:
                src = segs[int(rng.integers(len(segs)))]
                toks = src[: min(len(src), room)].copy()
                pred = np.ones(len(toks), dtype=bool)
                pred[0] = False
            else:
                n = min(int(rng.integers(c.seg_min, c.seg_max + 1)), room)
                toks = allowed[rng.integers(0, len(allowed), size=n)]
                pred = np.zeros(n, dtype=bool)
            segs.append(toks)
            seg_pred.append(pred)
            filled += len(toks)

        # Key value pairs at segment boundaries.
        slot = rng.integers(0, len(segs) + 1, size=c.n_pairs)
        order = np.argsort(slot, kind="stable")

        body = np.empty(body_len, dtype=np.int64)
        body_pred = np.zeros(body_len, dtype=bool)
        pair_pos = np.empty(c.n_pairs, dtype=np.int64)
        pos, j = 0, 0
        for si in range(len(segs) + 1):
            while j < c.n_pairs and slot[order[j]] == si:
                p = order[j]
                body[pos], body[pos + 1] = keys[p], vals[p]
                pair_pos[p] = pos
                pos += 2
                j += 1
            if si < len(segs):
                n = len(segs[si])
                body[pos:pos + n] = segs[si]
                body_pred[pos:pos + n] = seg_pred[si]
                pos += n
        depth = pair_pos / max(body_len - 2, 1)

        # Which pairs are queried. Gumbel top-k samples without replacement
        # with probability proportional to exp(skew * depth).
        score = skew * depth + rng.gumbel(size=c.n_pairs)
        asked = np.argsort(-score)[: c.n_queries]

        seq = np.empty(seq_len, dtype=np.int64)
        seq[0] = c.bos
        seq[1:1 + body_len] = body
        tok_kind = np.full(seq_len, OTHER, dtype=np.int8)      # kind of each token
        tok_kind[1:1 + body_len] = np.where(body_pred, COPY, NOOP)
        base = 1 + body_len
        ans_pos = np.empty(c.n_queries, dtype=np.int64)
        for qi, p in enumerate(asked):
            o = base + 3 * qi
            seq[o], seq[o + 1], seq[o + 2] = c.qry, keys[p], vals[p]
            tok_kind[o + 1] = NOOP
            tok_kind[o + 2] = ANSWER
            ans_pos[qi] = o + 1                                 # predicts the value

        # A position's type is the kind of the token it predicts.
        types = np.full(seq_len, OTHER, dtype=np.int8)
        types[:-1] = tok_kind[1:]
        return seq, types, ans_pos, depth[asked]

    def build(self, batch_size: int, seq_len: int, rng: np.random.Generator,
              skew: float | None = None):
        skew = self.cfg.query_skew if skew is None else skew
        out = [self._one(seq_len, rng, skew) for _ in range(batch_size)]
        seq, types, ans, dep = (np.stack(z) for z in zip(*out))
        return {
            "tokens": torch.from_numpy(seq),
            "types": torch.from_numpy(types.astype(np.int64)),
            "answer_pos": torch.from_numpy(ans),
            "answer_depth": torch.from_numpy(dep.astype(np.float32)),
        }

    def reserved_tokens(self):
        """Tokens that are never a valid answer: the sequence marker and the query marker."""
        return (self.cfg.bos, self.cfg.qry)

    def floors(self):
        """Blind chance and the score for returning any value in the context."""
        return 1.0 / self.cfg.value_range, 1.0 / self.cfg.n_pairs
