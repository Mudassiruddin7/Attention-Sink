"""Byte-level natural text, for a check outside the synthetic haystack.

Training draws windows of raw WikiText-2 training text. Nothing marks repeated
content, so whatever the model learns about copying comes from ordinary text.

Evaluation builds a copy probe from held-out WikiText-103 validation text:

    BOS  [passage ...  span at depth d  ... passage]  SEP  span

The span is `span` bytes taken from a uniformly drawn depth of the passage and
repeated after a separator. Scored positions predict bytes of the repeat after
its first `lead` bytes, so the span can be identified from what was already
read. With control=True the repeated span comes from elsewhere in the text
and is not in the context, which gives the accuracy a model reaches from
language statistics alone.

The batch has the same fields as sinkprobe.data.HaystackTask, so training and
sinkprobe.metrics.evaluate work unchanged: scored positions are labelled
ANSWER, answer_pos holds them and answer_depth holds the depth of the span.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import ANSWER, OTHER


@dataclass
class TextConfig:
    train_path: str = "data/wikitext2_train.txt"
    eval_path: str = "data/wikitext103_validation.txt"
    span: int = 32
    lead: int = 8
    n_queries: int = 8
    control: bool = False
    sep: int = 256
    bos: int = 257

    @property
    def vocab_size(self) -> int:
        return 258

    def to_dict(self):
        d = asdict(self)
        d["vocab_size"] = self.vocab_size
        return d


class TextTask:
    def __init__(self, cfg: TextConfig):
        self.cfg = cfg
        self.train_bytes = np.frombuffer(open(cfg.train_path, "rb").read(), dtype=np.uint8)
        self.eval_bytes = np.frombuffer(open(cfg.eval_path, "rb").read(), dtype=np.uint8)

    def reserved_tokens(self):
        """Tokens that are never a valid answer: the separator and the sequence marker."""
        return (self.cfg.sep, self.cfg.bos)

    def build(self, batch_size: int, seq_len: int, rng: np.random.Generator, skew=None):
        """skew is None for training windows; any value builds the copy probe."""
        if skew is None:
            return self._train(batch_size, seq_len, rng)
        return self._probe(batch_size, seq_len, rng)

    def _train(self, b: int, t: int, rng):
        starts = rng.integers(0, len(self.train_bytes) - t, size=b)
        x = np.stack([self.train_bytes[s:s + t] for s in starts]).astype(np.int64)
        x[:, 0] = self.cfg.bos
        return {"tokens": torch.from_numpy(x),
                "types": torch.full((b, t), OTHER, dtype=torch.long),
                "answer_pos": torch.zeros(b, 1, dtype=torch.long),
                "answer_depth": torch.zeros(b, 1)}

    def _probe(self, b: int, t: int, rng):
        c = self.cfg
        body = t - 2 - c.span
        seqs, types, ans, dep = [], [], [], []
        sep_idx = t - c.span - 1
        scored = np.arange(sep_idx + c.lead, t - 1)          # predict repeat bytes lead..span-1
        for _ in range(b):
            s = int(rng.integers(0, len(self.eval_bytes) - body))
            passage = self.eval_bytes[s:s + body].astype(np.int64)
            pos = int(rng.integers(0, body - c.span))
            if c.control:
                o = int(rng.integers(0, len(self.eval_bytes) - c.span))
                span = self.eval_bytes[o:o + c.span].astype(np.int64)
            else:
                span = passage[pos:pos + c.span]
            seq = np.concatenate([[c.bos], passage, [c.sep], span])
            ty = np.full(t, OTHER, dtype=np.int64)
            pick = np.sort(rng.choice(scored, size=c.n_queries, replace=False))
            ty[pick] = ANSWER
            seqs.append(seq)
            types.append(ty)
            ans.append(pick)
            dep.append(np.full(c.n_queries, pos / max(1, body - c.span), dtype=np.float32))
        return {"tokens": torch.from_numpy(np.stack(seqs)),
                "types": torch.from_numpy(np.stack(types)),
                "answer_pos": torch.from_numpy(np.stack(ans)),
                "answer_depth": torch.from_numpy(np.stack(dep))}

    def floors(self):
        return 1.0 / 256, 0.0
