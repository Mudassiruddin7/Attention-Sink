"""Checks for scripts/retrieval_attention.py: gold positions and exact attention rows.

    python tests/test_retrieval_attention.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from sinkprobe.data import HaystackTask, TaskConfig      # noqa: E402
from sinkprobe.layers import SoftmaxAttention            # noqa: E402
from sinkprobe.model import ModelConfig, TinyLM          # noqa: E402

import retrieval_attention as ra                         # noqa: E402

LENGTH = 128


def batch(seed=0, bs=4):
    task = HaystackTask(TaskConfig())
    return task, task.build(bs, LENGTH, np.random.default_rng(seed), skew=0.0)


def test_gold_positions_point_at_the_queried_pair():
    task, d = batch()
    x, ans = d["tokens"], d["answer_pos"]
    key_pos, val_pos = ra.gold_positions(x, ans, LENGTH - 3 * task.cfg.n_queries)
    assert torch.equal(x.gather(1, key_pos), x.gather(1, ans))          # the same key token
    assert torch.equal(x.gather(1, val_pos), x.gather(1, ans + 1))      # followed by the answer
    assert bool((val_pos < LENGTH - 3 * task.cfg.n_queries).all())      # inside the haystack


def test_attention_rows_match_the_model_attention():
    torch.manual_seed(0)
    task, d = batch(seed=1)
    ans = d["answer_pos"]
    # Bound keys with the length scaling active (logn_ref below the length), rotary
    # positions, and a NoPE hybrid with its global layer last.
    for variant, extra in (("hybrid_bka_first", dict(logn_ref=64)), ("softmax", {}), ("hybrid_nope", {})):
        model = TinyLM(ModelConfig.from_variant(variant, vocab_size=task.cfg.vocab_size, **extra)).eval()
        inputs = {}
        hooks = [blk.mix.register_forward_pre_hook(lambda m, args, i=i: inputs.__setitem__(i, args[0]))
                 for i, blk in enumerate(model.blocks) if isinstance(blk.mix, SoftmaxAttention)]
        with torch.no_grad():
            _, stats = model(d["tokens"], collect=True)
        for hk in hooks:
            hk.remove()
        checked = 0
        for i, aux in enumerate(stats["layers"]):
            if "a0" not in aux:
                continue
            with torch.no_grad():
                p = ra.attention_rows(model.blocks[i].mix, inputs[i], ans)  # (B, H, Q, T)
            expected = aux["a0"].gather(2, ans[:, None, :].expand(-1, p.shape[1], -1))
            assert torch.allclose(p[..., 0], expected, atol=1e-5), (variant, i)
            assert torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-5)
            future = torch.arange(LENGTH).view(1, 1, 1, -1) > ans.view(ans.shape[0], 1, -1, 1)
            assert float(p.masked_select(future).abs().max()) == 0.0
            checked += 1
        assert checked == (8 if variant == "softmax" else 2), (variant, checked)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
