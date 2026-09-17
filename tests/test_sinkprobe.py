"""Checks that the pieces of the suite do what the paper says they do.

    python tests/test_sinkprobe.py

No test framework is needed. Each check prints one line and the script exits
non zero if any check fails. Everything runs on the CPU in well under a
minute.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.costmodel import HybridSpec                                    # noqa: E402
from sinkprobe.data import ANSWER, COPY, NOOP, HaystackTask, TaskConfig        # noqa: E402
from sinkprobe.layers import (SoftmaxAttention, gated_delta_rule_chunked,     # noqa: E402
                              gated_delta_rule_recurrent, softpick)
from sinkprobe.metrics import (attention_diagnostics, trials_for_halfwidth,   # noqa: E402
                               uniform_sink_reference, wilson_interval)
from sinkprobe.model import VARIANTS, ModelConfig, TinyLM                      # noqa: E402
from sinkprobe.residuals import BlockAttentionResidual                        # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(f"[{'pass' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_task():
    cfg = TaskConfig()
    task = HaystackTask(cfg)
    rng = np.random.default_rng(0)
    d = task.build(32, 256, rng)
    x, ty, ap, dep = d["tokens"], d["types"], d["answer_pos"], d["answer_depth"]
    body_len = 256 - 1 - 3 * cfg.n_queries
    ok_value, ok_once, ok_depth = True, True, True
    for b in range(x.shape[0]):
        for qi in range(cfg.n_queries):
            p = int(ap[b, qi])
            key, val = int(x[b, p]), int(x[b, p + 1])
            ok_value &= cfg.value_base <= val < cfg.value_base + cfg.value_range
            hits = (x[b, 1:1 + body_len] == key).nonzero().flatten()
            ok_once &= len(hits) == 1
            if len(hits) == 1:
                kp = int(hits[0])
                ok_value &= int(x[b, 1 + kp + 1]) == val
                ok_depth &= abs(kp / (body_len - 2) - float(dep[b, qi])) < 1e-6
    check("every answer is the value stored with the queried key", ok_value)
    check("every queried key appears exactly once in the haystack", ok_once)
    check("answer depth labels match the stored position", ok_depth)
    check("answer positions are labelled ANSWER",
          bool((ty.gather(1, ap) == ANSWER).all()) and int((ty == ANSWER).sum()) == ap.numel())

    all_noise = HaystackTask(TaskConfig(p_noop=1.0)).build(16, 256, rng)["types"]
    no_noise = HaystackTask(TaskConfig(p_noop=0.0)).build(16, 256, rng)["types"]
    check("p_noop=1 leaves no copy positions", int((all_noise == COPY).sum()) == 0)
    share = float((no_noise == COPY).float().mean())
    check("p_noop=0 makes most positions copies", share > 0.5, f"copy share {share:.2f}")

    late = HaystackTask(TaskConfig(query_skew=8.0)).build(64, 256, rng)["answer_depth"].mean()
    early = HaystackTask(TaskConfig(query_skew=-8.0)).build(64, 256, rng)["answer_depth"].mean()
    check("query_skew tilts which pairs are asked", late > 0.65 and early < 0.35,
          f"mean depth {float(late):.2f} vs {float(early):.2f}")


def test_delta_rule_is_exact():
    torch.manual_seed(0)
    b, h, t, dk, dv = 2, 2, 37, 6, 5
    q = F.normalize(torch.randn(b, h, t, dk, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(b, h, t, dk, dtype=torch.float64), dim=-1)
    v = torch.randn(b, h, t, dv, dtype=torch.float64)
    beta = torch.rand(b, h, t, dtype=torch.float64)
    la = -2 * torch.rand(b, h, t, dtype=torch.float64)
    for delta in (True, False):
        o_ref, s_ref = gated_delta_rule_recurrent(q, k, v, beta, la, delta=delta)
        worst = 0.0
        for c in (1, 8, 16, 64):
            o, s = gated_delta_rule_chunked(q, k, v, beta, la, chunk=c, delta=delta)
            worst = max(worst, (o - o_ref).abs().max().item(), (s - s_ref).abs().max().item())
        check(f"chunked delta rule equals the recurrence (delta={delta})", worst < 1e-10,
              f"max error {worst:.1e}")


def test_causality():
    torch.manual_seed(0)
    for variant in VARIANTS:
        m = TinyLM(ModelConfig.from_variant(variant, vocab_size=50, d_model=32, n_layers=8,
                                            n_heads=2, d_ff=64, chunk=8)).eval()
        x = torch.randint(0, 50, (2, 29))
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 50
        with torch.no_grad():
            drift = (m(x)[0][:, :-1] - m(x2)[0][:, :-1]).abs().max().item()
        check(f"{variant} is causal", drift < 1e-5, f"max drift {drift:.1e}")


def test_normalisers():
    torch.manual_seed(0)
    t = 12
    mask = torch.ones(t, t, dtype=torch.bool).tril()
    p = softpick(torch.randn(3, t, t), mask)
    check("softpick rows are non negative and sum to at most one",
          bool((p >= 0).all()) and bool((p.sum(-1) <= 1 + 1e-6).all()))
    check("softpick puts nothing on future positions", float(p.masked_select(~mask).abs().sum()) == 0.0)
    z = softpick(-torch.rand(1, t, t) - 0.1, mask)
    check("softpick can return an all zero row", float(z.abs().max()) < 1e-6)

    layer = SoftmaxAttention(16, 2, rope=False, sink_logit=True).eval()
    with torch.no_grad():
        layer.sink.fill_(1.5)
        _, aux = layer(torch.randn(2, t, 16), collect=True)
    total = aux["row_sum"] + aux["virt"]
    check("token mass plus virtual sink mass is one", float((total - 1).abs().max()) < 1e-5)


def test_metrics():
    t = 64
    a = torch.zeros(1, 1, t)
    a[..., 1:] = 1.0 / torch.arange(2, t + 1, dtype=torch.float32)
    stats = {"layers": [{"a0": a, "ent": torch.ones(1, 1, t), "kind": "S"}]}
    diag = attention_diagnostics(stats, torch.zeros(1, t, dtype=torch.long))
    check("uniform attention has sink ratio one", abs(diag["sink_ratio"] - 1) < 1e-6,
          f"ratio {diag['sink_ratio']:.4f}")
    check("uniform reference is ln(T)/T scale, not 1/T",
          uniform_sink_reference(256) > 4 / 256, f"{uniform_sink_reference(256):.4f}")
    lo, hi = wilson_interval(50, 100)
    check("Wilson interval brackets the estimate", lo < 0.5 < hi)
    check("sample size formula", trials_for_halfwidth(0.05) == 385)


def test_attention_residuals():
    torch.manual_seed(0)
    r = BlockAttentionResidual(8, 4, 2)
    e = torch.randn(2, 5, 8)
    r.reset(e)
    out = r.read(0)
    check("first layer reads the embedding", torch.allclose(out, e))
    r.write(0, torch.randn(2, 5, 8))
    r.write(1, torch.randn(2, 5, 8))
    r.read(2)
    w = r.weights()
    check("depth weights form a distribution", abs(float(w.sum()) - 1) < 1e-5, f"{w.tolist()}")


def test_cost_model():
    spec = HybridSpec()
    check("linear layer fraction is 69 of 93", abs(spec.constant_layer_fraction() - 69 / 93) < 1e-9)
    check("recurrent state does not grow with context",
          spec.linear_state_bytes(4_096) == spec.linear_state_bytes(1_048_576))
    check("global cache doubles with context",
          spec.global_cache_bytes(8_192) == 2 * spec.global_cache_bytes(4_096))


def main():
    for fn in (test_task, test_delta_rule_is_exact, test_causality, test_normalisers,
               test_metrics, test_attention_residuals, test_cost_model):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
