"""Checks for the sink intervention.

    python tests/test_interventions.py

The exact output formula must agree with softmax attention computed with the
bias added to the logit of position 0, and the biased forward must reproduce
the original layer when the bias is zero.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.interventions import (SINK_BIAS, biased_first_key_output,      # noqa: E402
                                     biased_softmax_forward, first_key_attention)
from sinkprobe.layers import SoftmaxAttention                                 # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(f"[{'pass' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def explicit(q, k, v, scale, bias):
    t = q.shape[2]
    s = (q @ k.transpose(-1, -2)) * scale
    if bias == float("-inf"):
        s[..., 1:, 0] = float("-inf")
    else:
        s[..., 1:, 0] = s[..., 1:, 0] + bias
    mask = torch.ones(t, t, dtype=torch.bool).tril()
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
    return p @ v, p[..., 0]


def test_exact_formula():
    torch.manual_seed(0)
    b, h, t, d = 2, 3, 17, 8
    q, k, v = (torch.randn(b, h, t, d, dtype=torch.float64) for _ in range(3))
    scale = d ** -0.5
    o, p0 = explicit(q, k, v, scale, 0.0)
    fk = first_key_attention(q, k, scale)
    err = (fk.double() - p0).abs().max().item()
    check("first-key attention matches the explicit softmax", err < 1e-5, f"max error {err:.1e}")
    for bias in (-2.0, 3.0, float("-inf")):
        ob, p0b = explicit(q, k, v, scale, bias)
        new, p0n = biased_first_key_output(o, v[:, :, :1], p0, bias)
        err_o = (new - ob).abs().max().item()
        err_p = (p0n - p0b)[:, :, 1:].abs().max().item()
        check(f"exact output at bias {bias}", err_o < 1e-9, f"max error {err_o:.1e}")
        check(f"first-key attention after bias {bias}", err_p < 1e-9, f"max error {err_p:.1e}")


def test_biased_forward():
    torch.manual_seed(0)
    x = torch.randn(2, 23, 32)
    for kw in (dict(), dict(gate=True), dict(sink_logit=True), dict(rope=False),
               dict(rope=False, kv_conv=4, logn_ref=8)):
        layer = SoftmaxAttention(32, 4, **kw).eval()
        if layer.sink is not None:
            layer.sink.data.fill_(0.7)
        with torch.no_grad():
            ref, _ = layer(x, collect=True)
            SINK_BIAS["value"] = 0.0
            out, _ = biased_softmax_forward(layer, x, collect=True)
            SINK_BIAS["value"] = float("-inf")
            _, aux = biased_softmax_forward(layer, x, collect=True)
        SINK_BIAS["value"] = 0.0
        err = (out - ref).abs().max().item()
        check(f"biased forward at bias 0 reproduces the layer {kw}", err < 1e-5, f"max error {err:.1e}")
        left = aux["a0"][:, :, 1:].abs().max().item()
        check(f"bias -inf leaves no attention on position 0 {kw}", left < 1e-6, f"max {left:.1e}")


def test_length_scaling():
    """Scaling is the identity up to the reference length and sharpens beyond it."""
    from sinkprobe.layers import LOGN
    torch.manual_seed(0)
    layer = SoftmaxAttention(32, 4, rope=False, kv_conv=4, logn_ref=16).eval()
    short, long_ = torch.randn(2, 16, 32), torch.randn(2, 40, 32)
    with torch.no_grad():
        LOGN["enabled"] = True
        a, _ = layer(short, collect=True)
        LOGN["enabled"] = False
        b, _ = layer(short, collect=True)
        c, aux_off = layer(long_, collect=True)
        LOGN["enabled"] = True
        d, aux_on = layer(long_, collect=True)
    check("length scaling is the identity within the reference length", (a - b).abs().max() < 1e-6)
    check("length scaling changes outputs beyond the reference length", (c - d).abs().max() > 1e-4)
    check("length scaling lowers attention entropy beyond the reference length",
          float(aux_on["ent"][:, :, 20:].mean()) < float(aux_off["ent"][:, :, 20:].mean()))


def main():
    test_exact_formula()
    test_biased_forward()
    test_length_scaling()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
