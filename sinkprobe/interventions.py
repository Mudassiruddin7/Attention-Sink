"""Causal tests of the attention sink: shift the logit of position 0 and re-measure.

If a model's ability to retrieve evidence runs through its sink, pushing
attention off position 0 should change where it can retrieve. If the two are
separate, retrieval should hold while the sink disappears.

Two implementations give the same result.

biased_first_key_output
    Exact output of softmax attention after adding `bias` to the logit of key
    0, computed from the unbiased output. With p0 the unbiased attention on
    key 0, the biased distribution is p0' = p0 e^b / Z and pj' = pj / Z for
    j > 0, with Z = 1 + p0 (e^b - 1). The output is therefore
    o' = (o + p0 (e^b - 1) v0) / Z. Query 0 sees only key 0 and is unchanged.
    This form keeps the memory-efficient SDPA kernel for the main output, so
    it is what the released-checkpoint probe uses at long context.

biased_softmax_forward
    A forward for sinkprobe.layers.SoftmaxAttention that adds the bias to the
    scores directly. The trained models in this repository are small enough
    for the explicit form, and the tests check the two against each other.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .layers import apply_rope, softpick

SINK_BIAS = {"value": 0.0}


def first_key_attention(query, key, scale, chunk_budget: float = 1.5e7):
    """Attention on key 0 for every query, via a chunked logsumexp. (B,H,T,dh) -> (B,H,T)."""
    b, h, t, _ = query.shape
    kt = key.float().transpose(-1, -2)
    keys = torch.arange(t, device=query.device)
    out = torch.empty(b, h, t, device=query.device, dtype=torch.float32)
    chunk = max(1, int(chunk_budget // max(1, b * h * t)))
    for s in range(0, t, chunk):
        e = min(t, s + chunk)
        logits = (query[:, :, s:e].float() @ kt) * scale
        pos = torch.arange(s, e, device=query.device)
        logits = logits.masked_fill(keys[None, None, None, :] > pos[None, None, :, None], float("-inf"))
        out[:, :, s:e] = torch.exp(logits[..., 0] - torch.logsumexp(logits, dim=-1))
    return out


def biased_first_key_output(o, v0, p0, bias: float):
    """o: (B,H,T,dv) unbiased output, v0: (B,H,1,dv) value at position 0, p0: (B,H,T)."""
    eb = 0.0 if bias == float("-inf") else math.exp(bias)
    f = (p0 * (eb - 1.0)).unsqueeze(-1).to(o.dtype)
    new = (o + f * v0) / (1.0 + f).clamp_min(1e-12)
    new[:, :, 0] = o[:, :, 0]
    p0_new = p0 * eb / (1.0 + p0 * (eb - 1.0)).clamp_min(1e-12)
    p0_new[:, :, 0] = 1.0
    return new, p0_new


def biased_softmax_forward(self, x: torch.Tensor, collect: bool = False):
    """SoftmaxAttention.forward with SINK_BIAS["value"] added to the logit of key 0."""
    bias = SINK_BIAS["value"]
    b, t, d = x.shape
    q, k, v = self.project(x)
    scores = ((q @ k.transpose(-1, -2)) / math.sqrt(self.dh)).float()
    mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    col = torch.zeros(t, device=x.device)
    col[0] = bias if bias != float("-inf") else -1e9
    col_bias = col.clone()
    scores = scores + col_bias                        # shifts key 0 for every query
    scores[..., 0, 0] = scores[..., 0, 0] - col_bias[0]   # query 0 has only key 0
    virt = None
    if self.norm == "softpick":
        p = softpick(scores, mask)
    else:
        scores = scores.masked_fill(~mask, float("-inf"))
        if self.sink is not None:
            sink = self.sink.float().view(1, self.h, 1, 1).expand(b, self.h, t, 1)
            full = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)
            p, virt = full[..., :-1], full[..., -1]
        else:
            p = torch.softmax(scores, dim=-1)
    o = p.to(v.dtype) @ v
    aux = {}
    if collect:
        aux["a0"] = p[..., 0].cpu()
        aux["ent"] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).cpu()
        aux["amax"] = p.amax(-1).cpu()
        aux["row_sum"] = p.sum(-1).cpu()
        if virt is not None:
            aux["virt"] = virt.cpu()
    o = o.transpose(1, 2).reshape(b, t, d)
    if self.gate is not None:
        g = torch.sigmoid(self.gate(x))
        o = o * g
        if collect:
            aux["gate"] = g.float().mean(-1).cpu()
            aux["gate_lt_01"] = (g.float() < 0.1).float().mean(-1).cpu()
    return self.out(o), aux


def install_synthetic_bias():
    """Route every SoftmaxAttention through the biased forward in this process."""
    from .layers import SoftmaxAttention
    SoftmaxAttention.forward = biased_softmax_forward
