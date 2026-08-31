"""Attention layers used by the SinkProbe pilot models.

Three token mixing layers are provided.

SoftmaxAttention
    Standard causal multi head softmax attention with optional rotary
    position embeddings. This is the baseline that is known to form an
    attention sink on the first position.

SoftmaxAttention with gated=True
    The same layer with an input dependent, channel wise sigmoid gate
    applied to the scaled dot product output, following the gating recipe
    reported by Qiu et al. (NeurIPS 2025). Kimi K3 uses a full rank
    version of this gate on its Gated MLA layers.

KDALikeAttention
    A linear attention layer with a channel wise, lower bounded forget
    gate and a fixed size recurrent state, evaluated with the chunkwise
    parallel form. This is a deliberately simplified stand in for Kimi
    Delta Attention. The delta rule correction term of the real layer is
    not implemented, so the pilot measures the effect of a decaying fixed
    size state rather than of the exact Kimi kernel.

All layers return (output, aux) where aux carries the diagnostic tensors
that metrics.py consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.eps) * self.weight


def build_rope_cache(head_dim: int, max_len: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_len).float()
    freqs = torch.outer(pos, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x has shape (batch, heads, time, head_dim)
    t = x.shape[2]
    cos = cos[:t].view(1, 1, t, -1)
    sin = sin[:t].view(1, 1, t, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


@dataclass
class AttnConfig:
    d_model: int = 128
    n_heads: int = 4
    max_len: int = 4096
    chunk: int = 32
    log_decay_floor: float = -0.25


class SoftmaxAttention(nn.Module):
    """Causal softmax attention. Set gated=True for the SDPA output gate."""

    def __init__(self, cfg: AttnConfig, use_rope: bool = True, gated: bool = False):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.use_rope = use_rope
        self.gated = gated
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if gated:
            # Channel wise, input dependent, full rank gate on the SDPA output.
            self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if use_rope:
            cos, sin = build_rope_cache(self.dh, cfg.max_len)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, collect: bool = False):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.h, self.dh).transpose(1, 2)
        v = v.view(b, t, self.h, self.dh).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)

        aux = {}
        if collect:
            # Attention paid to position zero, averaged over heads and over
            # every query position from index one onwards.
            aux["first_token_attn"] = probs[:, :, 1:, 0].mean().detach()
            p = probs[:, :, 1:, :].clamp_min(1e-12)
            aux["attn_entropy"] = (-(p * p.log()).sum(-1)).mean().detach()
            aux["head_first_attn"] = probs[:, :, 1:, 0].mean(dim=(0, 2)).detach()

        out = out.transpose(1, 2).reshape(b, t, d)
        if self.gated:
            g = torch.sigmoid(self.gate(x))
            if collect:
                aux["gate_score"] = g.mean().detach()
                aux["gate_sparsity"] = (g < 0.01).float().mean().detach()
            out = out * g
        return self.proj(out), aux


class KDALikeAttention(nn.Module):
    """Linear attention with a channel wise lower bounded decay gate.

    The recurrent state has shape (head_dim, head_dim) and does not grow
    with sequence length, which is the property that carries the cost
    argument in the paper. Positional information comes from the decay
    schedule alone, so the layer runs without any position encoding.
    """

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.chunk = cfg.chunk
        self.floor = cfg.log_decay_floor
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.decay = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_norm = RMSNorm(self.dh)
        # The decay is exp(-softplus(x)), so a large positive bias forgets
        # quickly and a large negative bias keeps the state. Channels are
        # spread across that range at the start, from a decay near 0.78 to
        # one near 0.999. The fast channels carry order, since a value that
        # fades tells the layer how long ago something happened, and the slow
        # channels carry content over long spans. Starting every channel at
        # one end removes one of the two abilities and the layer does not
        # recover it during training.
        with torch.no_grad():
            span = torch.linspace(-1.0, -7.0, cfg.d_model)
            self.decay.bias.copy_(span)

    def _split(self, z: torch.Tensor) -> torch.Tensor:
        b, t, _ = z.shape
        return z.view(b, t, self.h, self.dh).transpose(1, 2)

    def forward(self, x: torch.Tensor, collect: bool = False):
        b, t, d = x.shape
        c = self.chunk
        pad = (c - t % c) % c
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        tp = x.shape[1]

        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        q = F.silu(q)
        k = F.normalize(k, dim=-1)

        # Channel wise decay in log space, lower bounded so that the
        # chunkwise form stays numerically stable.
        log_a = -F.softplus(self.decay(x))
        log_a = log_a.clamp(min=self.floor)
        log_a = self._split(log_a)

        nc = tp // c
        shape = (b, self.h, nc, c, self.dh)
        q, k, v, log_a = (z.reshape(*shape) for z in (q, k, v, log_a))

        cum = log_a.cumsum(dim=3)                      # running decay in chunk
        total = cum[:, :, :, -1:, :]                   # decay across a full chunk
        qd = q * cum.exp()
        kd = k * (-cum).exp()
        kt = k * (total - cum).exp()

        causal = torch.ones(c, c, dtype=torch.bool, device=x.device).tril()
        intra = torch.einsum("bhncd,bhnmd->bhncm", qd, kd)
        intra = intra.masked_fill(~causal, 0.0)
        out = torch.einsum("bhncm,bhnmd->bhncd", intra, v)

        # Sequential pass over chunks carries the fixed size state forward.
        state = torch.zeros(b, self.h, self.dh, self.dh, device=x.device, dtype=x.dtype)
        inter = []
        state_norms = []
        for i in range(nc):
            inter.append(torch.einsum("bhcd,bhde->bhce", qd[:, :, i], state))
            upd = torch.einsum("bhcd,bhce->bhde", kt[:, :, i], v[:, :, i])
            # Decay acts on the key axis of the state, broadcast over values.
            decay_chunk = total[:, :, i, 0, :].exp().unsqueeze(-1)
            state = state * decay_chunk + upd
            if collect:
                state_norms.append(state.norm().detach())
        out = out + torch.stack(inter, dim=2)

        out = out.reshape(b, self.h, tp, self.dh)
        out = self.out_norm(out)
        out = out.transpose(1, 2).reshape(b, tp, d)
        g = torch.sigmoid(self.gate(x))
        out = self.proj(out * g)[:, :t]

        aux = {}
        if collect:
            aux["gate_score"] = g.mean().detach()
            aux["mean_decay"] = log_a.exp().mean().detach()
            if state_norms:
                aux["state_norm"] = torch.stack(state_norms).mean()
        return out, aux
