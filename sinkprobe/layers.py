"""Token mixing layers for the SinkProbe testbed.

Softmax attention comes in forms that differ in exactly one place.

    plain      softmax over causal scores
    gate       elementwise sigmoid gate on the attention output, computed from
               the layer input (the G1 gate of Qiu et al., 2025, used in
               Qwen3-Next and Qwen3.5)
    sink       one learned logit per head that competes inside the softmax and
               is then dropped, so a head can park mass somewhere that is not a
               token (the attention sink parameter of gpt-oss)
    softpick   rectified softmax whose rows need not sum to one
               (Zuhri et al., 2025)

The linear layer is the gated delta rule (Yang et al., 2025) with a short
causal convolution and a gated output norm. This is the linear layer of
Qwen3-Next and Qwen3.5, and the scalar decay case of Kimi Delta Attention.
With delta=False the delta correction is removed and what is left is a
decaying fixed size state, which is the layer used in our first pilot.

Every layer returns (output, aux). aux is empty unless collect=True, in which
case it carries per sequence, per head, per query tensors on the CPU that
sinkprobe.metrics turns into diagnostics.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


@lru_cache(maxsize=32)
def _rope_tables(t: int, dh: int, base: float, device: str):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
    ang = torch.arange(t, dtype=torch.float32)[:, None] * inv[None, :]
    return ang.cos().to(device), ang.sin().to(device)


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary embedding in the rotate-half layout. x is (B, H, T, dh)."""
    t, dh = x.shape[-2], x.shape[-1]
    cos, sin = _rope_tables(t, dh, base, str(x.device))
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def softpick(scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """Rectified softmax, relu(e^s - 1) / (sum |e^s - 1| + eps).

    Written with the row maximum factored out so it does not overflow.
    Masked positions contribute nothing to either sum.
    """
    s = scores.masked_fill(~mask, 0.0)
    m = s.amax(dim=-1, keepdim=True).clamp_min(0.0)
    e = (torch.exp(s - m) - torch.exp(-m)).masked_fill(~mask, 0.0)
    return F.relu(e) / (e.abs().sum(dim=-1, keepdim=True) + eps * torch.exp(-m))


# Evaluation-time switch for the length-scaling ablation. Training at or below
# the reference length never scales, so turning this off changes nothing there.
LOGN = {"enabled": True}


class SoftmaxAttention(nn.Module):
    """Causal attention.

    kv_conv > 0 binds each key and value to its local context with a residual
    depthwise causal convolution of that width (Bound-Key Attention). With
    logn_ref > 0, the logits of query t are multiplied by
    max(1, log(t + 1) / log(logn_ref)), so the margin needed to single out one
    key among t + 1 does not shrink as the context grows past logn_ref.
    """

    def __init__(self, d_model: int, n_heads: int, rope: bool = True,
                 qk_norm: bool = True, gate: bool = False,
                 sink_logit: bool = False, norm: str = "softmax",
                 kv_conv: int = 0, logn_ref: int = 0):
        super().__init__()
        assert norm in ("softmax", "softpick")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.rope = rope
        self.norm = norm
        self.logn_ref = logn_ref
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.dh) if qk_norm else None
        self.k_norm = RMSNorm(self.dh) if qk_norm else None
        self.gate = nn.Linear(d_model, d_model, bias=False) if gate else None
        self.sink = nn.Parameter(torch.zeros(n_heads)) if sink_logit else None
        self.k_conv = (nn.Conv1d(d_model, d_model, kv_conv, groups=d_model,
                                 padding=kv_conv - 1, bias=False) if kv_conv else None)
        self.v_conv = (nn.Conv1d(d_model, d_model, kv_conv, groups=d_model,
                                 padding=kv_conv - 1, bias=False) if kv_conv else None)

    def project(self, x: torch.Tensor):
        """Queries, keys and values as (B, H, T, dh), ready for the attention map."""
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        if self.k_conv is not None:
            k = k + self.k_conv(k.transpose(1, 2))[..., :t].transpose(1, 2)
            v = v + self.v_conv(v.transpose(1, 2))[..., :t].transpose(1, 2)
        q, k, v = (z.reshape(b, t, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            q, k = apply_rope(q), apply_rope(k)
        if self.logn_ref and LOGN["enabled"] and t > self.logn_ref:
            visible = torch.arange(1, t + 1, device=x.device, dtype=torch.float32)
            lam = torch.clamp(visible.log() / math.log(self.logn_ref), min=1.0)
            q = q * lam.to(q.dtype).view(1, 1, t, 1)
        return q, k, v

    def forward(self, x: torch.Tensor, collect: bool = False):
        b, t, d = x.shape
        q, k, v = self.project(x)

        aux = {}
        fast = (not collect) and self.norm == "softmax" and self.sink is None
        if fast:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            if self.norm == "softpick":
                p = softpick(scores.float(), mask).to(v.dtype)
                virt = None
            else:
                scores = scores.float().masked_fill(~mask, float("-inf"))
                if self.sink is not None:
                    col = self.sink.float().view(1, self.h, 1, 1).expand(b, self.h, t, 1)
                    full = torch.softmax(torch.cat([scores, col], dim=-1), dim=-1)
                    p, virt = full[..., :-1], full[..., -1]
                else:
                    p, virt = torch.softmax(scores, dim=-1), None
                p = p.to(v.dtype)
            o = p @ v
            if collect:
                pf = p.float()
                aux["a0"] = pf[..., 0].cpu()                                # (B,H,T)
                ent = -(pf * pf.clamp_min(1e-12).log()).sum(-1)
                aux["ent"] = ent.cpu()                                     # (B,H,T)
                aux["amax"] = pf.amax(-1).cpu()                            # (B,H,T)
                aux["row_sum"] = pf.sum(-1).cpu()                          # (B,H,T)
                if virt is not None:
                    aux["virt"] = virt.float().cpu()                       # (B,H,T)

        o = o.transpose(1, 2).reshape(b, t, d)
        if self.gate is not None:
            g = torch.sigmoid(self.gate(x))
            o = o * g
            if collect:
                aux["gate"] = g.float().mean(-1).cpu()                     # (B,T)
                aux["gate_lt_01"] = (g.float() < 0.1).float().mean(-1).cpu()
        return self.out(o), aux


# ---------------------------------------------------------------------------
# Gated delta rule
# ---------------------------------------------------------------------------

def gated_delta_rule_recurrent(q, k, v, beta, log_alpha, delta: bool = True,
                               state=None):
    """Reference recurrence, one token at a time. Used by the tests.

    S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T,   o_t = S_t^T q_t
    with S in R^{dk x dv}. Without the delta rule the update is
    S_t = a_t S_{t-1} + b_t k_t v_t^T.
    """
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    s = q.new_zeros(b, h, dk, dv) if state is None else state
    outs = []
    for i in range(t):
        s = s * log_alpha[:, :, i].exp()[..., None, None]
        kt, vt, bt = k[:, :, i], v[:, :, i], beta[:, :, i][..., None]
        if delta:
            mem = torch.einsum("bhd,bhde->bhe", kt, s)
            upd = bt * (vt - mem)
        else:
            upd = bt * vt
        s = s + kt[..., :, None] * upd[..., None, :]
        outs.append(torch.einsum("bhd,bhde->bhe", q[:, :, i], s))
    return torch.stack(outs, dim=2), s


def gated_delta_rule_chunked(q, k, v, beta, log_alpha, chunk: int = 32,
                             delta: bool = True, state=None):
    """Exact chunkwise form of gated_delta_rule_recurrent.

    Inside a chunk with cumulative log decay G, the written vectors
    d_t = b_t (v_t - a_t S_{t-1}^T k_t) solve a unit lower triangular system
    (I + L) D = diag(b) (V - R), where L_tj = b_t exp(G_t - G_j) k_t.k_j for
    j < t and R_t = exp(G_t) S_0^T k_t. Outputs and the carried state follow
    in closed form. Decay enters only through differences of cumulative sums,
    so nothing overflows and no lower bound on the decay is needed.
    """
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    c = chunk
    pad = (-t) % c
    if pad:
        # Padding writes nothing (beta 0) and does not decay (log alpha 0).
        q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v))
        beta, log_alpha = F.pad(beta, (0, pad)), F.pad(log_alpha, (0, pad))
    n = (t + pad) // c
    q = q.reshape(b, h, n, c, dk)
    k = k.reshape(b, h, n, c, dk)
    v = v.reshape(b, h, n, c, dv)
    beta = beta.reshape(b, h, n, c)
    G = log_alpha.reshape(b, h, n, c).cumsum(-1)

    incl = torch.ones(c, c, dtype=torch.bool, device=q.device).tril()
    decay = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()
    W = (q @ k.transpose(-1, -2)) * decay                                   # j <= t
    if delta:
        strict = incl & ~torch.eye(c, dtype=torch.bool, device=q.device)
        L = beta[..., :, None] * (k @ k.transpose(-1, -2)) * decay * strict
        eye = torch.eye(c, dtype=q.dtype, device=q.device)
        # One batched triangular solve for every chunk at once, so the loop
        # below is matrix products only.
        A_inv = torch.linalg.solve_triangular(L + eye, eye.expand_as(L).contiguous(),
                                              upper=False, unitriangular=True)
    eG = G.exp()                                                            # exp(G_t)
    eGC = (G[..., -1:] - G).exp()                                           # exp(G_C - G_j)

    s = q.new_zeros(b, h, dk, dv) if state is None else state
    outs = []
    for i in range(n):
        ki, vi, bi, egi = k[:, :, i], v[:, :, i], beta[:, :, i], eG[:, :, i]
        if delta:
            r = egi[..., None] * (ki @ s)                                   # (B,H,C,dv)
            D = A_inv[:, :, i] @ (bi[..., None] * (vi - r))
        else:
            D = bi[..., None] * vi
        outs.append(egi[..., None] * (q[:, :, i] @ s) + W[:, :, i] @ D)
        s = G[:, :, i, -1].exp()[..., None, None] * s \
            + (ki * eGC[:, :, i][..., None]).transpose(-1, -2) @ D
    o = torch.stack(outs, dim=2).reshape(b, h, n * c, dv)[:, :, :t]
    return o, s


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int, delta: bool = True,
                 conv: bool = True, conv_kernel: int = 4, chunk: int = 32):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.delta = delta
        self.chunk = chunk
        self.qkvz = nn.Linear(d_model, 4 * d_model, bias=False)
        self.ba = nn.Linear(d_model, 2 * n_heads, bias=True)
        self.conv = (nn.Conv1d(3 * d_model, 3 * d_model, conv_kernel,
                               groups=3 * d_model, padding=conv_kernel - 1,
                               bias=False) if conv else None)
        # Per head decay rates spread over two orders of magnitude at the
        # start, following the Mamba2 and Gated DeltaNet initialisation.
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(1.0, 16.0).log())
        dt = torch.exp(torch.empty(n_heads).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm = RMSNorm(self.dh)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, collect: bool = False):
        b, t, d = x.shape
        q, k, v, z = self.qkvz(x).split(d, dim=-1)
        if self.conv is not None:
            qkv = self.conv(torch.cat([q, k, v], -1).transpose(1, 2))[..., :t]
            q, k, v = F.silu(qkv.transpose(1, 2)).split(d, dim=-1)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)

        def heads(zz):
            return zz.view(b, t, self.h, self.dh).transpose(1, 2)

        q = F.normalize(heads(q).float(), dim=-1) / math.sqrt(self.dh)
        k = F.normalize(heads(k).float(), dim=-1)
        v = heads(v).float()
        bl, al = self.ba(x).float().split(self.h, dim=-1)
        beta = torch.sigmoid(bl).transpose(1, 2)                              # (B,H,T)
        log_alpha = (-self.A_log.float().exp() * F.softplus(al + self.dt_bias.float())).transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            o, _ = gated_delta_rule_chunked(q, k, v, beta, log_alpha,
                                            self.chunk, self.delta)
        o = self.o_norm(o.to(x.dtype)).transpose(1, 2).reshape(b, t, d)
        o = o * F.silu(z)
        aux = {}
        if collect:
            aux["beta"] = beta.mean(1).cpu()                                   # (B,T)
            aux["alpha"] = log_alpha.exp().mean(1).cpu()                       # (B,T)
        return self.out(o), aux
