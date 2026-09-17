"""Small language models that differ in one mixing rule at a time.

A model is described by a layout string repeated over depth, where S is a
softmax layer and L is a gated delta rule layer, plus a handful of switches.
The named variants below form the ladder used in the paper. Each step of the
main ladder changes exactly one thing relative to the step before it.

    softmax         S layers, rotary positions
    gate            + sigmoid output gate on every softmax layer
    hybrid          three gated delta rule layers per gated softmax layer
    hybrid_nope     + no position encoding on the softmax layers
    hybrid_attnres  + Block Attention Residuals over depth

Side branches change one thing relative to softmax or hybrid.

    sinklogit       learned per head sink logit (gpt-oss)
    softpick        rectified softmax (Zuhri et al., 2025)
    hybrid_nodelta  hybrid with the delta correction removed

Blocks use the usual sequential pre-norm update. A block hands its residual
contribution to the depth mixer, so the standard residual sum and attention
residuals share one interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .layers import GatedDeltaNet, RMSNorm, SoftmaxAttention, SwiGLU
from .residuals import BlockAttentionResidual, StandardResidual


VARIANTS = {
    "softmax": dict(),
    "gate": dict(gate=True),
    "hybrid": dict(layout="LLLS", gate=True),
    "hybrid_nope": dict(layout="LLLS", gate=True, rope=False),
    "hybrid_attnres": dict(layout="LLLS", gate=True, rope=False, depth_mix="attnres"),
    "sinklogit": dict(sink_logit=True),
    "softpick": dict(norm="softpick"),
    "hybrid_nodelta": dict(layout="LLLS", gate=True, delta=False),
    # Single factors added to the softmax baseline. Together with softmax, gate
    # and hybrid these give each mechanism in isolation and a 2x2 of
    # output gate by delta-rule layers.
    "hybrid_nogate": dict(layout="LLLS"),
    "nope": dict(rope=False),
    "attnres": dict(depth_mix="attnres"),
    # Bound-Key Attention: keys and values bound to their local context by a
    # short causal convolution, matching without positional rotation, and a
    # logit scale that keeps the matching margin as the context grows past the
    # training length. bka_rope keeps rotary positions, for the ablation.
    "bka": dict(rope=False, kv_conv=4, logn_ref=256),
    "bka_rope": dict(kv_conv=4, logn_ref=256),
    # Bound keys in the two global layers of the NoPE hybrid. The delta-rule
    # layers carry order in a form that does not depend on context length,
    # which a pure attention stack without positions lacks (bka fails past the
    # training length). With the global layer last in each block, the bound
    # keys neither sped up learning nor made it more reliable.
    "hybrid_bka": dict(layout="LLLS", gate=True, rope=False, kv_conv=4, logn_ref=256),
    # The same with the bound-key layer first in each block, so the retrieval
    # hop matches on token embeddings instead of the output of three delta-rule
    # layers.
    "hybrid_bka_first": dict(layout="SLLL", gate=True, rope=False, kv_conv=4, logn_ref=256),
    # Global layer first without bound keys, to separate placement from binding.
    # The length scaling is the identity at the training length, so training
    # differs from hybrid_bka_first only in the key and value convolutions.
    "hybrid_nope_first": dict(layout="SLLL", gate=True, rope=False, logn_ref=256),
}

MAIN_LADDER = ["softmax", "gate", "hybrid", "hybrid_nope", "hybrid_attnres"]
SIDE_BRANCHES = ["sinklogit", "softpick", "hybrid_nodelta"]
SINGLE_FACTORS = ["gate", "hybrid_nogate", "nope", "attnres"]


@dataclass
class ModelConfig:
    vocab_size: int = 386
    d_model: int = 128
    n_layers: int = 8
    n_heads: int = 4
    d_ff: int = 384
    layout: str = "S"
    rope: bool = True
    qk_norm: bool = True
    gate: bool = False
    sink_logit: bool = False
    norm: str = "softmax"
    delta: bool = True
    conv: bool = True
    chunk: int = 32
    depth_mix: str = "residual"
    block_size: int = 4
    kv_conv: int = 0
    logn_ref: int = 0

    @classmethod
    def from_variant(cls, variant: str, **overrides):
        spec = dict(VARIANTS[variant])
        spec.update(overrides)
        return cls(**spec)

    def kinds(self):
        return [self.layout[i % len(self.layout)] for i in range(self.n_layers)]

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        if kind == "S":
            self.mix = SoftmaxAttention(cfg.d_model, cfg.n_heads, rope=cfg.rope,
                                        qk_norm=cfg.qk_norm, gate=cfg.gate,
                                        sink_logit=cfg.sink_logit, norm=cfg.norm,
                                        kv_conv=cfg.kv_conv, logn_ref=cfg.logn_ref)
        elif kind == "L":
            self.mix = GatedDeltaNet(cfg.d_model, cfg.n_heads, delta=cfg.delta,
                                     conv=cfg.conv, chunk=cfg.chunk)
        else:
            raise ValueError(f"unknown layer kind {kind}")
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor, collect: bool = False):
        a, aux = self.mix(self.n1(x), collect=collect)
        f = self.ffn(self.n2(x + a))
        return a + f, aux


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, k) for k in cfg.kinds()])
        rule = BlockAttentionResidual if cfg.depth_mix == "attnres" else StandardResidual
        self.depth_mix = rule(cfg.d_model, cfg.n_layers, cfg.block_size)
        self.norm_out = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        # Variants whose attention is computed explicitly hold a T x T map.
        self.explicit_attention = cfg.norm != "softmax" or cfg.sink_logit
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, collect: bool = False):
        x = self.embed(idx)
        self.depth_mix.reset(x)
        layers = []
        for i, blk in enumerate(self.blocks):
            h = self.depth_mix.read(i)
            out, aux = blk(h, collect=collect)
            self.depth_mix.write(i, out)
            if collect:
                hf = h.detach().float().abs()
                aux["kind"] = blk.kind
                aux["h_absmax"] = hf.amax(-1).cpu()                 # (B,T)
                aux["h_median"] = hf.median().item()
                layers.append(aux)
        h = self.depth_mix.final()
        stats = {}
        if collect:
            hf = h.detach().float().abs()
            stats = {"layers": layers, "final_absmax": hf.amax(-1).cpu(),
                     "final_median": hf.median().item()}
            w = self.depth_mix.weights()
            stats["depth_weights"] = None if w is None else w.tolist()
        return self.head(self.norm_out(h)), stats

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
