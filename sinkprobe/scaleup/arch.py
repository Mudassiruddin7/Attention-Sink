"""The size ladder and the designs compared at every rung.

The architecture itself is the one from the small-scale study, so the comparison is
not confounded by an implementation change: layouts cycle over depth, which is what
lets the same design run at 8 and at 20 layers. Two outside baselines are included
because a claim about layer order needs something other than its own family to beat:
an all-attention transformer and an all-linear model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from ..model import ModelConfig, TinyLM
from ..vargap import BoundAttention

# name: (d_model, n_layers, n_heads)
SIZES: Dict[str, Tuple[int, int, int]] = {
    "xs": (128, 8, 4),         # the size the earlier paper used, kept as the anchor
    "s": (256, 8, 4),
    "m": (512, 12, 8),
    "l": (768, 16, 12),
    "xl": (1024, 20, 16),
}

DESIGNS: Dict[str, Dict] = {
    # hybrids: S is a global softmax layer, L a gated delta rule layer
    "global_last":   dict(layout="LLLS", bind="none", rope=False),
    "global_first":  dict(layout="SLLL", bind="none", rope=False),
    "bkf_conv4":     dict(layout="SLLL", bind="conv", width=4, rope=False),
    "bkf_conv16":    dict(layout="SLLL", bind="conv", width=16, rope=False),
    "dyn_first":     dict(layout="SLLL", bind="dyn", width=16, rope=False),
    "dyn_last":      dict(layout="LLLS", bind="dyn", width=16, rope=False),
    # outside baselines
    "all_attention": dict(layout="S", bind="none", rope=True),
    "all_linear":    dict(layout="L", bind="none", rope=False),
}


@dataclass
class ArchConfig:
    size: str = "s"
    design: str = "dyn_first"
    vocab_size: int = 387
    logn_ref: int = 1024           # length the log-length scale is calibrated at

    def to_dict(self) -> Dict:
        return {"size": self.size, "design": self.design, "vocab_size": self.vocab_size,
                "logn_ref": self.logn_ref}


def build(cfg: ArchConfig) -> Tuple[TinyLM, ModelConfig]:
    if cfg.size not in SIZES:
        raise ValueError(f"unknown size {cfg.size}; have {sorted(SIZES)}")
    if cfg.design not in DESIGNS:
        raise ValueError(f"unknown design {cfg.design}; have {sorted(DESIGNS)}")
    d_model, n_layers, n_heads = SIZES[cfg.size]
    spec = DESIGNS[cfg.design]
    mcfg = ModelConfig(vocab_size=cfg.vocab_size, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, d_ff=3 * d_model, layout=spec["layout"],
                       rope=spec["rope"], gate=True, kv_conv=0, logn_ref=cfg.logn_ref)
    model = TinyLM(mcfg)
    if spec["bind"] != "none":
        for blk in model.blocks:
            if blk.kind == "S":
                blk.mix = BoundAttention(d_model, n_heads, rope=spec["rope"], gate=True,
                                         logn_ref=cfg.logn_ref, bind=spec["bind"],
                                         width=spec.get("width", 4))
    return model, mcfg


def describe(cfg: ArchConfig) -> Dict:
    """Parameter count and layer makeup without training anything."""
    model, mcfg = build(cfg)
    kinds = mcfg.kinds()
    from ..vargap import Binder
    binder = sum(p.numel() for m in model.modules() if isinstance(m, Binder)
                 for p in m.parameters())
    return {"size": cfg.size, "design": cfg.design, "params": model.n_params(),
            "d_model": mcfg.d_model, "n_layers": mcfg.n_layers, "n_heads": mcfg.n_heads,
            "global_layers": kinds.count("S"), "linear_layers": kinds.count("L"),
            "layout": mcfg.layout, "binder_params": binder}


def ladder(sizes=("s", "m", "l"), designs=("global_last", "dyn_first")) -> list:
    return [describe(ArchConfig(size=s, design=d)) for s in sizes for d in designs]
