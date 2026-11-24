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
    # The 2x2 of layer order by binding. Only the diagonal was run before, so placement
    # and binding could not be told apart; the two off-diagonal cells are what separate
    # them.
    "global_last":   dict(layout="LLLS", bind="none", rope=False),   # neither
    "global_first":  dict(layout="SLLL", bind="none", rope=False),   # order only
    "bkl_conv4":     dict(layout="LLLS", bind="conv", width=4, rope=False),   # binding only
    "bkf_conv4":     dict(layout="SLLL", bind="conv", width=4, rope=False),   # both
    # The width ladder. A width-w causal convolution read at the value position reaches
    # w-1 tokens back, so if binding works by carrying the key into the value vector its
    # effect has to stop at a distance set by w and by nothing else. Widths 2 and 3 cannot
    # reach any key at all, since the value sits at least two positions after it.
    "bkf_conv2":     dict(layout="SLLL", bind="conv", width=2, rope=False),
    "bkf_conv3":     dict(layout="SLLL", bind="conv", width=3, rope=False),
    "bkf_conv6":     dict(layout="SLLL", bind="conv", width=6, rope=False),
    "bkf_conv8":     dict(layout="SLLL", bind="conv", width=8, rope=False),
    "bkf_conv16":    dict(layout="SLLL", bind="conv", width=16, rope=False),
    "bkf_conv32":    dict(layout="SLLL", bind="conv", width=32, rope=False),
    # Content-chosen spans, kept so the earlier comparison can be repeated.
    "dyn_first":     dict(layout="SLLL", bind="dyn", width=16, rope=False),
    "dyn_last":      dict(layout="LLLS", bind="dyn", width=16, rope=False),
    # Fixed taps at several offsets: a window with holes in it, which separates "reaches
    # far" from "reaches everything in between".
    "multi_first":   dict(layout="SLLL", bind="multi", rope=False),
    # outside baselines
    "all_attention": dict(layout="S", bind="none", rope=True),
    "all_linear":    dict(layout="L", bind="none", rope=False),
}

# How the task lays a pair out, which is what turns a width into a distance.
# A block is [key] [filler x gap] [value], so the value sits gap+1 after its key.
VALUE_OFFSET = 1
MULTI_OFFSETS = (1, 2, 4, 8, 16)         # the fixed taps of the "multi" binder


def binder_reach(design: str, value_offset: int = VALUE_OFFSET):
    """Largest key-value gap the binder can bridge in one step, or None if it has none.

    The binder is read at the value position and mixes in the w-1 positions before it.
    The key is inside that window when gap + value_offset <= w - 1, that is
    gap <= w - 1 - value_offset. A reach below 1 means the window never holds a key at
    all, which makes that design the control for "a convolution helps for some other
    reason".
    """
    spec = DESIGNS.get(design)
    if spec is None or spec["bind"] == "none":
        return None
    if spec["bind"] == "multi":
        return max(MULTI_OFFSETS) - value_offset
    return spec.get("width", 4) - 1 - value_offset


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
