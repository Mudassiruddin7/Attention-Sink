"""Needle in a haystack harness for a released checkpoint.

This is the part of SinkProbe that implements the protocol registered in
the paper. It is separated from the pilot code because it needs model
weights, which the pilot does not.

Two modes are supported.

local
    A Hugging Face causal language model loaded with
    output_attentions=True. Sink mass and massive activation are read
    directly from the attention maps and hidden states.

api
    A served endpoint that returns text only. Attention maps are not
    available, so only position resolved recall and the recency gap can
    be measured. The suite reports which metrics are missing rather than
    filling them in.

Nothing here runs during the pilot. It is included so that the full scale
protocol is code rather than prose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .metrics import wilson_interval

FILLER = (
    "The city council met again on Tuesday to review the drainage plan. "
    "Rain had been steady all week and the lower streets flooded twice. "
    "A survey of the older pipes was ordered before any work began. "
)

NEEDLE = ("The maintenance code for the north pumping station is {code}. ")

QUESTION = ("What is the maintenance code for the north pumping station? "
            "Answer with the code only.")


@dataclass
class ProtocolConfig:
    """The measurement grid registered in the paper."""
    lengths: List[int] = field(default_factory=lambda: [
        4_096, 32_768, 131_072, 262_144, 524_288, 786_432, 1_048_576])
    depths: List[float] = field(default_factory=lambda: [
        i / 10 for i in range(11)])
    trials_per_depth: int = 196          # gives about a 7 point half width
    seed: int = 0

    def total_sequences(self) -> int:
        return len(self.depths) * self.trials_per_depth

    def tokens_at(self, length: int) -> int:
        return self.total_sequences() * length


def make_code(rng: random.Random) -> str:
    """An arbitrary code that cannot be answered from stored knowledge."""
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
                   for _ in range(8))


def build_context(length_tokens: int, depth: float, rng: random.Random,
                  chars_per_token: float = 4.0):
    """Return (prompt, answer) with the needle planted at the given depth."""
    code = make_code(rng)
    needle = NEEDLE.format(code=code)
    target_chars = int(length_tokens * chars_per_token)
    reps = max(1, target_chars // len(FILLER))
    haystack = FILLER * reps
    cut = int(len(haystack) * min(max(depth, 0.0), 1.0))
    cut = min(cut, len(haystack))
    prompt = haystack[:cut] + needle + haystack[cut:] + "\n\n" + QUESTION
    return prompt, code


def run_protocol(generate: Callable[[str], str], cfg: ProtocolConfig,
                 lengths: Optional[List[int]] = None) -> Dict:
    """Sweep the grid with a caller supplied generate function.

    generate takes a prompt and returns the model's text answer. Keeping
    it as a callback means the same harness drives a local model, a served
    endpoint or a stub used in tests.
    """
    rng = random.Random(cfg.seed)
    out = {"config": cfg.__dict__, "rows": []}
    for length in (lengths or cfg.lengths):
        for depth in cfg.depths:
            hits = 0
            for _ in range(cfg.trials_per_depth):
                prompt, code = build_context(length, depth, rng)
                answer = generate(prompt)
                hits += int(code.lower() in answer.lower())
            n = cfg.trials_per_depth
            lo, hi = wilson_interval(hits, n)
            out["rows"].append({"length": length, "depth": round(depth, 2),
                                "hits": hits, "n": n, "acc": hits / n,
                                "ci_low": lo, "ci_high": hi})
    return out


def recency_gap(rows: List[Dict], length: int) -> float:
    """Late quarter recall minus early quarter recall at one length."""
    sel = [r for r in rows if r["length"] == length]
    early = [r["acc"] for r in sel if r["depth"] <= 0.25]
    late = [r["acc"] for r in sel if r["depth"] >= 0.75]
    if not early or not late:
        return float("nan")
    return sum(late) / len(late) - sum(early) / len(early)


def sink_mass_from_attentions(attentions, skip_first_query: bool = True):
    """Mean attention on position zero from a Hugging Face attentions tuple.

    attentions is a tuple with one tensor per layer, each shaped
    (batch, heads, queries, keys). Layers that return None, which is what
    a linear attention layer does, are skipped and counted separately.
    """
    import torch
    used, total = 0, 0.0
    per_layer = []
    for att in attentions:
        if att is None:
            continue
        start = 1 if skip_first_query else 0
        val = att[:, :, start:, 0].mean().item()
        per_layer.append(val)
        total += val
        used += 1
    if used == 0:
        return {"sink_mass": float("nan"), "layers_used": 0,
                "per_layer": []}
    return {"sink_mass": total / used, "layers_used": used,
            "per_layer": per_layer,
            "sink_mass_max_layer": max(per_layer)}


def massive_activation_from_hidden(hidden_states):
    """Mean over layers of the largest absolute hidden state value."""
    vals = [h.abs().max().item() for h in hidden_states if h is not None]
    if not vals:
        return {"max_activation": float("nan"), "max_activation_peak": float("nan")}
    return {"max_activation": sum(vals) / len(vals),
            "max_activation_peak": max(vals)}
