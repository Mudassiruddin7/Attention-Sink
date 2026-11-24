"""How far a released model can look, measured the same way as the small ones.

Training a 3B model from scratch is not on the table here, so the scale objection is
answered from the other end: take hybrid models that are already trained, and measure the
one thing this paper is about, which is how retrieval accuracy depends on distance.

The probe needs no prompt format and no instruction following, which matters because a
base model handed an invented key-value syntax will floor everywhere and say nothing. A
random token sequence is repeated, and at each position in the second copy the model has
to predict the token that followed this one the first time round. Varying the length of
the repeated block varies how far back the match sits, and that is the distance axis:

    lookback d:   r[0] r[1] ... r[d-1] | r[0] r[1] ... r[d-1]
                                         ^ predicting r[i+1] from a cue d positions back

A model whose in-context lookup is a short window copies at small d and fails at large d.
A model with a working attention lookup copies at every d. That is the same shape the
small models are scored on, so the two can be read on one axis.

One thing to know before reading a single row. Accuracy rises with d on a healthy model
rather than staying flat: a short block gives the model fewer tokens in which to notice
that anything is being repeated, so the first few positions of a small block are genuinely
harder. SmolLM2-135M, full attention throughout, measures 0.33 at d=4, 0.87 at d=16 and
0.93 at d=64. The absolute shape is therefore not interpretable on its own, and a row that
climbs is not evidence of anything. What is interpretable is the comparison: a hybrid that
turns over and falls at large d where a full-attention model of the same size keeps
climbing. That is why the controls are not optional.

Two things are reported beside it. The short convolution width is read out of the model's
own configuration rather than assumed, because in these stacks it sits inside the
recurrent block rather than on the keys of an attention layer, and that difference should
be visible to a reader. And a full-attention model of about the same size is probed as a
control, since a profile with no edge is only informative next to one that has one.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# Hybrid stacks whose recurrent layers carry a short causal convolution, and
# full-attention models of about the same size to read them against. All public and
# ungated at the time of writing; anything that moves is skipped with a note rather than
# failing the run.
HYBRIDS = [
    "tiiuae/Falcon-H1-0.5B-Base",
    "tiiuae/Falcon-H1-1.5B-Base",
    "tiiuae/Falcon-H1-3B-Base",
    "ibm-granite/granite-4.0-h-tiny",
    "Zyphra/Zamba2-2.7B",
]
CONTROLS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]

# Powers of two, so the axis is readable on a log scale and spans a short convolution's
# window at one end and several hundred tokens at the other.
DISTANCES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512)

CONV_KEYS = ("conv_kernel", "mamba_d_conv", "d_conv", "conv_bias_size",
             "mamba_conv_kernel", "short_conv_kernel_size", "conv_L_cache")


def conv_width(config) -> Optional[int]:
    """The short-convolution width, read out of the model's own configuration."""
    d = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    for key in CONV_KEYS:
        v = d.get(key)
        if isinstance(v, int) and 1 < v < 64:
            return v
    for sub in ("mamba_config", "ssm_cfg", "text_config"):
        inner = d.get(sub)
        if isinstance(inner, dict):
            for key in CONV_KEYS:
                v = inner.get(key)
                if isinstance(v, int) and 1 < v < 64:
                    return v
    return None


def _safe_token_ids(tokenizer, n: int, rng: np.random.Generator) -> np.ndarray:
    """Ordinary vocabulary ids, with specials and unused slots kept out.

    A random draw over the whole vocabulary picks up reserved and never-trained ids on
    some of these tokenizers, and a model asked to copy those behaves unlike one asked to
    copy text.
    """
    size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    special = set(tokenizer.all_special_ids or [])
    lo, hi = max(10, int(size * 0.02)), int(size * 0.80)
    pool = [i for i in range(lo, hi) if i not in special]
    if len(pool) < n:
        pool = [i for i in range(size) if i not in special]
    return rng.choice(np.asarray(pool, dtype=np.int64), size=n, replace=True)


@torch.no_grad()
def copy_vs_distance(model, tokenizer, distances: Sequence[int] = DISTANCES,
                     n_seq: int = 24, device: str = "cuda", seed: int = 0,
                     max_batch: int = 8, verbose: bool = True) -> Dict:
    """Next-token accuracy on a repeated block, against how far back the match sits."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for d in distances:
        block = np.stack([_safe_token_ids(tokenizer, d, rng) for _ in range(n_seq)])
        seq = np.concatenate([block, block], axis=1)            # (N, 2d)
        hits = total = 0
        for start in range(0, n_seq, max_batch):
            chunk = torch.from_numpy(seq[start:start + max_batch]).to(device)
            logits = model(chunk).logits.float()
            # positions d..2d-2 of the second copy predict block[i+1]; the cue sits d back
            for i in range(d - 1):
                pos = d + i
                pred = logits[:, pos].argmax(-1)
                want = chunk[:, pos + 1]
                hits += int((pred == want).sum())
                total += int(pred.numel())
        acc = hits / max(1, total)
        lo, hi = _wilson(hits, total)
        out[str(d)] = {"accuracy": acc, "n": total, "lo": lo, "hi": hi}
        if verbose:
            print(f"    lookback {d:4d}  copy {acc:.3f} [{lo:.3f},{hi:.3f}] "
                  f"over {total} predictions", flush=True)
    return out


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def edge_of(profile: Dict, drop: float = 0.5) -> Optional[float]:
    """The lookback at which copying falls to half what it was at the shortest one.

    Only meaningful for a profile that falls. A released model's profile rises, so this
    returns None or infinity for all of them and `summarise` is what to read instead. It
    is kept because the small models can produce a falling profile, which is the whole
    point of the comparison.
    """
    keys = sorted(profile, key=lambda s: int(s))
    if not keys:
        return None
    near = profile[keys[0]]["accuracy"]
    if near < 0.2:
        return None                     # nothing to find an edge in
    for k in keys:
        if profile[k]["accuracy"] < near * drop:
            return float(k)
    return float("inf")                 # no edge inside the range tested


def summarise(profile: Dict) -> Dict:
    """Plateau, and how far back you have to go to reach it.

    For a rising profile these are the numbers that separate the two ways a model can be
    worse: a lower ceiling, which is a copying deficit at every distance, and a later
    ceiling, which is a range deficit. A short window predicts the second. Reporting only
    one accuracy, at one distance, cannot tell them apart.
    """
    keys = sorted(profile, key=lambda s: int(s))
    if not keys:
        return {}
    ys = [profile[k]["accuracy"] for k in keys]
    plateau = max(ys[-3:]) if len(ys) >= 3 else max(ys)
    def first_at(frac):
        for k, y in zip(keys, ys):
            if y >= frac * plateau:
                return int(k)
        return None
    turning = None                      # where it stops rising, if it ever does
    for i in range(1, len(ys)):
        if ys[i] < ys[i - 1] - 0.06:    # a real drop, not sampling noise
            turning = int(keys[i])
            break
    return {"plateau": plateau, "d90": first_at(0.90), "d99": first_at(0.99),
            "turns_over_at": turning, "monotone": turning is None}


def probe_one(name: str, device: str = "cuda", distances=DISTANCES, n_seq: int = 24,
              dtype: str = "bfloat16", verbose: bool = True) -> Dict:
    """Load one released model, measure its distance profile, then free it."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    record: Dict = {"model": name, "distances": list(distances), "n_seq": n_seq}
    t0 = time.time()
    try:
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=False)
    except Exception as exc:
        record["skipped"] = f"{type(exc).__name__}: {exc}"
        return record
    record["conv_width"] = conv_width(cfg)
    record["architecture"] = (getattr(cfg, "architectures", None) or ["?"])[0]
    record["n_layers"] = getattr(cfg, "num_hidden_layers", None)
    # The same arithmetic as for the small models: a value sits one past the gap, so a
    # convolution of width w reaches a cue at w-2. Reported, not assumed to apply: in
    # these stacks the convolution sits inside the recurrent block, not on the keys of an
    # attention layer.
    record["predicted_reach"] = (None if record["conv_width"] is None
                                 else record["conv_width"] - 2)
    if verbose:
        print(f"  {name}: {record['architecture']}, {record['n_layers']} layers, "
              f"conv width {record['conv_width']}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=getattr(torch, dtype), device_map=None).to(device).eval()
    except Exception as exc:
        record["skipped"] = f"{type(exc).__name__}: {exc}"
        return record
    record["params"] = sum(p.numel() for p in model.parameters())
    try:
        record["copy_profile"] = copy_vs_distance(model, tok, distances, n_seq,
                                                  device=device, verbose=verbose)
        record["edge"] = edge_of(record["copy_profile"])
        record["summary"] = summarise(record["copy_profile"])
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    record["seconds"] = round(time.time() - t0, 1)
    return record


def probe_all(names: Sequence[str] = tuple(HYBRIDS) + tuple(CONTROLS),
              out_dir: Optional[Path] = None, device: str = "cuda",
              distances=DISTANCES, n_seq: int = 24, store=None,
              verbose: bool = True) -> Dict[str, Dict]:
    """Every model in turn, each written out as it finishes."""
    results: Dict[str, Dict] = {}
    for name in names:
        slug = name.replace("/", "__")
        path = (out_dir / f"released_{slug}.json") if out_dir else None
        if path and path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
            if verbose:
                print(f"  {name}: already measured", flush=True)
            continue
        if verbose:
            print(f"\n=== {name} ===", flush=True)
        rec = probe_one(name, device=device, distances=distances, n_seq=n_seq,
                        verbose=verbose)
        results[name] = rec
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec), encoding="utf-8")
            if store is not None and getattr(store, "enabled", False):
                store.put(path)
        if rec.get("skipped") and verbose:
            print(f"  skipped: {rec['skipped']}", flush=True)
    return results
