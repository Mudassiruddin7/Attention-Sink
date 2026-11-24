"""What the binder and the global layers actually do, tested by intervention.

Behaviour alone cannot say how a model retrieves. Four tests here, all at test time, all
on the same items so every condition of one model is a paired comparison:

    binder_knockout     leave every weight in place and switch the binding off. If the
                        binding is what carries a key into its value, accuracy inside its
                        window collapses and accuracy beyond it does not move.
    binder_patch        replace what the binder mixes in with the term computed on a
                        different input. Presence is kept and content is destroyed, which
                        separates "the layer needs this information" from "the layer needs
                        something in this slot".
    head_ablation       zero one head of one global layer at a time, reported inside the
                        binder's window and beyond it, so a head that only serves the
                        short-range shortcut is told apart from one that serves the lookup.
    answer_row_attention
                        where each head of each global layer looks when it has to answer:
                        at the value, at the key, or at position zero. Reported against the
                        key-value distance.

Nothing here trains, and nothing here is left behind: every switch is restored in a
finally block, so a model can be scored again afterwards.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..vargap import Binder, wilson
from .corpus import BOS, QRY, SEP, ByteWindows
from .evaluate import item_seed, score, split_by_reach
from .tasks import GAPPED, LongContextTask, study_config
from .train import _autocast


def global_layers(model) -> List[int]:
    return [i for i, blk in enumerate(model.blocks) if blk.kind == "S"]


def binders(model) -> List[Binder]:
    return [m for m in model.modules() if isinstance(m, Binder) and m.mode != "none"]


# --------------------------------------------------------------------- switches
@contextmanager
def binding_off(model):
    """Every binder in the model silenced, then restored."""
    got = binders(model)
    for b in got:
        b.enabled = False
    try:
        yield len(got)
    finally:
        for b in got:
            b.enabled = True


@contextmanager
def heads_zeroed(model, layer: int, heads: Sequence[int]):
    """The output slices of the named heads of one layer set to zero before the
    output projection, which is where a head's contribution is still separable."""
    attn = model.blocks[layer].mix
    dh = attn.dh

    def pre(_module, args):
        (o,) = args
        o = o.clone()
        for h in heads:
            o[..., h * dh:(h + 1) * dh] = 0
        return (o,)

    handle = attn.out.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------- probes
def binder_knockout(model, task: LongContextTask, length: int, n_items: int, device: str,
                    reach: Optional[int], amp: bool = True, seed_tag: str = "knockout") -> Dict:
    """Score with the binding on and off, split at the binder's edge."""
    seed = item_seed(seed_tag, task.cfg.kind, length)
    intact = score(model, task, length, n_items, device, batch=16, amp=amp, seed=seed)
    with binding_off(model) as n_off:
        removed = score(model, task, length, n_items, device, batch=16, amp=amp, seed=seed)
    out = {"binders_silenced": n_off, "reach": reach,
           "intact": {k: intact[k] for k in ("em", "n", "lo", "hi")},
           "removed": {k: removed[k] for k in ("em", "n", "lo", "hi")},
           "intact_by_reach": split_by_reach(intact, reach),
           "removed_by_reach": split_by_reach(removed, reach)}
    a, b = intact.get("hits"), removed.get("hits")
    if a and b and len(a) == len(b):
        out["paired"] = mcnemar(a, b)
        gaps = intact.get("gaps") or []
        if gaps and len(gaps) == len(a) and reach is not None:
            inside = [(x, y) for g, x, y in zip(gaps, a, b) if g <= reach]
            beyond = [(x, y) for g, x, y in zip(gaps, a, b) if g > reach]
            if inside:
                out["paired_in_reach"] = mcnemar([x for x, _ in inside],
                                                 [y for _, y in inside])
            if beyond:
                out["paired_beyond_reach"] = mcnemar([x for x, _ in beyond],
                                                     [y for _, y in beyond])
    return out


@torch.no_grad()
def binder_patch(model, task: LongContextTask, length: int, n_items: int, device: str,
                 reach: Optional[int], amp: bool = True, batch: int = 16,
                 seed_tag: str = "patch") -> Dict:
    """Keep the binder's output in place but compute it from a different input.

    For each batch the items are rolled by one, the binder's added term is captured on
    the rolled tokens, and that term is then injected while the real tokens are scored.
    Shapes match by construction, so nothing about the layer changes except what the
    binding says.
    """
    got = binders(model)
    if not got:
        return {"skipped": "no binder in this design"}
    model.eval()
    rng = np.random.default_rng(item_seed(seed_tag, task.cfg.kind, length))
    hits: List[int] = []
    gaps: List[int] = []
    done = 0
    captured: Dict[int, torch.Tensor] = {}

    originals = [b.mix_term for b in got]

    def capturing(binder, idx):
        def wrapper(z, x, collect=False):
            term, off = originals[idx](z, x, collect=collect)
            captured[idx] = term.detach()
            return term, off
        return wrapper

    try:
        while done < n_items:
            n = min(batch, n_items - done)
            b = task.build(n, length, rng)
            tok = b["tokens"].to(device)
            rolled = torch.roll(tok, shifts=1, dims=0)
            # pass one: capture what the binder makes of a different input
            captured.clear()
            for i, binder in enumerate(got):
                binder.mix_term = capturing(binder, i)
            with _autocast(device, amp):
                model(rolled)
            for i, binder in enumerate(got):
                binder.mix_term = originals[i]
            # pass two: score the real input with that term injected
            for i, binder in enumerate(got):
                binder.patch = captured.get(i)
            with _autocast(device, amp):
                logits, _ = model(tok)
            for binder in got:
                binder.patch = None
            logits = logits.float()
            logits[..., [BOS, QRY, SEP]] = -float("inf")
            pos = b["answer_pos"].to(device)
            pred = logits.gather(1, pos[..., None].expand(-1, -1, logits.size(-1))).argmax(-1)
            want = tok.gather(1, pos + 1)
            hits.extend(int(x) for x in (pred == want).flatten().tolist())
            gaps.extend(int(g) for g in b["answer_gap"].flatten().tolist())
            done += n
    finally:
        for i, binder in enumerate(got):
            binder.mix_term = originals[i]
            binder.patch = None
        model.train()

    k, n = int(sum(hits)), len(hits)
    lo, hi = wilson(k, n)
    res = {"em": k / max(1, n), "n": n, "lo": lo, "hi": hi, "hits": hits, "gaps": gaps}
    return {"patched": {kk: res[kk] for kk in ("em", "n", "lo", "hi")},
            "patched_by_reach": split_by_reach(res, reach), "reach": reach}


def head_ablation(model, task: LongContextTask, length: int, n_items: int, device: str,
                  reach: Optional[int], amp: bool = True, seed_tag: str = "heads") -> Dict:
    """Each head of each global layer removed on its own, scored inside and beyond reach."""
    seed = item_seed(seed_tag, task.cfg.kind, length)
    rows = []
    base = score(model, task, length, n_items, device, batch=16, amp=amp, seed=seed)
    rows.append({"layer": None, "head": None, "label": "intact", "em": base["em"],
                 "n": base["n"], "by_reach": split_by_reach(base, reach)})
    for i in global_layers(model):
        n_heads = model.blocks[i].mix.h
        for h in list(range(n_heads)) + [None]:          # None removes the whole layer
            use = list(range(n_heads)) if h is None else [h]
            with heads_zeroed(model, i, use):
                r = score(model, task, length, n_items, device, batch=16, amp=amp,
                          seed=seed)
            rows.append({"layer": i, "head": h,
                         "label": f"L{i}" if h is None else f"L{i}h{h}",
                         "em": r["em"], "n": r["n"],
                         "by_reach": split_by_reach(r, reach)})
    return {"reach": reach, "conditions": rows}


@torch.no_grad()
def answer_row_attention(model, task: LongContextTask, length: int, n_items: int,
                         device: str, amp: bool = True, batch: int = 8,
                         seed_tag: str = "rows") -> Dict:
    """Where each head of each global layer looks on the rows that must answer.

    Only the answer rows are built, so no T x T map is ever held. For each row the mass
    is read at the value of the queried pair, at the queried key itself, and at position
    zero, and is reported against the key-value distance.
    """
    if task.cfg.kind not in GAPPED:
        return {"skipped": f"{task.cfg.kind} has no key-value distance"}
    model.eval()
    rng = np.random.default_rng(item_seed(seed_tag, task.cfg.kind, length))
    layers = global_layers(model)
    if not layers:
        return {"skipped": "no global layer in this design"}

    seen: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for i in layers:
        def pre(_m, args, i=i):
            captured[i] = args[0].detach()
        handles.append(model.blocks[i].mix.register_forward_pre_hook(pre))

    done = 0
    try:
        while done < n_items:
            n = min(batch, n_items - done)
            b = task.build(n, length, rng)
            tok = b["tokens"].to(device)
            captured.clear()
            with _autocast(device, amp):
                model(tok)
            ap = b["answer_pos"].to(device)                       # (B, Q)
            kp = b["key_pos"].to(device)
            vp = b["value_pos"].to(device)
            gaps = b["answer_gap"]
            for i in layers:
                x = captured[i]
                proj = model.blocks[i].mix.project(x)
                q, k = proj[0].float(), proj[1].float()           # (B, H, T, dh)
                bsz, heads, t, dh = q.shape
                rows = q.gather(2, ap[:, None, :, None].expand(-1, heads, -1, dh))
                scores = torch.einsum("bhqd,bhtd->bhqt", rows, k) / math.sqrt(dh)
                allowed = (torch.arange(t, device=device)[None, None, None, :]
                           <= ap[:, None, :, None])
                p = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
                m_val = p.gather(3, vp[:, None, :, None].expand(-1, heads, -1, -1))
                m_key = p.gather(3, kp[:, None, :, None].expand(-1, heads, -1, -1))
                m_zero = p[..., :1].expand(-1, -1, p.size(2), -1)
                top = p.argmax(-1)
                for h in range(heads):
                    for bi in range(bsz):
                        for qi in range(ap.size(1)):
                            g = int(gaps[bi, qi])
                            rec = seen.setdefault((i, h), {})
                            rec.setdefault("gap", []).append(g)
                            rec.setdefault("value", []).append(float(m_val[bi, h, qi, 0]))
                            rec.setdefault("key", []).append(float(m_key[bi, h, qi, 0]))
                            rec.setdefault("sink", []).append(float(m_zero[bi, h, qi, 0]))
                            rec.setdefault("top_is_value", []).append(
                                float(int(top[bi, h, qi]) == int(vp[bi, qi])))
            done += n
    finally:
        for hd in handles:
            hd.remove()
        model.train()

    out = {}
    for (i, h), rec in sorted(seen.items()):
        gap = np.asarray(rec["gap"])
        entry = {"layer": i, "head": h, "n": int(gap.size)}
        for field in ("value", "key", "sink", "top_is_value"):
            arr = np.asarray(rec[field], dtype=float)
            entry[field] = float(arr.mean())
            entry[f"{field}_by_gap"] = {int(g): float(arr[gap == g].mean())
                                        for g in sorted(set(gap.tolist()))}
        out[f"L{i}h{h}"] = entry
    return {"heads": out, "n_items": done}


@torch.no_grad()
def prefix_matching(model, device: str, length: int = 128, n_seq: int = 16,
                    vocab_lo: int = 1, vocab_hi: int = 255,
                    amp: bool = True) -> Dict:
    """The prefix-matching score of every head, measured directly.

    A long-range copying score was used as a stand-in for induction-head formation in the
    earlier study, but copying is exactly what a recurrent state can do without an
    induction head, and these stacks are mostly recurrent, so the stand-in is leaky in
    the architecture being studied. This measures the thing itself: on a random sequence
    repeated twice, a head at position t in the second copy should attend to t-L+1, the
    token that followed this one the first time round.
    """
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(31337)
    half = length // 2
    toks = torch.randint(vocab_lo, vocab_hi, (n_seq, half), generator=g)
    seq = torch.cat([toks, toks], dim=1).to(device)           # (N, 2*half)
    layers = global_layers(model)
    if not layers:
        return {"skipped": "no global layer in this design"}

    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for i in layers:
        def pre(_m, args, i=i):
            captured[i] = args[0].detach()
        handles.append(model.blocks[i].mix.register_forward_pre_hook(pre))
    try:
        with _autocast(device, amp):
            model(seq)
        t = seq.size(1)
        # query positions inside the second copy, skipping the first one, whose target
        # would be the very start of the sequence
        qs = torch.arange(half + 1, t, device=device)
        targets = qs - half + 1
        out = {}
        for i in layers:
            proj = model.blocks[i].mix.project(captured[i])
            q, k = proj[0].float(), proj[1].float()
            n, h, _, dh = q.shape
            rows = q[:, :, qs, :]                              # (N, H, Q, dh)
            scores = torch.einsum("bhqd,bhtd->bhqt", rows, k) / math.sqrt(dh)
            allowed = (torch.arange(t, device=device)[None, None, None, :]
                       <= qs[None, None, :, None])
            p = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
            mass = p.gather(3, targets[None, None, :, None]
                            .expand(n, h, -1, -1)).squeeze(-1)  # (N, H, Q)
            top = (p.argmax(-1) == targets[None, None, :]).float()
            for head in range(h):
                out[f"L{i}h{head}"] = {
                    "layer": i, "head": head,
                    "prefix_matching": float(mass[:, head].mean()),
                    "top_is_induction": float(top[:, head].mean())}
    finally:
        for hd in handles:
            hd.remove()
        model.train()
    best = max(out.values(), key=lambda v: v["prefix_matching"]) if out else {}
    return {"heads": out, "length": length, "n_seq": n_seq,
            "best_head_score": best.get("prefix_matching"),
            "n_heads_above_0.2": sum(1 for v in out.values()
                                     if v["prefix_matching"] > 0.2)}


# --------------------------------------------------------------------- stats
def mcnemar(a: Sequence[int], b: Sequence[int]) -> Dict:
    """Exact two-sided McNemar test on paired 0/1 outcomes from the same items."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return {"p": 1.0, "n_discordant": 0, "a_only": 0, "b_only": 0,
                "diff": 0.0}
    kk = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(kk + 1)) / 2 ** n)
    return {"p": p, "n_discordant": n, "a_only": b10, "b_only": b01,
            "diff": (sum(b) - sum(a)) / max(1, len(a))}


def probe_all(model, windows: ByteWindows, kind: str, length: int, device: str,
              reach: Optional[int], n_items: int = 256, rows_items: int = 32,
              gap_max: int = 16, amp: bool = True, verbose: bool = True) -> Dict:
    """Every probe above, on one model, at one length."""
    task = LongContextTask(study_config(kind, gap_max=gap_max), windows)
    if task.required_length() > length:
        return {"skipped": f"{kind} needs {task.required_length()} tokens"}
    out = {"kind": kind, "length": length, "reach": reach}
    tag = "probe"                    # one item sample for every condition below
    if verbose:
        print("  binder knockout", flush=True)
    out["binder_knockout"] = binder_knockout(model, task, length, n_items, device, reach,
                                             amp=amp, seed_tag=tag)
    if verbose:
        print("  binder patch", flush=True)
    out["binder_patch"] = binder_patch(model, task, length, n_items, device, reach,
                                       amp=amp, seed_tag=tag)
    if verbose:
        print("  head ablation", flush=True)
    out["head_ablation"] = head_ablation(model, task, length, n_items, device, reach,
                                         amp=amp, seed_tag=tag)
    if verbose:
        print(f"  answer-row attention on {rows_items} items", flush=True)
    out["answer_row_attention"] = answer_row_attention(model, task, length, rows_items,
                                                       device, amp=amp)
    if verbose:
        print("  prefix matching", flush=True)
    out["prefix_matching"] = prefix_matching(model, device, amp=amp)
    return out
