"""Scoring, with the uncertainty attached.

Items are generated from a seed fixed by task and length, so every design is scored
on identical inputs and differences can be tested pairwise instead of being eyeballed
across separate samples. Per-item outcomes are kept in the run record, which is what
makes a paired bootstrap possible after the fact.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..vargap import holm, paired_sign_test, t_interval, wilson
from .corpus import BOS, QRY, SEP, ByteWindows
from .tasks import LongContextTask, study_config
from .train import _autocast, bits_per_byte

ITEM_SEED = 90_000          # same items for every design, so comparisons are paired


@torch.no_grad()
def score(model, task: LongContextTask, length: int, n_items: int, device: str,
          batch: int = 32, amp: bool = True, seed: Optional[int] = None) -> Dict:
    """Exact match with a Wilson interval, a breakdown by distance, and per-item hits."""
    model.eval()
    rng = np.random.default_rng(ITEM_SEED + (seed if seed is not None else 0))
    hits: List[int] = []
    gaps: List[int] = []
    done = 0
    while done < n_items:
        n = min(batch, n_items - done)
        b = task.build(n, length, rng)
        tok = b["tokens"].to(device)
        with _autocast(device, amp):
            logits, _ = model(tok)
        logits = logits.float()
        logits[..., [BOS, QRY, SEP]] = -float("inf")        # markers are never an answer
        pos = b["answer_pos"].to(device)
        pred = logits.gather(1, pos[..., None].expand(-1, -1, logits.size(-1))).argmax(-1)
        want = tok.gather(1, pos + 1)
        ok = (pred == want).flatten().tolist()
        hits.extend(int(x) for x in ok)
        gaps.extend(int(g) for g in b["answer_gap"].flatten().tolist())
        done += n
    model.train()
    n = len(hits)
    k = int(sum(hits))
    lo, hi = wilson(k, n)
    by_gap: Dict[int, Dict] = {}
    for g, h in zip(gaps, hits):
        a, c = by_gap.get(g, (0, 0))
        by_gap[g] = (a + h, c + 1)
    return {"em": k / max(1, n), "n": n, "lo": lo, "hi": hi, "chance": task.chance(),
            "by_gap": {int(g): {"em": a / c, "n": c} for g, (a, c) in sorted(by_gap.items())},
            "hits": hits}


def evaluate_suite(model, windows: ByteWindows, kinds: Sequence[str], lengths: Sequence[int],
                   n_items: int, device: str, gap_max: int = 16, amp: bool = True,
                   verbose: bool = True) -> Dict:
    """Every task at every length it fits in, plus language modelling cost."""
    out: Dict[str, Dict] = {}
    for kind in kinds:
        task = LongContextTask(study_config(kind, gap_max=gap_max), windows)
        need = task.required_length()
        per_length = {}
        for L in lengths:
            if L < need:
                continue
            per_length[str(L)] = score(model, task, L, n_items, device, amp=amp,
                                       seed=hash((kind, L)) % 10_000)
            if verbose:
                r = per_length[str(L)]
                print(f"  {kind:12s} len {L:6d} EM {r['em']:.3f} "
                      f"[{r['lo']:.3f}, {r['hi']:.3f}] chance {r['chance']:.3f}", flush=True)
        out[kind] = {"required_length": need, "lengths": per_length}
    out["language_model"] = {
        "lengths": {str(L): {"bits_per_byte": round(
            bits_per_byte(model, windows, L, 32, device, amp=amp,
                          rng=np.random.default_rng(4242)), 4)} for L in lengths}}
    return out


def paired_bootstrap(a_hits: Sequence[int], b_hits: Sequence[int], n_boot: int = 2000,
                     seed: int = 0) -> Dict:
    """Interval for b minus a on the same items, which an unpaired test would widen."""
    a = np.asarray(a_hits, dtype=float)
    b = np.asarray(b_hits, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = (b[idx] - a[idx]).mean(axis=1)
    return {"diff": float((b - a).mean()), "lo": float(np.quantile(draws, 0.025)),
            "hi": float(np.quantile(draws, 0.975)), "n": int(n),
            "wins": int(((b - a) > 0).sum()), "losses": int(((b - a) < 0).sum())}


def compare(records: Dict[str, Dict], baseline: str, candidate: str, kind: str,
            length: int) -> Optional[Dict]:
    """Paired comparison of two designs on one task at one length."""
    try:
        a = records[baseline]["tasks"][kind]["lengths"][str(length)]["hits"]
        b = records[candidate]["tasks"][kind]["lengths"][str(length)]["hits"]
    except KeyError:
        return None
    return paired_bootstrap(a, b)


def family_tests(records: Dict[str, Dict], baseline: str, candidate: str,
                 kinds: Sequence[str], length: int) -> Dict:
    """One comparison per task, with Holm applied across the family."""
    rows, raw = {}, {}
    for kind in kinds:
        cmp = compare(records, baseline, candidate, kind, length)
        if cmp is None or cmp["n"] == 0:
            continue
        a = records[baseline]["tasks"][kind]["lengths"][str(length)]["hits"]
        b = records[candidate]["tasks"][kind]["lengths"][str(length)]["hits"]
        test = paired_sign_test(a, b)     # on 0/1 items this is McNemar's exact test
        rows[kind] = {**cmp, "p": test["p"]}
        raw[kind] = test["p"]
    adjusted = holm(raw) if raw else {}
    for kind, p in adjusted.items():
        rows[kind]["p_holm"] = p
    return rows


def seed_summary(values: Sequence[float]) -> Dict:
    """Mean with a Student interval over seeds, for the few places we can afford seeds."""
    m, lo, hi = t_interval(list(values))
    return {"mean": m, "lo": lo, "hi": hi, "n": len(values)}
