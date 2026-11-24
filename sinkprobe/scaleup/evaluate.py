"""Scoring, with the uncertainty attached.

Items are generated from a seed fixed by task and length, so every design is scored
on identical inputs and differences can be tested pairwise instead of being eyeballed
across separate samples. Per-item outcomes are kept in the run record, which is what
makes a paired bootstrap possible after the fact.
"""

from __future__ import annotations

import math
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..vargap import holm, paired_sign_test, t_interval, wilson
from .corpus import BOS, QRY, SEP, ByteWindows
from .tasks import GAPPED, LongContextTask, gap_band_config, study_config
from .train import _autocast, bits_per_byte

ITEM_SEED = 90_000          # same items for every design, so comparisons are paired


def item_seed(*parts) -> int:
    """A seed that is the same in every process.

    This used to be hash((kind, length)). Python randomises the hash of a string per
    process unless PYTHONHASHSEED is set, so each run drew its own evaluation sample and
    the paired tests below were comparing designs on different items. crc32 is stable.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return ITEM_SEED + (zlib.crc32(key) % 1_000_000)


@torch.no_grad()
def score(model, task: LongContextTask, length: int, n_items: int, device: str,
          batch: int = 32, amp: bool = True, seed: Optional[int] = None) -> Dict:
    """Exact match with a Wilson interval, a breakdown by distance, and per-item hits.

    The per-item gap is kept beside the per-item hit, so any binning of the results can
    be redone after the fact without rerunning the model.
    """
    model.eval()
    rng = np.random.default_rng(ITEM_SEED + (seed if seed is not None else 0))
    hits: List[int] = []
    gaps: List[int] = []
    gold: List[int] = []
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
        gold.extend(int(g) for g in want.flatten().tolist())
        done += n
    model.train()
    n = len(hits)
    k = int(sum(hits))
    lo, hi = wilson(k, n)
    by_gap: Dict[int, Dict] = {}
    for g, h in zip(gaps, hits):
        a, c = by_gap.get(g, (0, 0))
        by_gap[g] = (a + h, c + 1)
    # The score of the best single constant answer on this sample: a floor that needs no
    # reading at all. Uniform chance understates it when the targets are real text, where
    # a few bytes carry most of the mass. It is only a useful floor when the answer space
    # is reused across items, which it is for bytes and is not for freshly drawn symbols.
    counts: Dict[int, int] = {}
    for g in gold:
        counts[g] = counts.get(g, 0) + 1
    floor = (max(counts.values()) / n) if n else 0.0
    return {"em": k / max(1, n), "n": n, "lo": lo, "hi": hi, "chance": task.chance(),
            "best_constant": floor, "n_distinct_targets": len(counts),
            "by_gap": {int(g): {"em": a / c, "n": c} for g, (a, c) in sorted(by_gap.items())},
            "hits": hits, "gaps": gaps}


def split_by_reach(result: Dict, reach: Optional[int]) -> Dict:
    """Split one score into the part the binder could reach and the part it could not.

    `reach` is the largest key-value gap at which the binder still carries the key into
    the value vector, which arch.binder_reach computes from the width and the layout of
    a pair. With no binder there is nothing to split by, and the whole sample is reported
    as beyond reach, which is the right baseline to compare against.
    """
    gaps = result.get("gaps")
    hits = result.get("hits")
    if not gaps or not hits or len(gaps) != len(hits):
        return {}
    bound = -1 if reach is None else reach
    k_in = n_in = k_out = n_out = 0
    for g, h in zip(gaps, hits):
        if g <= bound:
            k_in += h
            n_in += 1
        else:
            k_out += h
            n_out += 1
    out = {"reach": reach, "n_in": n_in, "n_out": n_out}
    if n_in:
        lo, hi = wilson(k_in, n_in)
        out["in_reach"] = {"em": k_in / n_in, "n": n_in, "lo": lo, "hi": hi}
    if n_out:
        lo, hi = wilson(k_out, n_out)
        out["beyond_reach"] = {"em": k_out / n_out, "n": n_out, "lo": lo, "hi": hi}
    return out


@torch.no_grad()
def gap_bands(model, windows: ByteWindows, kind: str, bands: Sequence[Tuple[int, int]],
              length: int, n_items: int, device: str, batch: int = 16, amp: bool = True,
              verbose: bool = True) -> Dict:
    """Score one band of key-value distances at a time, including bands never trained on.

    Training draws the gap from 1..16. A model that learned a lookup should not care
    where in that range a pair sat, nor should it fall off a cliff just past it. A model
    that learned to read a few tokens back has an edge, and this is where it shows.
    """
    out = {}
    for lo, hi in bands:
        task = LongContextTask(gap_band_config(kind, lo, hi), windows)
        need = task.required_length()
        if need > length:
            out[f"{lo}-{hi}"] = {"skipped": f"needs {need} tokens, scored at {length}"}
            continue
        r = score(model, task, length, n_items, device, batch=batch, amp=amp,
                  seed=item_seed("band", kind, lo, hi, length))
        r.pop("hits", None)                      # the band is the unit here, not the item
        r.pop("gaps", None)
        out[f"{lo}-{hi}"] = r
        if verbose:
            print(f"  gap {lo:3d}-{hi:<3d} EM {r['em']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] "
                  f"over {r['n']} items", flush=True)
    return out


def evaluate_suite(model, windows: ByteWindows, kinds: Sequence[str], lengths: Sequence[int],
                   n_items: int, device: str, gap_max: int = 16, amp: bool = True,
                   verbose: bool = True, reach: Optional[int] = None) -> Dict:
    """Every task at every length it fits in, plus language modelling cost.

    With `reach` given, each gapped task also reports the split at the binder's edge, so
    an aggregate number is never the only thing on the table.
    """
    out: Dict[str, Dict] = {}
    for kind in kinds:
        task = LongContextTask(study_config(kind, gap_max=gap_max), windows)
        need = task.required_length()
        per_length = {}
        for L in lengths:
            if L < need:
                continue
            batch = 32 if L <= 2048 else (8 if L <= 8192 else 2)
            r = score(model, task, L, n_items, device, batch=batch, amp=amp,
                      seed=item_seed(kind, L))
            if kind in GAPPED:
                r["by_reach"] = split_by_reach(r, reach)
            per_length[str(L)] = r
            if verbose:
                extra = ""
                br = r.get("by_reach") or {}
                if br.get("in_reach") or br.get("beyond_reach"):
                    # A binder wider than every gap drawn leaves the beyond bin empty,
                    # and one that reaches nothing leaves the inside bin empty.
                    def _em(side):
                        v = br.get(side)
                        return "n/a" if not v else f"{v['em']:.3f}"
                    extra = (f"  | gap<={br.get('reach')}: {_em('in_reach')}"
                             f"  gap>{br.get('reach')}: {_em('beyond_reach')}")
                print(f"  {kind:12s} len {L:6d} EM {r['em']:.3f} "
                      f"[{r['lo']:.3f}, {r['hi']:.3f}] chance {r["chance"]:.3f} const {r["best_constant"]:.3f}{extra}",
                      flush=True)
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
