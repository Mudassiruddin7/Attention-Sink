"""What the binder and the layer order cost, measured rather than argued.

A design that wins on accuracy but is slower or heavier has to say so. This records
parameters, forward latency and memory at several context lengths, and expresses each
design against the matched baseline at the same size.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .arch import ArchConfig, build, describe
from .train import _autocast


@torch.no_grad()
def forward_cost(model, device: str, lengths: Sequence[int] = (1024, 4096, 16384),
                 batch: int = 4, repeats: int = 3, amp: bool = True,
                 vocab_size: int = 387) -> Dict[str, Dict]:
    """Latency, throughput and peak memory for a forward pass at each length."""
    model.eval()
    out: Dict[str, Dict] = {}
    for L in lengths:
        try:
            x = torch.randint(0, vocab_size, (batch, L), device=device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            with _autocast(device, amp):                      # one warm-up, then timed
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(repeats):
                tick = time.time()
                with _autocast(device, amp):
                    model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - tick)
            best = float(np.median(times))
            out[str(L)] = {
                "seconds": round(best, 4),
                "tokens_per_second": round(batch * L / best, 1) if best else None,
                "peak_memory_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                   if device == "cuda" else None),
                "batch": batch}
            del x
        except torch.cuda.OutOfMemoryError:
            out[str(L)] = {"error": "out of memory", "batch": batch}
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            out[str(L)] = {"error": str(exc)[:200], "batch": batch}
    model.train()
    return out


def measure_design(size: str, design: str, device: str, lengths=(1024, 4096, 16384),
                   batch: int = 4, amp: bool = True) -> Dict:
    """Cost of a freshly built model, so this can run without training first."""
    model, mcfg = build(ArchConfig(size=size, design=design))
    model = model.to(device)
    record = {**describe(ArchConfig(size=size, design=design)),
              "forward": forward_cost(model, device, lengths, batch, amp=amp,
                                      vocab_size=mcfg.vocab_size)}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return record


def cost_table(sizes: Sequence[str], designs: Sequence[str], device: str,
               lengths=(1024, 4096), batch: int = 4, amp: bool = True,
               verbose: bool = True) -> List[Dict]:
    rows = []
    for size in sizes:
        for design in designs:
            row = measure_design(size, design, device, lengths, batch, amp)
            rows.append(row)
            if verbose:
                at = row["forward"].get(str(lengths[0]), {})
                print(f"  {size:3s} {design:14s} {row['params'] / 1e6:7.1f}M params "
                      f"{at.get('tokens_per_second', '-')!s:>10s} tok/s at {lengths[0]}",
                      flush=True)
    return rows


def relative(rows: Sequence[Dict], baseline: str = "global_last",
             length: int = 1024) -> List[Dict]:
    """Each design against the baseline at its own size: the overhead a reader cares about."""
    base = {r["size"]: r for r in rows if r["design"] == baseline}
    out = []
    for r in rows:
        b = base.get(r["size"])
        if not b:
            continue
        mine = r["forward"].get(str(length), {})
        theirs = b["forward"].get(str(length), {})
        speed = (mine.get("tokens_per_second"), theirs.get("tokens_per_second"))
        out.append({
            "size": r["size"], "design": r["design"],
            "params_vs_baseline": round(100 * (r["params"] / b["params"] - 1), 2),
            "binder_share": round(100 * r["binder_params"] / max(1, r["params"]), 2),
            "throughput_vs_baseline": (round(100 * (speed[0] / speed[1] - 1), 1)
                                       if speed[0] and speed[1] else None)})
    return out
