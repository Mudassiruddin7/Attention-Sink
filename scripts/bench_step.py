"""Time one optimiser step for every variant, with and without bf16 autocast.

    python scripts/bench_step.py --out results/bench/step_times.json

The numbers go into the compute appendix of the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.data import HaystackTask, TaskConfig          # noqa: E402
from sinkprobe.model import VARIANTS, ModelConfig, TinyLM     # noqa: E402
from sinkprobe.train import autocast, objective, param_groups  # noqa: E402


def bench(variant, task, batch, seq_len, amp, device, reps=15):
    torch.manual_seed(0)
    model = TinyLM(ModelConfig.from_variant(variant, vocab_size=task.cfg.vocab_size)).to(device)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-3, fused=True)
    b = task.build(batch, seq_len, np.random.default_rng(0))
    b = {k: v.to(device) for k, v in b.items()}
    torch.cuda.reset_peak_memory_stats()

    def step():
        with autocast(device, amp):
            logits, _ = model(b["tokens"])
        loss, _, _ = objective(logits, b, 1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return {"variant": variant, "batch": batch, "seq_len": seq_len, "amp": amp,
            "params": model.n_params(), "ms_per_step": 1000 * dt,
            "tokens_per_s": batch * seq_len / dt,
            "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench/step_times.json")
    ap.add_argument("--batch", type=int, nargs="+", default=[32])
    ap.add_argument("--seq-len", type=int, default=256)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    task = HaystackTask(TaskConfig())
    rows = []
    for batch in a.batch:
        for variant in VARIANTS:
            for amp in (False, True):
                r = bench(variant, task, batch, a.seq_len, amp, "cuda")
                rows.append(r)
                print(f"{variant:15s} B={batch:3d} amp={int(amp)} {r['ms_per_step']:7.1f} ms "
                      f"{r['tokens_per_s']:8.0f} tok/s peak {r['peak_mib']:6.0f} MiB", flush=True)
                torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "rows": rows}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
