"""One job per process, with a gate in front of the expensive part.

The gate exists because of how the previous attempt failed: the plumbing was tested
and the learnability was not, so forty-six runs reported chance accuracy on a task no
model could have solved. Nothing here spends GPU hours until a small model has shown
it can learn the easiest rung of the task.

Usage: python -m sinkprobe.scaleup.runner --spec job.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .arch import ArchConfig, build, describe
from .corpus import ByteWindows, CorpusConfig, ensure_corpus
from .evaluate import evaluate_suite, score
from .overhead import forward_cost
from .tasks import LongContextTask, easy_config, study_config
from .train import TrainConfig, quick_accuracy, train


class _Bypass(nn.Module):
    """Stands in for a layer's mixer, leaving the feed-forward path untouched."""

    def forward(self, x: torch.Tensor, collect: bool = False):
        return torch.zeros_like(x), {}


def ablate_global_layers(model, task: LongContextTask, length: int, n_items: int,
                         device: str, amp: bool = True) -> List[Dict]:
    """Remove one global layer at a time at test time and score what is left."""
    out = []
    for i, blk in enumerate(model.blocks):
        if blk.kind != "S":
            continue
        keep = blk.mix
        blk.mix = _Bypass().to(device)
        try:
            r = score(model, task, length, n_items, device, amp=amp, seed=11)
        finally:
            blk.mix = keep
        out.append({"layer": i, "em": r["em"], "n": r["n"], "lo": r["lo"], "hi": r["hi"]})
    return out


def gate(device: str, corpus: Dict, steps: int = 600, threshold: float = 0.8,
         seq_len: int = 256, verbose: bool = True) -> Dict:
    """Can a small model learn the easiest rung quickly? If not, nothing else should run."""
    train_w = ByteWindows(corpus["train"], seed=0)
    val_w = ByteWindows(corpus["val"], seed=1)
    cfg = TrainConfig(size="xs", design="bkf_conv4", seq_len=seq_len, batch=16, steps=steps,
                      task_rate=1.0, task_kinds=("kv_vargap",), train_queries=4,
                      eval_every=max(50, steps // 6), eval_items=128, ckpt_every=10 ** 9,
                      easy=True, amp=(device == "cuda"))
    record, model = train(cfg, train_w, val_w, device=device, ckpt_dir=None, verbose=verbose)
    reached = max((p["kv_accuracy"] for p in record["curve"]), default=0.0)
    passed = reached >= threshold
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"passed": bool(passed), "best_accuracy": reached, "threshold": threshold,
            "steps": steps, "curve": record["curve"],
            "note": ("a model learns this task, so the ladder is worth running" if passed
                     else "the easiest rung did not train: fix the task or the budget "
                          "before spending GPU hours")}


def calibrate(cfg: TrainConfig, windows: ByteWindows, device: str, steps: int = 20) -> Dict:
    """Measured tokens per second for one configuration, to price the run honestly."""
    probe = replace(cfg, steps=steps, eval_every=10 ** 9, ckpt_every=10 ** 9)
    model, _ = build(ArchConfig(size=cfg.size, design=cfg.design, logn_ref=cfg.seq_len))
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device == "cuda"))
    from .train import Mixture, _autocast, objective
    mix = Mixture(probe, windows, np.random.default_rng(0))
    for i in range(3):                                    # warm-up, not timed
        b = {k: v.to(device) for k, v in mix.next_batch().items()}
        with _autocast(device, cfg.amp):
            logits, _ = model(b["tokens"])
        loss, _, _ = objective(logits, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(steps):
        b = {k: v.to(device) for k, v in mix.next_batch().items()}
        with _autocast(device, cfg.amp):
            logits, _ = model(b["tokens"])
        loss, _, _ = objective(logits, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = (round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    per_step = elapsed / max(1, steps)
    return {"size": cfg.size, "design": cfg.design, "seconds_per_step": round(per_step, 4),
            "tokens_per_second": round(cfg.batch * cfg.seq_len / per_step, 1),
            "peak_memory_gb": peak}


def plan_steps(seconds_per_step: float, minutes_budget: float, max_steps: int) -> int:
    """How many steps fit in the time allowed for one run."""
    if seconds_per_step <= 0:
        return max_steps
    return max(200, min(max_steps, int(minutes_budget * 60 / seconds_per_step)))


def run_job(spec: Dict) -> Dict:
    """Train one model, score the whole suite, measure its cost, write one JSON."""
    out_dir = Path(spec.get("out_dir", "results_scale"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec['tag']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    device = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus = ensure_corpus(CorpusConfig(**spec.get("corpus", {})), verbose=True)
    train_w = ByteWindows(corpus["train"], seed=spec["train"].get("seed", 0))
    val_w = ByteWindows(corpus["val"], seed=1234)

    cfg = TrainConfig(**spec["train"])
    ckpt_dir = out_dir / "ckpt" / spec["tag"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record, model = train(cfg, train_w, val_w, device=device, ckpt_dir=ckpt_dir,
                          verbose=spec.get("verbose", True))

    ev = spec.get("eval", {})
    kinds = ev.get("kinds", ["kv_vargap", "kv_multikey", "multi_hop", "freq_sym", "span_copy"])
    lengths = ev.get("lengths", [cfg.seq_len])
    record["tasks"] = evaluate_suite(model, val_w, kinds, lengths,
                                     ev.get("n_items", 512), device, gap_max=cfg.gap_max,
                                     amp=cfg.amp, verbose=spec.get("verbose", True))
    probe = LongContextTask(study_config("kv_vargap", gap_max=cfg.gap_max), val_w)
    if spec.get("ablate", True) and probe.required_length() <= cfg.seq_len:
        record["ablate_global_layers"] = ablate_global_layers(
            model, probe, cfg.seq_len, ev.get("ablate_items", 256), device, amp=cfg.amp)
    oh = spec.get("overhead", {})
    record["forward_cost"] = forward_cost(model, device, oh.get("lengths", lengths),
                                          oh.get("batch", 4), amp=cfg.amp,
                                          vocab_size=record["model_config"]["vocab_size"])
    record.update({"tag": spec["tag"], "size": cfg.size, "design": cfg.design,
                   "seed": cfg.seed, "corpus": corpus["source"],
                   "job_seconds": round(time.time() - started, 1),
                   "arch_summary": describe(ArchConfig(size=cfg.size, design=cfg.design))})
    path.write_text(json.dumps(record), encoding="utf-8")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return record


def _cli():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to a JSON job specification")
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    out = run_job(spec)
    last = (out.get("curve") or [{}])[-1]
    print(f"done {spec['tag']} kv {last.get('kv_accuracy')} bpb {last.get('bits_per_byte')}",
          flush=True)


if __name__ == "__main__":
    _cli()
