"""Re-evaluate saved models, optionally with a change that needs no retraining.

    # Bound-Key Attention with its length scaling switched off
    python scripts/eval_ckpt.py --glob "results/runs/main/bka__*.pt" --no-logn --tag bka_nologn

    # NoPE softmax models with the length scaling switched on
    python scripts/eval_ckpt.py --glob "results/runs/main/nope__*.pt" --set logn_ref=256 --tag nope_logn

    # baselines evaluated at 16x the training length
    python scripts/eval_ckpt.py --glob "results/runs/main/softmax__p0.5__g0__s?.pt" --lengths 4096 --tag long

    # 32x and 64x, learned models only, without the attention diagnostics
    python scripts/eval_ckpt.py --glob "results/runs/main/hybrid_bka_first__p0.5__g0__s?.pt" --lengths 8192 16384 --collect-seqs 0 --learned-only --tag longer

The length scaling has no parameters and is the identity at the training
length, so switching it on or off at evaluation is an exact ablation of the
trained weights. Overrides that would add parameters are not allowed.
Writes results/evals/<tag>.json with one row per checkpoint and length.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.data import HaystackTask, TaskConfig      # noqa: E402
from sinkprobe.layers import LOGN                        # noqa: E402
from sinkprobe.metrics import evaluate                   # noqa: E402
from sinkprobe.model import ModelConfig, TinyLM          # noqa: E402
from sinkprobe.train import autocast, parse_kv           # noqa: E402

KEEP = ["recall", "recall_ci", "recency_gap", "middle_dip", "acc_q1", "acc_q2", "acc_q3", "acc_q4",
        "depth_slope", "sink_mass", "sink_ratio", "sink_noop", "sink_copy", "sink_answer",
        "entropy_norm", "act_max", "gate_mean", "ce_copy", "ce_answer", "ce_other",
        "acc_by_slot", "recall_after_first", "recall_no_marker", "marker_rate"]
ALLOWED = {"logn_ref"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    ap.add_argument("--eval-seqs", type=int, default=256)
    ap.add_argument("--set", nargs="*", help="configuration overrides without parameters, e.g. logn_ref=256")
    ap.add_argument("--no-logn", action="store_true", help="evaluate with the length scaling switched off")
    ap.add_argument("--set-task", nargs="*", help="task overrides, e.g. control=True for the text probe")
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="results/evals")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda when available)")
    ap.add_argument("--collect-seqs", type=int, default=8,
                    help="sequences for the attention diagnostics; 0 skips them (needed past 8K tokens on 4 GB)")
    ap.add_argument("--learned-only", action="store_true", help="skip models below --min-recall")
    ap.add_argument("--batch", type=int, default=None, help="sequences per batch (default: about 16K tokens)")
    a = ap.parse_args()

    overrides = parse_kv(a.set)
    task_overrides = parse_kv(a.set_task)
    bad = set(overrides) - ALLOWED
    if bad:
        raise SystemExit(f"overrides {sorted(bad)} would change the parameters; retrain instead")
    LOGN["enabled"] = not a.no_logn
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted({p for g in a.glob for p in glob.glob(g)})
    rows = []
    for path in paths:
        name = os.path.basename(path)[:-3]
        with open(path[:-3] + ".json", encoding="utf-8") as f:
            trained = json.load(f)
        train_len = trained["run"]["train_len"]
        train_recall = min(e["recall"] for e in trained["evals"] if e["seq_len"] == train_len)
        if a.learned_only and train_recall < a.min_recall:
            print(f"{name:30s} skipped: recall {train_recall:.3f} at the training length", flush=True)
            continue
        ck = torch.load(path, map_location=dev, weights_only=False)
        cfg = dict(ck["model_config"])
        cfg.update(overrides)
        model = TinyLM(ModelConfig(**cfg)).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        task_cfg = {k: v for k, v in ck["task_config"].items() if k != "vocab_size"}
        task_cfg.update(task_overrides)
        if trained["run"].get("task_kind", "haystack") == "text":
            from sinkprobe.text_task import TextConfig, TextTask
            task = TextTask(TextConfig(**task_cfg))
        else:
            task = HaystackTask(TaskConfig(**task_cfg))
        seed = trained["run"]["seed"]
        for i, L in enumerate(a.lengths):
            rng = np.random.default_rng(80_000 + 97 * i + seed)
            with autocast(dev, True):
                res = evaluate(model, task, L, a.eval_seqs, rng, dev, batch=a.batch, collect_seqs=a.collect_seqs)
            row = {"name": name, "variant": trained["run"]["variant"], "seed": seed,
                   "warmup": trained["run"].get("copy_warmup"), "train_recall": train_recall,
                   "learned": train_recall >= a.min_recall, "seq_len": L,
                   "logn": LOGN["enabled"], "overrides": overrides, "task_overrides": task_overrides}
            row.update({k: res[k] for k in KEEP if k in res})
            row["profile"] = res["profile"]
            rows.append(row)
            print(f"{name:30s} len {L:5d} recall {res['recall']:.3f} "
                  f"gap {res['recency_gap']:+.3f} sink {res.get('sink_mass', float('nan')):.3f}", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "rows": rows}, f)
    print(f"wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
