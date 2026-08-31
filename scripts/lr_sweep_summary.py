"""Learning step and recall of the learning-rate check (scripts/run_lr_sweep.sh).

Reads results/runs/lr_sweep/*.json, adds the main runs at the default learning rate
(3e-3, seeds 0 and 1) for reference, and writes results/summary_lr_sweep.json.

    python scripts/lr_sweep_summary.py
"""

from __future__ import annotations

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def learning_step(run, threshold=0.9):
    """First logged step with training recall above the threshold (None if never)."""
    for row in run["curve"]:
        if row.get("train_recall", 0) > threshold:
            return row["step"]
    return None


def summarise(path, lr):
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    evals = {e["seq_len"]: e for e in run["evals"]}
    return {"name": os.path.basename(path)[:-5], "variant": run["run"]["variant"], "seed": run["run"]["seed"],
            "lr": lr, "learning_step": learning_step(run),
            "recall_256": evals[256]["recall"], "recall_4096": evals[4096]["recall"]}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "lr_sweep", "*.json"))):
        lr = float(os.path.basename(path).split("__lr")[1][:-5])
        rows.append(summarise(path, lr))
    for variant in ("hybrid_bka_first", "hybrid_bka"):
        for seed in (0, 1):
            path = os.path.join(ROOT, "results", "runs", "main", f"{variant}__p0.5__g0__s{seed}.json")
            if os.path.exists(path):
                rows.append(summarise(path, 3e-3))
    with open(os.path.join(ROOT, "results", "summary_lr_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    for r in sorted(rows, key=lambda r: (r["lr"], r["variant"], r["seed"])):
        print(f"lr {r['lr']:.0e}  {r['variant']:18s} s{r['seed']}  learning step {str(r['learning_step']):>5s}  "
              f"recall 256 {100 * r['recall_256']:5.1f}  4096 {100 * r['recall_4096']:5.1f}")


if __name__ == "__main__":
    main()
