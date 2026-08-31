"""Does sink size track how fast a run learns retrieval?

Rank correlation between the sink mass at the training length (the points of Figure 1a,
results/summary_sink_recall.json) and the learning step (results/summary_discovery.json)
over the learned runs with the standard warm-up, with a bootstrap interval, and the same
correlation without the BKF runs and without any bound-key design.

    python scripts/sink_vs_step.py     # writes results/summary_sink_step.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def summary(x, y, rng, n_boot=10_000):
    r = spearmanr(x, y)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        b = spearmanr(x[i], y[i]).statistic
        if not np.isnan(b):
            boots.append(b)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": int(len(x)), "rho": float(r.statistic), "p": float(r.pvalue), "ci95": [float(lo), float(hi)]}


def main():
    points = load("results/summary_sink_recall.json")["points"]
    steps = {}
    for group, runs in load("results/summary_discovery.json").items():
        if "no warm-up" not in group:
            steps.update(runs)
    rows = [(p["run"], p["sink_mass"], steps[p["run"]]) for p in points if steps.get(p["run"]) is not None]
    names = [r[0] for r in rows]
    sink = np.array([r[1] for r in rows])
    step = np.array([r[2] for r in rows], dtype=float)
    rng = np.random.default_rng(0)
    out = {"all": summary(sink, step, rng)}
    keep = np.array(["hybrid_bka_first" not in n for n in names])
    out["without_bkf"] = summary(sink[keep], step[keep], rng)
    keep = np.array(["bka" not in n for n in names])
    out["without_bound_keys"] = summary(sink[keep], step[keep], rng)
    with open(os.path.join(ROOT, "results", "summary_sink_step.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k:20s} n={v['n']:2d} rho={v['rho']:+.3f} p={v['p']:.4f} "
              f"95% [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
