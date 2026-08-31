"""Two robustness summaries read from the saved runs, with no training.

    python scripts/robustness_summary.py

Usable context. The AI Index 2026 (after Burnham and Adamczewski, 2025) tracks
the input length at which models still reach 80% accuracy. The same measure
for the controlled models is the longest evaluated length at which recall is
at least 80%, with every shorter evaluated length also at least 80%, given as a
multiple of the training length. Lengths come from each run's own evaluations
and from saved evaluations of the unmodified models. A run that holds 80% at
its longest evaluated length is reported as at least that multiple; runs were
not all evaluated at the same lengths, so the grid is coarse for some.

Data change. Training switches from copy-rich data (p_noop 0) to the condition
data (p_noop 0.5) at step 1500. For runs whose training recall was at least 50%
at step 1400, the drop to their lowest training recall between steps 1500 and
1800 measures how much of what was learned on copy-rich data survives the
change.

Reads   results/runs/main/*.json, results/evals/<tag>.json for the tags in AS_TRAINED
Writes  paper/generated/table_usable_context.tex, paper/generated/table_data_change.tex,
        results/summary_robustness.json
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stats_placement import rank_sum_p   # noqa: E402

RUNS = "results/runs/main"
EVALS = "results/evals"
AS_TRAINED = ("long_baselines", "hybrid_bka_4096", "pilot_bka_logn_on", "pilot_nope_short")
LEVEL = 0.8
BEFORE, AFTER = 1400, (1500, 1800)
LABELS = {"softmax": "Softmax, RoPE", "gate": "Softmax + output gate", "sinklogit": "Softmax + sink logit",
          "softpick": "Softpick", "attnres": "Softmax + AttnRes", "nope": "Softmax, NoPE",
          "bka": "BKA, attention only", "hybrid": "Hybrid 3:1, RoPE", "hybrid_nogate": "Hybrid 3:1, RoPE, no gate",
          "hybrid_nodelta": "Hybrid 3:1, no delta rule", "hybrid_nope": "Hybrid 3:1, NoPE",
          "hybrid_attnres": "Hybrid 3:1, NoPE, AttnRes", "hybrid_bka": "Hybrid 3:1, NoPE, bound keys",
          "hybrid_nope_first": "Hybrid, NoPE layer first", "hybrid_bka_first": "Hybrid, bound-key layer first"}
ORDER = list(LABELS)


def load_runs():
    extra = defaultdict(dict)
    for tag in AS_TRAINED:
        path = os.path.join(EVALS, f"{tag}.json")
        if os.path.exists(path):
            for row in json.load(open(path, encoding="utf-8"))["rows"]:
                extra[row["name"]][row["seq_len"]] = row["recall"]
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        tc = r["task_config"]
        if tc.get("p_noop") is None or abs(tc["p_noop"] - 0.5) > 1e-9 or abs(tc.get("query_skew", 0.0)) > 1e-9:
            continue
        name = os.path.basename(p)[:-5]
        recall = {e["seq_len"]: e["recall"] for e in r["evals"]}
        for length, value in extra.get(name, {}).items():
            recall.setdefault(length, value)
        runs.append({"name": name, "variant": r["run"]["variant"], "warmup": r["run"].get("copy_warmup", 0),
                     "train_len": r["run"]["train_len"], "recall": recall,
                     "curve": {c["step"]: c["train_recall"] for c in r["curve"]}})
    return runs


def usable(run):
    """(multiple of the training length, True when it is only a lower bound)."""
    lengths = sorted(run["recall"])
    best = None
    for length in lengths:
        if run["recall"][length] < LEVEL:
            break
        best = length
    if best is None:
        return 0.0, False
    return best / run["train_len"], best == lengths[-1]


def data_change(run):
    before = run["curve"].get(BEFORE)
    after = [v for s, v in run["curve"].items() if AFTER[0] <= s <= AFTER[1]]
    if before is None or not after or before < 0.5:
        return None
    return before, min(after)


def multiple(m, lower, latex=False):
    text = f"{m:g}x"
    return (("$\\geq$" if latex else ">=") + text) if lower else text


def main():
    by_variant = defaultdict(list)
    for r in load_runs():
        if r["warmup"] == 1500:
            by_variant[r["variant"]].append(r)
    summary = {"usable_context": {}, "data_change": {}}
    usable_rows, change_rows = [], []
    print(f"Usable context: longest length with recall >= {LEVEL:.0%}, as a multiple of the training length")
    for v in sorted(by_variant, key=lambda v: ORDER.index(v) if v in ORDER else len(ORDER)):
        runs = by_variant[v]
        learned = [r for r in runs if r["recall"].get(r["train_len"], 0.0) >= 0.9]
        if learned:
            vals = [usable(r) for r in learned]
            median = float(np.median([m for m, _ in vals]))
            summary["usable_context"][v] = {"runs": [r["name"] for r in learned], "multiples": vals,
                                            "median": median}
            usable_rows.append(f"{LABELS.get(v, v)} & {len(learned)} & "
                               f"{', '.join(multiple(m, lo, True) for m, lo in vals)} & {median:g}x \\\\")
            print(f"  {LABELS.get(v, v):32s} {len(learned)} learned: {', '.join(multiple(m, lo) for m, lo in vals)}"
                  f" | median {median:g}x")
    print(f"Data change at step 1500: training recall at step {BEFORE} and lowest between {AFTER[0]} and {AFTER[1]}")
    for v in sorted(by_variant, key=lambda v: ORDER.index(v) if v in ORDER else len(ORDER)):
        changes = [(r, c) for r in by_variant[v] if (c := data_change(r)) is not None]
        if not changes:
            continue
        drops = [100 * (b - a) for _, (b, a) in changes]
        summary["data_change"][v] = {r["name"]: {"before": b, "lowest_after": a} for r, (b, a) in changes}
        pairs = ", ".join(f"{100 * b:.0f} to {100 * a:.0f}" for _, (b, a) in changes)
        change_rows.append(f"{LABELS.get(v, v)} & {len(changes)}/{len(by_variant[v])} & {pairs} & "
                           f"{np.mean(drops):.0f} \\\\")
        print(f"  {LABELS.get(v, v):32s} {len(changes)}/{len(by_variant[v])} runs: {pairs} | mean drop "
              f"{np.mean(drops):.0f} points")
    pair = ("hybrid_bka_first", "hybrid_nope_first")      # the same placement with and without bound keys
    if all(v in summary["data_change"] for v in pair):
        drops = [[100 * (x["before"] - x["lowest_after"]) for x in summary["data_change"][v].values()] for v in pair]
        p = rank_sum_p(drops[0], drops[1])
        summary["data_change_test"] = {"a": pair[0], "b": pair[1], "drops_a": drops[0], "drops_b": drops[1],
                                       "rank_sum_p": p}
        print(f"  drop, {LABELS[pair[0]]} against {LABELS[pair[1]]}: {np.mean(drops[0]):.0f} against "
              f"{np.mean(drops[1]):.0f} points, exact permutation p = {p:.4f}")
    os.makedirs("paper/generated", exist_ok=True)
    with open("paper/generated/table_usable_context.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(usable_rows) + "\n")
    with open("paper/generated/table_data_change.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(change_rows) + "\n")
    with open("results/summary_robustness.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
