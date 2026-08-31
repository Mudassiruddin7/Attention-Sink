"""Natural-text check: language modelling, and copying a repeated span by depth and length.

    python scripts/report_text.py

Reads   results/runs/text/*.json, results/evals/text_control.json
Writes  paper/generated/table_text.tex
        paper/figures/text_copy_profiles.{pdf,png}
        results/summary_text.json

Copy accuracy is next-byte accuracy on the repeated span after its first 8
bytes. The control repeats a span that is not in the context, so the
difference between the two is what reading the context adds. Language
modelling is the loss in bits per byte on the held-out probe passages. Designs
in PAIRS were trained side by side on the same seeds, which fixes the data
order, so their differences are also reported seed by seed.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import make_report as mr   # noqa: E402

LABEL = {"softmax": "Softmax, RoPE", "nope": "Softmax, NoPE", "bka": "BKA",
         "hybrid_nope": "Hybrid 3:1, NoPE", "hybrid_bka": "Hybrid 3:1, NoPE, bound keys",
         "hybrid_bka_first": "Hybrid, bound-key layer first"}
# The same colour per design as scripts/report_bka.py.
COLORS = {"softmax": "#0072B2", "nope": "#2F4F4F", "bka": "#777777", "hybrid_nope": "#D55E00",
          "hybrid_bka": "#E69F00", "hybrid_bka_first": "#009E73"}
ORDER = ["softmax", "nope", "bka", "hybrid_nope", "hybrid_bka", "hybrid_bka_first"]
PAIRS = [("hybrid_bka_first", "hybrid_bka")]      # first minus second, matched by seed
LENGTHS = [512, 1024, 2048]
INK, INK_MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def main():
    runs = []
    for p in sorted(glob.glob("results/runs/text/*.json")):
        r = json.load(open(p, encoding="utf-8"))
        evals = {e["seq_len"]: e for e in r["evals"]}
        for e in evals.values():
            mr.add_depth_fit(e)
        runs.append({"name": os.path.basename(p)[:-5], "variant": r["run"]["variant"],
                     "seed": r["run"]["seed"], "evals": evals})
    ctrl_path = "results/evals/text_control.json"
    ctrl = {}
    if os.path.exists(ctrl_path):
        ctrl = {(row["name"], row["seq_len"]): row
                for row in json.load(open(ctrl_path, encoding="utf-8"))["rows"]}

    summary, lines = {}, []
    for v in ORDER:
        rs = [r for r in runs if r["variant"] == v]
        if not rs:
            continue
        entry = {"n": len(rs)}
        for L in LENGTHS:
            copy = [100 * r["evals"][L]["recall"] for r in rs if L in r["evals"]]
            control = [100 * ctrl[(r["name"], L)]["recall"] for r in rs if (r["name"], L) in ctrl]
            entry[f"copy@{L}"] = mr.mean_ci(copy)
            entry[f"control@{L}"] = mr.mean_ci(control)
            if len(copy) == len(control) and copy:
                entry[f"advantage@{L}"] = mr.mean_ci([a - b for a, b in zip(copy, control)])
            entry[f"slope@{L}"] = mr.mean_ci([100 * r["evals"][L]["fit_slope"] for r in rs
                                              if L in r["evals"] and "fit_slope" in r["evals"][L]])
            entry[f"bpb@{L}"] = mr.mean_ci([r["evals"][L]["ce_other"] / math.log(2) for r in rs
                                            if L in r["evals"] and "ce_other" in r["evals"][L]])
        summary[LABEL[v]] = entry

        def c(key, digits=1):
            m, h, n = entry.get(key, (float("nan"), float("nan"), 0))
            if n == 0:
                return "--"
            return f"{m:.{digits}f}" + (f"{{\\scriptsize$\\pm${h:.{digits}f}}}" if np.isfinite(h) else "")
        lines.append(" & ".join([LABEL[v], str(len(rs)), c("copy@512"), c("copy@1024"), c("copy@2048"),
                                 c("control@2048"), c("advantage@2048"), c("slope@2048", 0),
                                 c("bpb@512", 3)]) + " \\\\")

    paired = {}
    for a, b in PAIRS:
        partner = {r["seed"]: r for r in runs if r["variant"] == b}
        diffs = {}
        for r in runs:
            o = partner.get(r["seed"])
            if r["variant"] != a or o is None:
                continue
            for L in LENGTHS:
                if L in r["evals"] and L in o["evals"]:
                    diffs.setdefault(f"bpb@{L}", []).append(
                        (r["evals"][L]["ce_other"] - o["evals"][L]["ce_other"]) / math.log(2))
                    diffs.setdefault(f"copy@{L}", []).append(100 * (r["evals"][L]["recall"] - o["evals"][L]["recall"]))
        if diffs:
            paired[f"{LABEL[a]} minus {LABEL[b]}"] = {k: {"per_seed": d, "mean": float(np.mean(d))}
                                                     for k, d in diffs.items()}

    mr.write("paper/generated/table_text.tex", "\n".join(lines) + "\n")
    with open("results/summary_text.json", "w", encoding="utf-8") as f:
        json.dump({"designs": summary, "paired": paired}, f, indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK_MUTED,
                         "ytick.color": INK_MUTED, "axes.axisbelow": True, "axes.grid": True,
                         "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                         "grid.linestyle": "-", "legend.frameon": False, "text.color": INK})
    os.makedirs("paper/figures", exist_ok=True)
    fig, ax = plt.subplots(1, len(LENGTHS), figsize=(6.8, 2.2), sharey=True)
    for j, L in enumerate(LENGTHS):
        for v in ORDER:
            rs = [r for r in runs if r["variant"] == v and L in r["evals"]]
            if not rs:
                continue
            arr = 100 * np.array([[b["acc"] for b in r["evals"][L]["profile"]] for r in rs])
            centres = [(i + 0.5) / arr.shape[1] for i in range(arr.shape[1])]
            if len(arr) > 1:
                ax[j].fill_between(centres, arr.min(0), arr.max(0), color=COLORS[v], alpha=0.12, lw=0)
            ax[j].plot(centres, arr.mean(0), marker="o", ms=3.5, mec="white", mew=0.8, lw=1.5, color=COLORS[v],
                       label=f"{LABEL[v]} (n={len(rs)})")
        ax[j].set_title(f"{L} bytes ({L // 512}x training length)")
        ax[j].set_xlabel("depth of the original span")
    ax[0].set_ylabel("copy accuracy (%)")
    ax[-1].legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/text_copy_profiles.{ext}", dpi=200)
    plt.close(fig)

    for label, e in summary.items():
        print(label, {k: (round(v[0], 2) if isinstance(v, (list, tuple)) else v) for k, v in e.items()})
    for label, d in paired.items():
        print(label, {k: [round(x, 3) for x in v["per_seed"]] for k, v in d.items()})


if __name__ == "__main__":
    main()
