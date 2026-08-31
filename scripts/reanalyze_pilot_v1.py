"""Re-read the first pilot against the correct null model for sink mass.

    python scripts/reanalyze_pilot_v1.py

The first version of the paper compared sink mass with 1/T, the share one
position would get if attention were spread evenly over all T positions.
That is not the reference for the averaged quantity it reported. Sink mass
averages A[t, 0] over query positions t = 1..T-1, and a causal head that
spreads attention evenly gives query t exactly 1/(t+1). The reference is

    sigma_unif(T) = (1 / (T - 1)) * sum_{t=1}^{T-1} 1 / (t + 1)  ~  (ln T - 1 + gamma) / T,

which is several times larger than 1/T and falls more slowly with T.

This script recomputes the ratios quoted in the first version from the saved
results, without retraining anything, and writes

    results/pilot_v1/reanalysis.json
    results/pilot_v1/reanalysis.md
    paper/generated/table_pilot_v1_null.tex
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.metrics import uniform_sink_reference   # noqa: E402

NAMES = {"dense": "Softmax", "dense_gated": "Softmax + gate", "hybrid": "Hybrid 3:1",
         "hybrid_ar": "Hybrid + AttnRes"}


def group(runs):
    g = {}
    for r in runs:
        g.setdefault(r["variant"], []).append(r)
    return g


def main():
    base = "results/pilot_v1"
    pilot = json.load(open(os.path.join(base, "pilot.json"), encoding="utf-8"))
    ctrl = json.load(open(os.path.join(base, "control_noaux.json"), encoding="utf-8"))
    rows = []
    for label, runs in list(group(pilot["runs"]).items()) + [("control_noaux", ctrl["runs"])]:
        lengths = [e["seq_len"] for e in runs[0]["evals"]]
        for i, T in enumerate(lengths):
            sinks = [r["evals"][i]["sink_mass"] for r in runs]
            recall = [r["evals"][i]["accuracy"] for r in runs]
            m = st.mean(sinks)
            ref = uniform_sink_reference(T)
            rows.append({"model": NAMES.get(label, "Softmax, answer-only loss"), "variant": label,
                         "seq_len": T, "sink_mass": m, "sink_sd": st.stdev(sinks),
                         "naive_ref": 1.0 / T, "naive_ratio": m * T,
                         "correct_ref": ref, "correct_ratio": m / ref,
                         "recall": st.mean(recall)})

    os.makedirs("paper/generated", exist_ok=True)
    with open(os.path.join(base, "reanalysis.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)

    md = ["# First pilot, re-read against the correct null model", "",
          "| Model | T | Sink mass (mean ± sd, 3 seeds) | 1/T | Ratio to 1/T | Uniform causal reference | Ratio to reference | Recall |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    tex = []
    for r in rows:
        md.append(f"| {r['model']} | {r['seq_len']} | {r['sink_mass']:.3f} ± {r['sink_sd']:.3f} | "
                  f"{r['naive_ref']:.4f} | {r['naive_ratio']:.1f} | {r['correct_ref']:.4f} | "
                  f"{r['correct_ratio']:.1f} | {100 * r['recall']:.1f} |")
        if r["seq_len"] in (96, 768):
            tex.append(f"{r['model']} & {r['seq_len']} & {r['sink_mass']:.3f} & "
                       f"{r['naive_ratio']:.1f} & {r['correct_ratio']:.1f} \\\\")
    by = {(r["variant"], r["seq_len"]): r for r in rows}
    d96, d768, c96 = by[("dense", 96)], by[("dense", 768)], by[("control_noaux", 96)]
    md += ["", "## Claims in the first version, recomputed", "",
           f"- \"roughly thirty times an even split\" at T = 96: {d96['naive_ratio']:.1f}x against 1/T, "
           f"{d96['correct_ratio']:.1f}x against the uniform causal reference.",
           f"- \"about ninety seven times an even share\" at T = 768: {d768['naive_ratio']:.1f}x against 1/T, "
           f"{d768['correct_ratio']:.1f}x against the reference.",
           f"- The answer-only control measured {c96['sink_mass']:.3f} at T = 96, and uniform causal attention gives "
           f"{c96['correct_ref']:.3f} ({c96['correct_ratio']:.2f}x). Positions outside the loss receive no gradient, "
           "so this control shows attention that was never trained, not a sink removed by the objective."]
    with open(os.path.join(base, "reanalysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open("paper/generated/table_pilot_v1_null.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
