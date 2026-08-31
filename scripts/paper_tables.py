"""Numbers and LaTeX tables for the ICLR paper, read from the saved evaluations.

    python scripts/paper_tables.py

Reads the single-pass rescoring of every main run (results/evals/rescore_*.json),
the discovery steps (results/summary_discovery.json) and the usable-context
summary (results/summary_robustness.json). Writes paper/iclr/generated/*.tex and
results/summary_paper.json. Recall is averaged over runs that learned the task
(recall of at least 90% at the training length); the learned count is reported
next to it.
"""

from __future__ import annotations

import json
import os
import re
import statistics as stats
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "iclr", "generated")

# Display order. Each design is (variant, label, group, citation key of the
# mechanism it reimplements, empty for our own variants).
# Row labels name the mechanism each design reimplements, with the source paper.
DESIGNS = [
    ("softmax", "Softmax with RoPE", "attn", "su2024roformer"),
    ("gate", "Gated attention", "attn", "qiu2025gated"),
    ("sinklogit", "Learned sink logit", "attn", "openai2025gptoss"),
    ("softpick", "Rectified softmax", "attn", "zuhri2025softpick"),
    ("attnres", "Attention residuals", "attn", "kimi2026attnres"),
    ("nope", "NoPE", "attn", "kazemnejad2023impact"),
    ("bka", r"NoPE, bound keys, $\lambda_t$", "attn", ""),
    ("hybrid", "Gated DeltaNet hybrid, RoPE", "llls", "yang2025gated"),
    ("hybrid_nogate", "same, no output gate", "llls", ""),
    ("hybrid_nope", "NoPE hybrid (standard)", "llls", "kimi2025linear"),
    ("hybrid_attnres", "NoPE hybrid, attention residuals", "llls", "kimi2026k3"),
    ("hybrid_bka", r"NoPE hybrid, bound keys, $\lambda_t$", "llls", "xu2024kv"),
    ("hybrid_nope_first", r"NoPE, $\lambda_t$, order of SWAN", "slll", "puvvada2025swan"),
    ("hybrid_bka_first", "BKF (proposed)", "slll", ""),
]
LENGTHS = [256, 1024, 2048, 4096]


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def rows_by_run():
    """One evaluation per (run, length), the latest file winning."""
    table = {}
    for path in ("results/evals/rescore_main.json", "results/evals/rescore_stage5.json",
                 "results/evals/rescore_longer.json"):
        for r in load(path)["rows"]:
            if r.get("warmup", 1500) != 1500:
                continue
            table[(r["name"], r["seq_len"])] = r
    return table


def mean(xs):
    return stats.fmean(xs) if xs else float("nan")


def median(xs):
    return stats.median(xs) if xs else float("nan")


def fmt(x, nd=1, dash="--"):
    return dash if x != x else f"{x:.{nd}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    table = rows_by_run()
    disc = load("results/summary_discovery.json")
    steps = {}
    for group in disc.values():
        for name, s in group.items():
            steps[name] = s
    usable = load("results/summary_robustness.json")["usable_context"]

    runs = defaultdict(set)
    for (name, _L) in table:
        variant = name.split("__")[0]
        runs[variant].add(name)

    summary = {}
    for variant, label, group, cite in DESIGNS:
        names = sorted(runs[variant])
        learned = [n for n in names if table.get((n, 256), {}).get("recall", 0) >= 0.9]
        rec = {}
        for L in LENGTHS + [8192, 16384]:
            vals = [table[(n, L)] for n in learned if (n, L) in table]
            rec[L] = {
                "n": len(vals),
                "recall": mean([100 * v["recall"] for v in vals]),
                "no_marker": mean([100 * v["recall_no_marker"] for v in vals]),
                "marker_rate": mean([100 * v["marker_rate"] for v in vals]),
                "after_first": mean([100 * v["recall_after_first"] for v in vals]),
                "min_recall": min([100 * v["recall"] for v in vals]) if vals else float("nan"),
            }
        disc_steps = [steps[n] for n in names if n in steps]
        learned_steps = [s for s in disc_steps if s is not None]
        sink = [table[(n, 256)]["sink_mass"] for n in learned if (n, 256) in table]
        u = usable.get(variant, {})
        summary[variant] = {
            "label": label, "group": group, "trained": len(names), "learned": len(learned),
            "steps_to_90": disc_steps, "median_steps": median(learned_steps),
            "recall": rec, "sink_mass": mean(sink), "sink_min": min(sink) if sink else None,
            "sink_max": max(sink) if sink else None,
            "usable_median": u.get("median"), "usable": u.get("multiples"),
        }

    # Main comparison table. The first column holds the group name, written
    # once per group as a rotated multirow cell.
    group_names = {"attn": "attention only", "llls": "global last", "slll": "global first"}
    group_sizes = defaultdict(int)
    for _v, _l, group, _c in DESIGNS:
        group_sizes[group] += 1
    lines = []
    prev = None
    for variant, label, group, cite in DESIGNS:
        s = summary[variant]
        if prev is not None and group != prev:
            lines.append(r"\midrule")
        first_of_group = group != prev
        prev = group
        r = s["recall"]
        ours = variant == "hybrid_bka_first"
        name = label + (rf"~\citep{{{cite}}}" if cite else "")
        if ours:
            name = r"\textbf{BKF (proposed)}"
        cells = [
            name,
            f"{s['learned']}/{s['trained']}",
            fmt(s["median_steps"], 0),
            fmt(r[1024]["recall"]), fmt(r[2048]["recall"]), fmt(r[4096]["recall"]),
            fmt(r[4096]["no_marker"]),
            fmt(s["sink_mass"], 3),
        ]
        if ours:
            cells = [cells[0]] + [rf"\textbf{{{c}}}" if c != "--" else c for c in cells[1:-1]] + [cells[-1]]
        lead = ""
        if first_of_group:
            n = group_sizes[group]
            label = group_names[group]
            if n <= 2:
                # Two rows are too short for one rotated line; stack the words.
                label = r"\tiny\shortstack{" + label.replace(" ", r"\\") + "}"
            else:
                label = r"\scriptsize " + label
            lead = rf"\multirow{{{n}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{label}}}}}"
        lines.append(lead + " & " + " & ".join(cells) + r" \\")
    with open(os.path.join(OUT, "table_main_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Predictions fixed before the runs, with the outcome scored from the result
    # files by scripts/table_predictions.py (results/summary_predictions.json).
    short = {
        0: "all 4 seeds at step 300",
        1: "97.0--100\\%",
        2: "both at step 300; 100\\% and 96.6\\%",
        3: "seed 4 learned at step 1{,}100",
        4: "seed 2 learned at 2{,}600; seed 3 passed on warm-up data only",
        5: "$+0.009$ bits per byte",
        6: "$+14.7$ against $+16.3$ points",
        7: "4 seeds lower by 3.9 to 63.9 points at 4{,}096; marker emission",
        8: "74.0$\\to$98.1, 55.0$\\to$90.0, 98.4$\\to$99.9, 82.8$\\to$99.2\\%",
        9: "0.57--0.69, lowest bin 0.50--0.67",
        10: "highest 0.003--0.004",
        11: "97.3--100\\% at 8{,}192; 98.5--100\\% at 16{,}384",
        12: "0.0 lost on all 6; 6.4 to 38.3 lost without bound keys",
        13: "seed 1: $-28.4$ at 2{,}048, all marker emission",
        14: "all 3 at step 300",
        15: "300 against 1{,}900, never, 1{,}400",
        16: "100\\% on all 3",
        17: "both at 300; 11/11 against 13/20, $p=0.033$",
    }
    # The paper avoids pronouns, so four phrases are reworded with the same meaning.
    wording = {
        "within 2 points of its standard-data recall": "within 2 points of the standard-data recall",
        "puts at least 0.25 on it at 4096": "puts at least 0.25 on the value at 4096",
        "no depth bin below half of its mean": "no depth bin below half of the head mean",
        "the layer-last seed beside it": "the layer-last seed trained alongside",
    }
    preds = load("results/summary_predictions.json")
    lines = []
    for i, p in enumerate(preds):
        text = p["prediction"]
        for old, new in wording.items():
            text = text.replace(old, new)
        text = text.replace("%", "\\%").replace("p < 0.05", "$p<0.05$")
        outcome = p["outcome"].replace("failed for seed", "failed, seed")
        lines.append(f"{p['stage']} & {text} & {outcome} & {short.get(i, '')} \\\\")
    with open(os.path.join(OUT, "table_predictions_short.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Statistical tests, as written by scripts/stats_placement.py.
    src = os.path.join(ROOT, "paper", "generated", "table_placement_tests.tex")
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f_in:
            text = f_in.read()
        # Paper names for the groups, and p-values in scientific notation.
        for old, new in [("SLLL, bound keys", "BKF"), ("SLLL, NoPE", "SLLL without bound keys"),
                         (": discovery", ": learning"), ("the first slot", "the first question"),
                         ("at 4096", "at 4{,}096 tokens"), ("length scaling on for every run",
                                                             "length scale on for all runs"),
                         ("rank sum", "rank-sum")]:
            text = text.replace(old, new)
        text = re.sub(r"\b(\d)e-0?(\d+)\b", r"$\1\\times10^{-\2}$", text)
        with open(os.path.join(OUT, "table_placement_tests.tex"), "w", encoding="utf-8") as f_out:
            f_out.write(text)
    else:
        print(f"skipped the tests table: run scripts/stats_placement.py first to write {src}")

    # Longer contexts for the three hybrids that hold retrieval, over the runs
    # evaluated at every length so that each row compares the same models.
    lines = []
    for variant, label in (("hybrid_nope", r"Global last (standard hybrid)"),
                           ("hybrid_nope_scaled", r"same, $\lambda_t$ at test"),
                           ("hybrid_bka", r"Global last, bound keys, $\lambda_t$"),
                           ("hybrid_bka_first", r"\textbf{BKF (proposed)}")):
        if variant == "hybrid_nope_scaled":
            # Separate test pass of the learned standard hybrids with the length scale on
            # (256 inputs at 4,096 tokens, 48 above); markers-excluded recall was not stored.
            learned = {n for n in runs["hybrid_nope"] if table.get((n, 256), {}).get("recall", 0) >= 0.9}
            cells = [label, str(len(learned))]
            for path, L in (("results/evals/llls_nope_logn.json", 4096),
                            ("results/evals/longer_llls_nope_logn.json", 8192),
                            ("results/evals/longer_llls_nope_logn.json", 16384)):
                rows = load(path)["rows"]
                cells.append(fmt(mean([100 * r["recall"] for r in rows
                                       if r["name"] in learned and r["seq_len"] == L])))
            cells += ["--", "--", "--"]
            lines.append(" & ".join(cells) + r" \\")
            continue
        names = [n for n in sorted(runs[variant])
                 if table.get((n, 256), {}).get("recall", 0) >= 0.9 and (n, 16384) in table]
        cells = [label, str(len(names))]
        for key in ("recall", "recall_no_marker"):
            for L in (4096, 8192, 16384):
                cell = fmt(mean([100 * table[(n, L)][key] for n in names]))
                cells.append(rf"\textbf{{{cell}}}" if variant == "hybrid_bka_first" else cell)
        lines.append(" & ".join(cells) + r" \\")
    with open(os.path.join(OUT, "table_longer.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(ROOT, "results", "summary_paper.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    for variant, *_ in DESIGNS:
        s = summary[variant]
        r = s["recall"]
        print(f"{s['label']:34s} {s['learned']:2d}/{s['trained']:2d} steps {fmt(s['median_steps'], 0):>5s} "
              f"| 4x {fmt(r[1024]['recall']):>5s} 8x {fmt(r[2048]['recall']):>5s} 16x {fmt(r[4096]['recall']):>5s} "
              f"(no marker {fmt(r[4096]['no_marker']):>5s}, marker {fmt(r[4096]['marker_rate']):>5s}, n={r[4096]['n']}) "
              f"| 32x {fmt(r[8192]['recall']):>5s}/{fmt(r[8192]['no_marker']):>5s} "
              f"64x {fmt(r[16384]['recall']):>5s}/{fmt(r[16384]['no_marker']):>5s} (n={r[16384]['n']}) "
              f"| sink {fmt(s['sink_mass'], 3)} usable {s['usable_median']}")


if __name__ == "__main__":
    main()
