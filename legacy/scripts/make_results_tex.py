"""Emit every table and plot in the results section of the paper.

Each output is stored under a marker name. scripts/build_paper.py then
substitutes those markers into the paper source, so no table cell and no
plot coordinate in Section VI is typed by hand.

    python scripts/make_results_tex.py

Writes paper/fragments.json and paper/results_tables.tex.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ORDER = ["dense", "dense_gated", "hybrid", "hybrid_ar"]
ROWNAME = {"dense": "Softmax", "dense_gated": "Softmax + gate",
           "hybrid": "Hybrid 3:1", "hybrid_ar": "Hybrid + AttnRes",
           "dense_noaux": "Softmax, no auxiliary term"}
COLOUR = {"dense": "sinkColor", "dense_gated": "gateColor",
          "hybrid": "kdaColor", "hybrid_ar": "depthColor"}
MARK = {"dense": "*", "dense_gated": "square*", "hybrid": "triangle*",
        "hybrid_ar": "diamond*"}
SOFTMAX_LAYERS = {"dense": 8, "dense_gated": 8, "hybrid": 2, "hybrid_ar": 2,
                  "dense_noaux": 8}


def ms(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), 0.0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def group(runs):
    g = {}
    for r in runs:
        g.setdefault(r["variant"], []).append(r)
    return g


def load(pilot_path="results/pilot.json",
         cost_path="results/costmodel.json",
         ctrl_path="results/control_noaux.json",
         out_dir="paper"):
    with open(pilot_path, encoding="utf-8") as f:
        pilot = json.load(f)
    with open(cost_path, encoding="utf-8") as f:
        cost = json.load(f)
    ctrl = None
    if ctrl_path and os.path.exists(ctrl_path):
        with open(ctrl_path, encoding="utf-8") as f:
            ctrl = json.load(f)
    return pilot, cost, ctrl


def table_main(g, lengths, n_seeds, chance, present):
    L = []
    a = L.append
    a("\\begin{table*}[!t]")
    a("\\centering")
    a("\\caption{Every diagnostic for the four models at four evaluation "
      "lengths. Recall is pooled over " + str(n_seeds) + " seeds and shown "
      "with the spread across seeds. Blind chance is "
      f"{100 * chance:.1f} percent and returning any value present in the "
      f"context scores {100 * present:.1f} percent, so only the distance "
      "above the second floor is retrieval. Sink mass averages the softmax "
      "layers only, and the count of those layers is given in the last "
      "column.}")
    a("\\label{tab:main}")
    a("\\footnotesize")
    a("\\renewcommand{\\arraystretch}{1.15}")
    a("\\begin{tabular}{@{}llccccccc@{}}")
    a("\\toprule")
    a("\\textbf{Model} & \\textbf{Length} & \\textbf{Recall} & "
      "\\textbf{Sink mass} & \\textbf{Worst layer} & \\textbf{Activation} & "
      "\\textbf{Entropy} & \\textbf{Recency} & \\textbf{Softmax} \\\\")
    a(" & (tokens) & (\\%) & $\\sinkmass$ & sink & $\\maxact$ & "
      "$\\mathcal{H}$ & gap & layers \\\\")
    a("\\midrule")
    for vi, v in enumerate(ORDER):
        if v not in g:
            continue
        for i, ln in enumerate(lengths):
            acc = ms([r["evals"][i]["accuracy"] for r in g[v]])
            sk = ms([r["evals"][i]["sink_mass"] for r in g[v]])
            skm = ms([r["evals"][i]["sink_mass_max_layer"] for r in g[v]])
            ac = ms([r["evals"][i]["max_activation"] for r in g[v]])
            en = ms([r["evals"][i]["attn_entropy"] for r in g[v]])
            rg = ms([r["evals"][i]["recency_gap"] for r in g[v]])
            name = f"\\textbf{{{ROWNAME[v]}}}" if i == 0 else ""
            a(f"{name} & {ln} & {100 * acc[0]:.1f} $\\pm$ {100 * acc[1]:.1f}"
              f" & {sk[0]:.3f} & {skm[0]:.3f} & {ac[0]:.1f} & {en[0]:.2f}"
              f" & {100 * rg[0]:+.1f} & {SOFTMAX_LAYERS[v]} \\\\")
        if vi != len(ORDER) - 1:
            a("\\midrule")
    a("\\bottomrule")
    a("\\end{tabular}")
    a("\\end{table*}")
    return "\n".join(L)


def table_control(g, gc, lengths):
    L = []
    a = L.append
    a("\\begin{table}[!t]")
    a("\\centering")
    a("\\caption{The control that changes the objective instead of the "
      "architecture. Both rows use the same softmax stack, the same data "
      "and the same seeds. The only difference is whether the model has to "
      "predict at every position or only at the answer. Values at the "
      "training length.}")
    a("\\label{tab:control}")
    a("\\footnotesize")
    a("\\renewcommand{\\arraystretch}{1.15}")
    a("\\begin{tabular}{@{}lcccc@{}}")
    a("\\toprule")
    a("\\textbf{Objective} & \\textbf{Sink} & \\textbf{Worst} & "
      "\\textbf{Activation} & \\textbf{Recall} \\\\")
    a(" & mass & layer & $\\maxact$ & (\\%) \\\\")
    a("\\midrule")
    rows = []
    if gc:
        k = list(gc.keys())[0]
        rows.append(("Answer position only", gc[k]))
    if "dense" in g:
        rows.append(("Every position", g["dense"]))
    for label, runs in rows:
        sk = ms([r["evals"][0]["sink_mass"] for r in runs])
        skm = ms([r["evals"][0]["sink_mass_max_layer"] for r in runs])
        ac = ms([r["evals"][0]["max_activation"] for r in runs])
        acc = ms([r["evals"][0]["accuracy"] for r in runs])
        a(f"{label} & {sk[0]:.3f} $\\pm$ {sk[1]:.3f} & {skm[0]:.3f} & "
          f"{ac[0]:.1f} & {100 * acc[0]:.1f} \\\\")
    a("\\bottomrule")
    a("\\end{tabular}")
    a("\\end{table}")
    return "\n".join(L)


def table_ablation(g, train_len):
    L = []
    a = L.append
    a("\\begin{table}[!t]")
    a("\\centering")
    a("\\caption{Each row adds one mechanism to the row above it. The third "
      "column compares sink mass against the untreated baseline in the "
      f"first row. Values at the training length of {train_len} tokens.}}")
    a("\\label{tab:ablation}")
    a("\\footnotesize")
    a("\\renewcommand{\\arraystretch}{1.15}")
    a("\\begin{tabular}{@{}lccc@{}}")
    a("\\toprule")
    a("\\textbf{Model} & \\textbf{Sink mass} & \\textbf{Against} & "
      "\\textbf{Recency gap} \\\\")
    a(" & $\\sinkmass$ & baseline & (points) \\\\")
    a("\\midrule")
    base = None
    for v in ORDER:
        if v not in g:
            continue
        sk = ms([r["evals"][0]["sink_mass"] for r in g[v]])
        rg = ms([r["evals"][0]["recency_gap"] for r in g[v]])
        if base is None:
            base, delta = sk[0], "baseline"
        else:
            delta = f"{100 * (sk[0] - base) / base:+.0f}\\%"
        a(f"{ROWNAME[v]} & {sk[0]:.3f} $\\pm$ {sk[1]:.3f} & {delta} & "
          f"{100 * rg[0]:+.1f} $\\pm$ {100 * rg[1]:.1f} \\\\")
    a("\\bottomrule")
    a("\\end{tabular}")
    a("\\end{table}")
    return "\n".join(L)


def table_quartiles(g, lengths):
    last = len(lengths) - 1
    L = []
    a = L.append
    a("\\begin{table}[!t]")
    a("\\centering")
    a("\\caption{Recall by quarter of the context at the longest evaluation "
      f"length of {lengths[last]} tokens, in percent. Q1 is the opening of "
      "the context and Q4 is nearest the question. The last column is the "
      "recency gap of Equation~\\ref{eq:recency}.}")
    a("\\label{tab:quartiles}")
    a("\\footnotesize")
    a("\\renewcommand{\\arraystretch}{1.15}")
    a("\\begin{tabular}{@{}lccccc@{}}")
    a("\\toprule")
    a("\\textbf{Model} & \\textbf{Q1} & \\textbf{Q2} & \\textbf{Q3} & "
      "\\textbf{Q4} & \\textbf{Gap} \\\\")
    a("\\midrule")
    for v in ORDER:
        if v not in g:
            continue
        qs = [ms([r["evals"][last][f"acc_q{k}"] for r in g[v]])[0]
              for k in range(1, 5)]
        gap = ms([r["evals"][last]["recency_gap"] for r in g[v]])
        a(f"{ROWNAME[v]} & " + " & ".join(f"{100 * q:.1f}" for q in qs) +
          f" & {100 * gap[0]:+.1f} \\\\")
    a("\\bottomrule")
    a("\\end{tabular}")
    a("\\end{table}")
    return "\n".join(L)


def table_seeds(g, train_len, n_seeds):
    L = []
    a = L.append
    a("\\begin{table}[!t]")
    a("\\centering")
    a("\\caption{Seed behaviour at the training length. The interval column "
      "is the Wilson half width for one trained model. The spread column is "
      f"the standard deviation across {n_seeds} seeds. A difference between "
      "two rows counts only if it exceeds both.}")
    a("\\label{tab:seeds}")
    a("\\footnotesize")
    a("\\renewcommand{\\arraystretch}{1.15}")
    a("\\begin{tabular}{@{}lcccc@{}}")
    a("\\toprule")
    a("\\textbf{Model} & \\textbf{Recall} & \\textbf{Interval} & "
      "\\textbf{Spread} & \\textbf{Sink spread} \\\\")
    a(" & (\\%) & (points) & (points) & $\\sinkmass$ \\\\")
    a("\\midrule")
    for v in ORDER:
        if v not in g:
            continue
        acc = ms([r["evals"][0]["accuracy"] for r in g[v]])
        hw = ms([(r["evals"][0]["ci_high"] - r["evals"][0]["ci_low"]) / 2
                 for r in g[v]])
        sk = ms([r["evals"][0]["sink_mass"] for r in g[v]])
        a(f"{ROWNAME[v]} & {100 * acc[0]:.1f} & $\\pm${100 * hw[0]:.1f} & "
          f"$\\pm${100 * acc[1]:.1f} & $\\pm${sk[1]:.3f} \\\\")
    a("\\bottomrule")
    a("\\end{tabular}")
    a("\\end{table}")
    return "\n".join(L)


def rows_published(g, lengths):
    """The rows of the comparison table that come from our own runs."""
    out = []
    for v in ORDER:
        if v not in g:
            continue
        sk = ms([r["evals"][0]["sink_mass"] for r in g[v]])[0]
        ac = ms([r["evals"][0]["max_activation"] for r in g[v]])[0]
        out.append(f"Ours, {ROWNAME[v].lower()} & {sk:.3f} & {ac:.0f} & "
                   "this work \\\\")
    return "\n".join(out)


def plot_lines(g, lengths, getter, scale=1.0, places=4):
    out = []
    for v in ORDER:
        if v not in g:
            continue
        pts = " ".join(
            f"({ln},{scale * ms([getter(r['evals'][i]) for r in g[v]])[0]:.{places}f})"
            for i, ln in enumerate(lengths))
        out.append(f"\\addplot[color={COLOUR[v]}, mark={MARK[v]}, "
                   f"mark size=1.5pt, line width=1.0pt] coordinates {{{pts}}};")
    return "\n".join(out)


def plot_depth(g, key):
    out = []
    for v in ORDER:
        if v not in g or key not in g[v][0]["depth_profile"]:
            continue
        n = len(g[v][0]["depth_profile"][key])
        pts = []
        for j in range(n):
            d = g[v][0]["depth_profile"][key][j]["depth"]
            acc = ms([r["depth_profile"][key][j]["acc"] for r in g[v]])[0]
            pts.append(f"({d:.2f},{100 * acc:.2f})")
        out.append(f"\\addplot[color={COLOUR[v]}, mark={MARK[v]}, "
                   f"mark size=1.3pt, line width=1.0pt] coordinates "
                   f"{{{' '.join(pts)}}};")
    return "\n".join(out)


def plot_loss(g):
    out = []
    for v in ORDER:
        if v not in g:
            continue
        steps = [c["step"] for c in g[v][0]["train_curve"]]
        pts = " ".join(
            f"({s},{ms([r['train_curve'][k]['loss'] for r in g[v]])[0]:.3f})"
            for k, s in enumerate(steps))
        out.append(f"\\addplot[color={COLOUR[v]}, mark={MARK[v]}, "
                   f"mark size=1.2pt, line width=0.9pt] coordinates "
                   f"{{{pts}}};")
    return "\n".join(out)


def plot_cache(cost):
    rows = cost["rows"]
    def blk(key, colour, mark, extra=""):
        pts = " ".join(f"({r['tokens']},{r[key]:.3f})" for r in rows)
        return (f"\\addplot[color={colour}, mark={mark}, mark size=1.5pt, "
                f"line width=1.0pt{extra}] coordinates {{{pts}}};")
    return "\n".join([
        blk("hybrid_total_gib", "kdaColor", "*"),
        blk("dense_cache_gib", "sinkColor", "square*"),
        blk("linear_state_gib", "gateColor", "triangle*", ", dashed"),
    ])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="results/pilot.json")
    ap.add_argument("--cost", default="results/costmodel.json")
    ap.add_argument("--control", default="results/control_noaux.json")
    ap.add_argument("--out-dir", default="paper")
    a = ap.parse_args()
    pilot, cost, ctrl = load(a.pilot, a.cost, a.control)
    cfg = pilot["config"]
    g = group(pilot["runs"])
    gc = group(ctrl["runs"]) if ctrl else None
    lengths = [e["seq_len"] for e in pilot["runs"][0]["evals"]]
    n_seeds = len({r["seed"] for r in pilot["runs"]})
    chance = 1.0 / cfg["n_values"]
    present = 1.0 / (cfg["n_distractors"] + 1)
    last_key = str(lengths[-1])

    frag = {
        "TABLE_MAIN": table_main(g, lengths, n_seeds, chance, present),
        "TABLE_CONTROL": table_control(g, gc, lengths),
        "TABLE_ABLATION": table_ablation(g, cfg["train_len"]),
        "TABLE_QUARTILES": table_quartiles(g, lengths),
        "TABLE_SEEDS": table_seeds(g, cfg["train_len"], n_seeds),
        "ROWS_PUBLISHED": rows_published(g, lengths),
        "PLOT_SINK": plot_lines(g, lengths, lambda e: e["sink_mass"]),
        "PLOT_ACC": plot_lines(g, lengths, lambda e: e["accuracy"],
                               scale=100.0, places=2),
        "PLOT_DEPTH": plot_depth(g, last_key),
        "PLOT_LOSS": plot_loss(g),
        "PLOT_CACHE": plot_cache(cost),
    }

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "fragments.json"), "w",
              encoding="utf-8") as f:
        json.dump(frag, f, indent=2)
    with open(os.path.join(a.out_dir, "results_tables.tex"), "w",
              encoding="utf-8") as f:
        for k, v in frag.items():
            f.write(f"% ===== {k} =====\n{v}\n\n")

    # A compact digest for writing the prose against.
    print("=" * 74)
    print(f"seeds {n_seeds}  steps {cfg['steps']}  train len {cfg['train_len']}"
          f"  chance {100 * chance:.1f}%  present-value floor "
          f"{100 * present:.1f}%")
    print("=" * 74)
    hdr = f"{'model':<20}{'len':>6}{'recall':>16}{'sink':>8}{'worst':>8}" \
          f"{'maxact':>9}{'ent':>7}{'recency':>9}"
    print(hdr)
    allg = dict(g)
    if gc:
        allg.update(gc)
    for v in list(ORDER) + [k for k in allg if k not in ORDER]:
        if v not in allg:
            continue
        for i, ln in enumerate(lengths):
            acc = ms([r["evals"][i]["accuracy"] for r in allg[v]])
            sk = ms([r["evals"][i]["sink_mass"] for r in allg[v]])
            skm = ms([r["evals"][i]["sink_mass_max_layer"] for r in allg[v]])
            ac = ms([r["evals"][i]["max_activation"] for r in allg[v]])
            en = ms([r["evals"][i]["attn_entropy"] for r in allg[v]])
            rg = ms([r["evals"][i]["recency_gap"] for r in allg[v]])
            print(f"{ROWNAME.get(v, v):<20}{ln:>6}"
                  f"{100 * acc[0]:>10.1f}+-{100 * acc[1]:<4.1f}"
                  f"{sk[0]:>8.3f}{skm[0]:>8.3f}{ac[0]:>9.1f}{en[0]:>7.2f}"
                  f"{100 * rg[0]:>+9.1f}")
        print("-" * 74)
    print("\ndepth profile at length", last_key)
    for v in ORDER:
        if v not in g:
            continue
        n = len(g[v][0]["depth_profile"][last_key])
        vals = [100 * ms([r["depth_profile"][last_key][j]["acc"]
                          for r in g[v]])[0] for j in range(n)]
        print(f"{ROWNAME[v]:<20}" + " ".join(f"{x:5.1f}" for x in vals))
    print("\ntrain minutes per model")
    for v in ORDER:
        if v in g:
            print(f"{ROWNAME[v]:<20}{ms([r['train_seconds'] for r in g[v]])[0] / 60:6.1f}"
                  f"   params {g[v][0]['params']}")
    print("\nwrote paper/fragments.json and paper/results_tables.tex")


if __name__ == "__main__":
    main()
