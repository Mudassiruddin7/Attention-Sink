"""Turn raw run files into every table, figure and quoted number in the paper.

    python scripts/make_report.py

Reads   results/runs/main/*.json   (controlled models, one file per run)
        results/hf/*.json          (released checkpoints)
Writes  paper/generated/*.tex      (table bodies)
        paper/figures/*.pdf, *.png
        results/summary.json       (every number the prose quotes)
        results/summary.md         (the same, readable)

Uncertainty is always across independently trained seeds: mean and a 95%
t interval. Differences between conditions use Welch's t interval. Rank
correlations across runs carry a percentile bootstrap interval.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats as sst

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.metrics import wilson_interval   # noqa: E402

LADDER = ["softmax", "gate", "hybrid", "hybrid_nope", "hybrid_attnres"]
SIDE = ["sinklogit", "softpick", "hybrid_nodelta"]
KNOB = ["softmax", "gate", "hybrid"]
LABEL = {"softmax": "Softmax", "gate": "+ output gate", "hybrid": "+ 3:1 delta-rule layers",
         "hybrid_nope": "+ no positional encoding", "hybrid_attnres": "+ attention residuals",
         "sinklogit": "Softmax + sink logit", "softpick": "Softpick",
         "hybrid_nodelta": "Hybrid, no delta rule", "hybrid_nogate": "Softmax + delta-rule layers",
         "nope": "Softmax, no positional encoding", "attnres": "Softmax + attention residuals",
         "bka": "Bound-Key Attention", "bka_rope": "Bound-Key Attention with RoPE"}
SHORT = {"softmax": "Softmax", "gate": "Gated", "hybrid": "Hybrid", "hybrid_nope": "Hybrid NoPE",
         "hybrid_attnres": "Hybrid AttnRes", "sinklogit": "Sink logit", "softpick": "Softpick",
         "hybrid_nodelta": "Hybrid no-delta", "hybrid_nogate": "Delta layers", "nope": "NoPE",
         "attnres": "AttnRes", "bka": "BKA", "bka_rope": "BKA RoPE"}
COLOR = {"softmax": "#0072B2", "gate": "#E69F00", "hybrid": "#009E73", "hybrid_nope": "#56B4E9",
         "hybrid_attnres": "#CC79A7", "sinklogit": "#D55E00", "softpick": "#999999",
         "hybrid_nodelta": "#6B8E23", "hybrid_nogate": "#8B4513", "nope": "#2F4F4F",
         "attnres": "#9370DB", "bka": "#D55E00", "bka_rope": "#F0A500"}
TARGET = {"gate": "forced output with nothing to read", "hybrid_nogate": "cost of a growing cache",
          "nope": "positional extrapolation", "attnres": "dilution of early layers",
          "sinklogit": "a place to park attention", "softpick": "the sum-to-one constraint"}
SINGLE = ["gate", "hybrid_nogate", "nope", "attnres", "sinklogit", "softpick"]
MAIN_WARMUP = 1500
MIN_RECALL = 0.9
TRAIN_LEN = 256
RNG = np.random.default_rng(0)
SUMMARY: dict = {}


# ---------------------------------------------------------------------------
# loading and statistics
# ---------------------------------------------------------------------------

def add_depth_fit(e):
    """Weighted least squares of recall on (d - 0.5) and (d - 0.5)^2 over the depth bins.

    The slope is positive when evidence near the question is favoured and the
    curvature is positive when both ends beat the middle, so the two numbers
    summarise the shape of p(d) rather than only its end points.
    """
    prof = [b for b in e.get("profile", []) if b["n"] > 0]
    if len(prof) < 4:
        return
    d = np.array([b["centre"] for b in prof]) - 0.5
    y = np.array([b["acc"] for b in prof])
    w = np.sqrt(np.array([b["n"] for b in prof], dtype=float))
    X = np.stack([np.ones_like(d), d, d ** 2], 1)
    beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    e["fit_slope"], e["fit_curv"] = float(beta[1]), float(beta[2])


def load_runs(d="results/runs/main"):
    runs = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        for e in r["evals"]:
            add_depth_fit(e)
        runs.append({"name": os.path.basename(p)[:-5], "variant": r["run"]["variant"],
                     "warmup": int(r["run"].get("copy_warmup", 0)),
                     "learned": any(e["seq_len"] == TRAIN_LEN and e["recall"] >= MIN_RECALL
                                    for e in r["evals"]),
                     "seed": r["run"]["seed"], "p_noop": float(r["task_config"]["p_noop"]),
                     "skew": float(r["task_config"]["query_skew"]),
                     "evals": {e["seq_len"]: e for e in r["evals"]}, "probes": r["probes"],
                     "curve": r["curve"], "params": r["params"],
                     "train_seconds": r["train_seconds"], "floors": r["floors"]})
    return runs


def pick(runs, variant=None, p_noop=0.5, skew=0.0, warmup=None, learned_only=True):
    w = MAIN_WARMUP if warmup is None else warmup
    return [r for r in runs if (variant is None or r["variant"] == variant)
            and abs(r["p_noop"] - p_noop) < 1e-9 and abs(r["skew"] - skew) < 1e-9
            and r["warmup"] == w and (r["learned"] or not learned_only)]


def values(runs, key, length=TRAIN_LEN, scale=1.0):
    out = []
    for r in runs:
        v = r["evals"].get(length, {}).get(key)
        if v is not None and np.isfinite(v):
            out.append(scale * v)
    return np.array(out, dtype=float)


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return float(v[0]), float("nan"), 1
    h = sst.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean()), float(h), len(v)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return float(d), 0.0, 0.0
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(d), float(sst.t.ppf(0.975, dof) * se), float(2 * sst.t.sf(abs(d / se), dof))


def spearman_boot(x, y, n=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = sst.spearmanr(x, y).statistic
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    boots = [sst.spearmanr(x[i], y[i]).statistic for i in idx]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


def slope_ci(x, y):
    res = sst.linregress(x, y)
    h = sst.t.ppf(0.975, len(x) - 2) * res.stderr
    return float(res.slope), float(h), float(res.pvalue)


def cell(v, digits=2, scale=1.0, signed=False):
    m, h, n = mean_ci(np.asarray(v) * scale)
    if n == 0:
        return "--"
    s = f"{m:+.{digits}f}" if signed else f"{m:.{digits}f}"
    return s if not np.isfinite(h) else s + f"{{\\scriptsize$\\pm${h:.{digits}f}}}"


OUT = "."


def write(path, text):
    path = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def table_learnability(runs):
    """How often each variant learns retrieval at all, over every trained seed."""
    order = LADDER + SINGLE + SIDE
    variants = sorted({r["variant"] for r in runs}, key=lambda v: order.index(v) if v in order else 99)
    out, lines = {}, []
    for v in variants:
        rs = pick(runs, v, learned_only=False)
        if not rs:
            continue
        learned = [r for r in rs if r["learned"]]
        rec = [r["evals"][TRAIN_LEN]["recall"] for r in rs]
        out[v] = {"learned": len(learned), "trained": len(rs),
                  "recall_by_seed": {str(r["seed"]): r["evals"][TRAIN_LEN]["recall"] for r in rs},
                  "long_recall_learned": mean_ci(values(learned, "recall", max(rs[0]["evals"]), 100))}
        lines.append(f"{LABEL.get(v, v)} & {len(learned)}/{len(rs)} & "
                     + ", ".join(f"{100 * x:.0f}" for x in rec) + " \\\\")
    write("paper/generated/table_learnability.tex", "\n".join(lines) + "\n")
    SUMMARY["learnability"] = out


def table_ladder(runs):
    lines, summ = [], {}
    long_len = max(runs[0]["evals"])
    for group in (LADDER, SIDE):
        for v in group:
            rs = pick(runs, v)
            if not rs:
                continue
            row = [LABEL[v], str(len(rs)),
                   cell(values(rs, "recall"), 1, 100),
                   cell(values(rs, "recall", long_len), 1, 100),
                   cell(values(rs, "sink_mass"), 3),
                   cell(values(rs, "sink_ratio"), 1),
                   cell(values(rs, "sink_noop"), 3),
                   cell(values(rs, "sink_copy"), 3),
                   cell(values(rs, "act_max"), 0),
                   cell(values(rs, "recency_gap", long_len), 1, 100, signed=True)]
            lines.append(" & ".join(row) + " \\\\")
            summ[v] = {f"{k}@{L}": mean_ci(values(rs, k, L, s)) for k, L, s in [
                ("recall", TRAIN_LEN, 100), ("recall", long_len, 100), ("sink_mass", TRAIN_LEN, 1),
                ("sink_ratio", TRAIN_LEN, 1), ("sink_rate", TRAIN_LEN, 1), ("sink_noop", TRAIN_LEN, 1),
                ("sink_copy", TRAIN_LEN, 1), ("sink_answer", TRAIN_LEN, 1), ("act_max", TRAIN_LEN, 1),
                ("gate_mean", TRAIN_LEN, 1), ("gate_noop", TRAIN_LEN, 1), ("gate_copy", TRAIN_LEN, 1),
                ("virtual_sink", TRAIN_LEN, 1), ("recency_gap", TRAIN_LEN, 100),
                ("recency_gap", long_len, 100), ("middle_dip", long_len, 100)]
                if len(values(rs, k, L)) > 0} | {"n": len(rs)}
            ratio = [r["evals"][TRAIN_LEN]["sink_noop"] / r["evals"][TRAIN_LEN]["sink_copy"]
                     for r in rs if r["evals"].get(TRAIN_LEN, {}).get("sink_copy")]
            if ratio:
                summ[v][f"noop_over_copy@{TRAIN_LEN}"] = mean_ci(ratio)
        lines.append("\\midrule")
    write("paper/generated/table_ladder.tex", "\n".join(lines[:-1]) + "\n")
    SUMMARY["ladder"] = summ


def table_steps(runs):
    """Incremental ablation: each ladder step against the step before it."""
    lines, summ = [], {}
    long_len = max(runs[0]["evals"])
    for prev, cur in zip(LADDER[:-1], LADDER[1:]):
        a, b = pick(runs, prev), pick(runs, cur)
        if len(a) < 2 or len(b) < 2:
            continue
        cols = []
        for key, L, scale, dg in [("sink_ratio", TRAIN_LEN, 1, 1), ("sink_noop", TRAIN_LEN, 1, 3),
                                  ("recall", TRAIN_LEN, 100, 1), ("recall", long_len, 100, 1),
                                  ("recency_gap", long_len, 100, 1)]:
            d, h, p = welch(values(a, key, L, scale), values(b, key, L, scale))
            cols.append(f"{d:+.{dg}f}{{\\scriptsize$\\pm${h:.{dg}f}}}")
            summ[f"{cur}:{key}@{L}"] = {"diff": d, "ci": h, "p": p}
        lines.append(LABEL[cur] + " & " + " & ".join(cols) + " \\\\")
    write("paper/generated/table_steps.tex", "\n".join(lines) + "\n")
    SUMMARY["steps"] = summ


def dose_response(runs):
    summ, lines = {}, []
    ps = sorted({r["p_noop"] for r in runs if r["skew"] == 0.0 and r["warmup"] == MAIN_WARMUP})
    for v in KNOB:
        xs, ys, gaps = [], [], []
        row = [SHORT[v]]
        for p in ps:
            rs = pick(runs, v, p_noop=p)
            sr = values(rs, "sink_ratio")
            xs += [p] * len(sr)
            ys += list(sr)
            gaps += list(values(rs, "recency_gap", max(runs[0]["evals"]), 100))
            row.append(cell(sr, 1))
        if len(set(xs)) > 2:
            summ[v] = {"slope": slope_ci(xs, ys), "rho": spearman_boot(xs, ys)}
            if len(gaps) == len(xs):
                summ[v]["rho_gap"] = spearman_boot(xs, gaps)
        lines.append(" & ".join(row) + " \\\\")
    header = "Model & " + " & ".join(f"$p={p:g}$" for p in ps) + " \\\\"
    write("paper/generated/table_dose.tex", header + "\n\\midrule\n" + "\n".join(lines) + "\n")
    SUMMARY["dose"] = summ


def skew_effect(runs):
    summ, lines = {}, []
    long_len = max(runs[0]["evals"])
    for v in KNOB:
        base = pick(runs, v, skew=0.0)
        row = [SHORT[v]]
        for g in (-4.0, 4.0):
            rs = pick(runs, v, skew=g)
            if len(rs) < 2 or len(base) < 2:
                row += ["--", "--"]
                continue
            dg = welch(values(base, "recency_gap", TRAIN_LEN, 100), values(rs, "recency_gap", TRAIN_LEN, 100))
            ds = welch(values(base, "sink_ratio"), values(rs, "sink_ratio"))
            summ[f"{v}:skew{g:+g}"] = {"gap_diff": dg, "sink_ratio_diff": ds,
                                        "gap_long": mean_ci(values(rs, "recency_gap", long_len, 100))}
            row += [f"{dg[0]:+.1f}{{\\scriptsize$\\pm${dg[1]:.1f}}}",
                    f"{ds[0]:+.1f}{{\\scriptsize$\\pm${ds[1]:.1f}}}"]
        lines.append(" & ".join(row) + " \\\\")
    write("paper/generated/table_skew.tex", "\n".join(lines) + "\n")
    SUMMARY["skew"] = summ


def noop_enrichment(runs):
    """Is first-token attention higher at queries with nothing to read than at copy queries?"""
    base = [r for r in runs if r["skew"] == 0.0 and abs(r["p_noop"] - 0.5) < 1e-9
            and r["warmup"] == MAIN_WARMUP and r["learned"]
            and r["evals"][TRAIN_LEN].get("sink_copy") is not None]
    if not base:
        return
    d = np.array([r["evals"][TRAIN_LEN]["sink_noop"] - r["evals"][TRAIN_LEN]["sink_copy"] for r in base])
    ratio = [r["evals"][TRAIN_LEN]["sink_noop"] / r["evals"][TRAIN_LEN]["sink_copy"] for r in base]
    pos = int((d > 0).sum())
    SUMMARY["noop_enrichment"] = {
        "runs": len(d), "noop_higher": pos,
        "sign_test_p": float(sst.binomtest(pos, len(d), 0.5).pvalue),
        "wilcoxon_p": float(sst.wilcoxon(d).pvalue) if len(d) >= 6 else None,
        "mean_diff": mean_ci(d), "median_ratio": float(np.median(ratio))}


def ppl_summary(d="results/interventions"):
    out = {}
    for q in sorted(glob.glob(os.path.join(d, "ppl_*.json"))):
        r = json.load(open(q, encoding="utf-8"))
        out[r["model"].split("/")[-1]] = [{k: v for k, v in row.items() if k != "nll_per_window"}
                                          for row in r["rows"]]
    if out:
        SUMMARY["perplexity_check"] = out


def dissociation(runs):
    long_len = max(runs[0]["evals"])
    base = [r for r in runs if r["skew"] == 0.0 and r["warmup"] == MAIN_WARMUP and r["learned"]]
    out = {}
    for L in (TRAIN_LEN, long_len):
        rs = [r for r in base if r["evals"].get(L, {}).get("sink_ratio") is not None]
        x = [r["evals"][L]["sink_ratio"] for r in rs]
        for key in ("recency_gap", "middle_dip", "recall"):
            y = [r["evals"][L][key] for r in rs]
            if len(x) > 5:
                out[f"sink_ratio~{key}@{L}"] = spearman_boot(x, y) + (len(x),)
    SUMMARY["dissociation"] = out


def step_ms(bench, v):
    rows = [r for r in (bench or {}).get("rows", []) if r["variant"] == v and r["amp"]]
    return f"{rows[0]['ms_per_step']:.0f}" if rows else "--"


def table_mechanisms(runs, bench=None):
    """Each mechanism alone against the softmax baseline, with Welch 95% intervals."""
    long_len = max(runs[0]["evals"])
    base = pick(runs, "softmax")
    if len(base) < 2:
        return
    cols = [("sink_ratio", TRAIN_LEN, 1, 1), ("sink_noop", TRAIN_LEN, 1, 3),
            ("recall", TRAIN_LEN, 100, 1), ("recall", long_len, 100, 1),
            ("fit_slope", long_len, 100, 0), ("fit_curv", long_len, 100, 0),
            ("act_max", TRAIN_LEN, 1, 0)]
    lines, summ = [], {}
    for v in SINGLE:
        rs = pick(runs, v)
        if len(rs) < 2:
            continue
        cells, entry = [], {"params": rs[0]["params"], "n": len(rs)}
        for key, L, scale, dg in cols:
            a, b = values(base, key, L, scale), values(rs, key, L, scale)
            if len(a) < 2 or len(b) < 2:
                cells.append("--")
                continue
            d, h, pv = welch(a, b)
            cells.append(f"{d:+.{dg}f}{{\\scriptsize$\\pm${h:.{dg}f}}}")
            entry[f"{key}@{L}"] = {"diff": d, "ci": h, "p": pv}
        lines.append(f"{LABEL[v]} & {TARGET.get(v, '')} & " + " & ".join(cells)
                     + f" & {step_ms(bench, v)} \\\\")
        summ[v] = entry
    write("paper/generated/table_mechanisms.tex", "\n".join(lines) + "\n")
    SUMMARY["mechanisms"] = summ


def gate_by_delta(runs, n_boot=5000):
    """2x2 of output gate by delta-rule layers: the gate's effect in each layout and the difference."""
    cells = {k: pick(runs, k) for k in ("softmax", "gate", "hybrid_nogate", "hybrid")}
    if any(len(v) < 2 for v in cells.values()):
        return
    long_len = max(runs[0]["evals"])
    out = {}
    for key, L, scale in [("sink_ratio", TRAIN_LEN, 1), ("sink_noop", TRAIN_LEN, 1),
                          ("recall", long_len, 100), ("fit_slope", long_len, 100)]:
        vals = {k: values(v, key, L, scale) for k, v in cells.items()}
        if any(len(v) < 2 for v in vals.values()):
            continue

        def contrast(idx):
            m = {k: vals[k][idx[k]].mean() for k in vals}
            return m["gate"] - m["softmax"], m["hybrid"] - m["hybrid_nogate"]
        g_soft, g_hyb = contrast({k: np.arange(len(v)) for k, v in vals.items()})
        boots = np.array([contrast({k: RNG.integers(0, len(v), len(v)) for k, v in vals.items()})
                          for _ in range(n_boot)])

        def pct(z):
            return [float(x) for x in np.percentile(z, [2.5, 97.5])]
        out[f"{key}@{L}"] = {"gate_in_softmax": [float(g_soft)] + pct(boots[:, 0]),
                             "gate_in_hybrid": [float(g_hyb)] + pct(boots[:, 1]),
                             "interaction": [float(g_hyb - g_soft)] + pct(boots[:, 1] - boots[:, 0])}
    SUMMARY["gate_by_delta"] = out


def warmup_ablation(runs):
    out, lines = {}, []
    for v in ["softmax", "gate", "hybrid"]:
        with_w, without = pick(runs, v, learned_only=False), pick(runs, v, warmup=0, learned_only=False)
        if not with_w or not without:
            continue
        row, entry = [SHORT[v]], {}
        for key, scale, dg in [("recall", 100, 1), ("ce_copy", 1, 2), ("sink_ratio", 1, 1),
                               ("sink_noop", 1, 3)]:
            a, b = values(without, key, TRAIN_LEN), values(with_w, key, TRAIN_LEN)
            row += [cell(a, dg, scale), cell(b, dg, scale)]
            entry[key] = {"without": mean_ci(a * scale), "with": mean_ci(b * scale)}
        lines.append(" & ".join(row) + " \\\\")
        out[v] = entry
    if lines:
        write("paper/generated/table_warmup.tex", "\n".join(lines) + "\n")
    SUMMARY["warmup"] = out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures(runs, hf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif"})
    os.makedirs(os.path.join(OUT, "paper/figures"), exist_ok=True)
    long_len = max(runs[0]["evals"])

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"paper/figures/{name}.pdf"))
        fig.savefig(os.path.join(OUT, f"paper/figures/{name}.png"), dpi=200)
        plt.close(fig)

    # dose response: sink by position type and recency gap against p_noop
    ps = sorted({r["p_noop"] for r in runs if r["skew"] == 0.0 and r["warmup"] == MAIN_WARMUP})
    if len(ps) > 2:
        fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.1))
        for v in KNOB:
            for j, (key, L, scale) in enumerate([("sink_ratio", TRAIN_LEN, 1),
                                                 ("sink_noop", TRAIN_LEN, 1),
                                                 ("recency_gap", long_len, 100)]):
                m = [mean_ci(values(pick(runs, v, p_noop=p), key, L, scale)) for p in ps]
                ax[j].errorbar(ps, [a for a, _, _ in m], yerr=[0 if not np.isfinite(h) else h for _, h, _ in m],
                               marker="o", ms=3, lw=1.2, capsize=2, color=COLOR[v], label=SHORT[v])
        ax[0].set_ylabel("sink mass / uniform")
        ax[1].set_ylabel("sink mass at no-op queries")
        ax[2].set_ylabel(f"recency gap at {long_len} (pts)")
        for a in ax:
            a.set_xlabel("share of no-op segments $p$")
        ax[2].axhline(0, color="k", lw=0.5)
        ax[0].legend(frameon=False)
        save(fig, "dose_response")

    # sink by query type for every variant at the base setting
    vs = [v for v in LADDER + SIDE if pick(runs, v)]
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    width = 0.27
    for j, (key, name) in enumerate([("sink_noop", "no-op queries"), ("sink_copy", "copy queries"),
                                     ("sink_answer", "answer queries")]):
        m = [mean_ci(values(pick(runs, v), key)) for v in vs]
        ax.bar(np.arange(len(vs)) + (j - 1) * width, [a for a, _, _ in m], width,
               yerr=[0 if not np.isfinite(h) else h for _, h, _ in m], capsize=1.5,
               color=["#444444", "#999999", "#DDDDDD"][j], edgecolor="k", lw=0.4, label=name)
    ax.set_xticks(np.arange(len(vs)), [SHORT[v] for v in vs])
    ax.set_ylabel("attention on position 0")
    ax.legend(frameon=False, ncol=3)
    save(fig, "sink_by_query_type")

    # depth profiles at the training length and the longest length
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.2), sharey=True)
    for j, L in enumerate((TRAIN_LEN, long_len)):
        for v in LADDER:
            rs = pick(runs, v)
            if not rs:
                continue
            prof = np.array([[b["acc"] for b in r["evals"][L]["profile"]] for r in rs]) * 100
            centres = [b["centre"] for b in rs[0]["evals"][L]["profile"]]
            m = np.nanmean(prof, axis=0)
            if len(rs) > 1:
                h = sst.t.ppf(0.975, len(rs) - 1) * np.nanstd(prof, axis=0, ddof=1) / math.sqrt(len(rs))
                ax[j].fill_between(centres, m - h, m + h, color=COLOR[v], alpha=0.15, lw=0)
            ax[j].plot(centres, m, marker="o", ms=2.5, lw=1.2, color=COLOR[v], label=SHORT[v])
        ax[j].set_title(f"evaluated at {L} tokens" + (" (training length)" if L == TRAIN_LEN else ""))
        ax[j].set_xlabel("depth of the queried pair")
    ax[0].set_ylabel("recall (%)")
    ax[1].legend(frameon=False)
    save(fig, "depth_profiles")

    # each mechanism alone: p(d) at the training length and the longest length
    single = ["softmax"] + [v for v in SINGLE if pick(runs, v)]
    if len(single) > 1 and pick(runs, "softmax"):
        fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.2), sharey=True)
        for j, L in enumerate((TRAIN_LEN, long_len)):
            for v in single:
                rs = pick(runs, v)
                prof = np.array([[b["acc"] for b in r["evals"][L]["profile"]] for r in rs]) * 100
                centres = [b["centre"] for b in rs[0]["evals"][L]["profile"]]
                m = np.nanmean(prof, axis=0)
                if len(rs) > 1:
                    h = sst.t.ppf(0.975, len(rs) - 1) * np.nanstd(prof, axis=0, ddof=1) / math.sqrt(len(rs))
                    ax[j].fill_between(centres, m - h, m + h, color=COLOR[v], alpha=0.15, lw=0)
                ax[j].plot(centres, m, marker="o", ms=2.5, lw=1.2, color=COLOR[v], label=SHORT[v])
            ax[j].set_title(f"evaluated at {L} tokens")
            ax[j].set_xlabel("depth of the queried pair")
        ax[0].set_ylabel("recall (%)")
        ax[1].legend(frameon=False, fontsize=6)
        save(fig, "depth_profiles_single_factors")

    # dissociation scatter
    base = [r for r in runs if r["skew"] == 0.0 and r["warmup"] == MAIN_WARMUP and r["learned"]]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.2))
    for r in base:
        for j, L in enumerate((TRAIN_LEN, long_len)):
            e = r["evals"].get(L, {})
            if e.get("sink_ratio") is None:
                continue
            ax[j].scatter(e["sink_ratio"], 100 * e["recency_gap"], s=9, color=COLOR[r["variant"]],
                          alpha=0.8, lw=0)
    for j, L in enumerate((TRAIN_LEN, long_len)):
        ax[j].set_xlabel(f"sink mass / uniform at {L}")
        ax[j].axhline(0, color="k", lw=0.5)
        rho = SUMMARY.get("dissociation", {}).get(f"sink_ratio~recency_gap@{L}")
        if rho:
            ax[j].set_title(f"Spearman $\\rho$ = {rho[0]:.2f} [{rho[1]:.2f}, {rho[2]:.2f}]")
    ax[0].set_ylabel("recency gap (pts)")
    save(fig, "dissociation")

    # skew: depth profiles for softmax and hybrid under three training skews
    gs = sorted({r["skew"] for r in runs if r["warmup"] == MAIN_WARMUP})
    if len(gs) > 1:
        fig, ax = plt.subplots(1, len(KNOB), figsize=(6.8, 2.0), sharey=True)
        for j, v in enumerate(KNOB):
            for g, ls in zip(gs, ["--", "-", ":"]):
                rs = pick(runs, v, skew=g)
                if not rs:
                    continue
                prof = np.array([[b["acc"] for b in r["evals"][TRAIN_LEN]["profile"]] for r in rs]) * 100
                centres = [b["centre"] for b in rs[0]["evals"][TRAIN_LEN]["profile"]]
                ax[j].plot(centres, np.nanmean(prof, 0), ls=ls, lw=1.3, color=COLOR[v],
                           label=f"skew {g:+g}")
            ax[j].set_title(SHORT[v])
            ax[j].set_xlabel("depth of the queried pair")
        ax[0].set_ylabel("recall (%)")
        ax[0].legend(frameon=False)
        save(fig, "skew_profiles")

    # training dynamics of the sink at no-op and copy queries
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.0), sharey=True)
    for v in ["softmax", "gate", "hybrid"]:
        rs = pick(runs, v)
        if not rs:
            continue
        steps = [p["step"] for p in rs[0]["probes"]]
        for j, key in enumerate(("sink_noop", "sink_copy")):
            arr = np.array([[p.get(key, np.nan) for p in r["probes"]] for r in rs])
            ax[j].plot(steps, np.nanmean(arr, 0), marker="o", ms=2.5, lw=1.2, color=COLOR[v], label=SHORT[v])
    ax[0].set_title("no-op queries")
    ax[1].set_title("copy queries")
    for a in ax:
        a.set_xlabel("training step")
    ax[0].set_ylabel("attention on position 0")
    ax[0].legend(frameon=False)
    save(fig, "sink_dynamics")



# ---------------------------------------------------------------------------
# released checkpoints
# ---------------------------------------------------------------------------

def hf_figures(hf):
    if not hf:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif"})
    os.makedirs(os.path.join(OUT, "paper/figures"), exist_ok=True)
    names = sorted(hf)

    def label(L):
        return f"{round(L / 1024)}K" if L >= 1024 else str(L)

    fig, ax = plt.subplots(2, len(names), figsize=(6.8, 3.9), sharex=True, squeeze=False)
    for j, name in enumerate(names):
        d = hf[name]["by_length"]
        for L in sorted(d):
            prof = d[L]["profile"]
            ax[0, j].errorbar([p[0] for p in prof], [100 * p[1] for p in prof],
                              yerr=[[100 * (p[1] - p[2]) for p in prof], [100 * (p[3] - p[1]) for p in prof]],
                              marker="o", ms=2.5, lw=1.1, capsize=1.5, label=label(L))
            if "profile_logprob" in d[L]:
                pl = d[L]["profile_logprob"]
                ax[1, j].plot([p[0] for p in pl], [p[1] for p in pl], marker="o", ms=2.5, lw=1.2)
        ax[0, j].set_title(name)
        ax[1, j].set_xlabel("depth of the needle")
    ax[0, 0].set_ylabel("exact match (%)")
    ax[1, 0].set_ylabel("answer log probability")
    ax[0, -1].legend(frameon=False, title="context")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"paper/figures/hf_depth_profiles.{ext}"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, len(names), figsize=(6.8, 2.0), sharey=True, squeeze=False)
    for j, name in enumerate(names):
        d = hf[name]["by_length"]
        for L in sorted(d):
            spl = d[L].get("sink_per_layer")
            if spl:
                layers = sorted(spl, key=int)
                ax[0, j].plot([int(k) for k in layers], [spl[k] for k in layers], marker="o",
                              ms=2.5, lw=1.2, label=label(L))
        ax[0, j].set_title(name)
        ax[0, j].set_xlabel("layer index (softmax layers only)")
        from matplotlib.ticker import MaxNLocator
        ax[0, j].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0, 0].set_ylabel("attention on position 0")
    ax[0, -1].legend(frameon=False, title="context")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"paper/figures/hf_sink_per_layer.{ext}"), dpi=200)
    plt.close(fig)


HF_RAW: dict = {}


def depth_fit(depth, y, n_boot=2000):
    """Least squares y ~ 1 + (d - 0.5) + (d - 0.5)^2 with a percentile bootstrap over trials.

    A positive slope means evidence near the question is favoured; a positive
    curvature means the two ends beat the middle.
    """
    depth, y = np.asarray(depth, float), np.asarray(y, float)
    X = np.stack([np.ones_like(depth), depth - 0.5, (depth - 0.5) ** 2], 1)

    def fit(idx):
        return np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    beta = fit(np.arange(len(y)))
    boots = np.array([fit(RNG.integers(0, len(y), len(y))) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"slope": (float(beta[1]), float(lo[1]), float(hi[1])),
            "curvature": (float(beta[2]), float(lo[2]), float(hi[2]))}


def hf_compare(n_boot=2000):
    """Between-model differences at each length both models reached."""
    out = {}
    names = sorted(HF_RAW)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for L in sorted(set(HF_RAW[a]) & set(HF_RAW[b])):
                ca, da = HF_RAW[a][L]["correct"], HF_RAW[a][L]["depth"]
                cb, db = HF_RAW[b][L]["correct"], HF_RAW[b][L]["depth"]

                def gap(c, d):
                    q1, q4 = c[d < 0.25], c[d >= 0.75]
                    return (q4.mean() if len(q4) else np.nan) - (q1.mean() if len(q1) else np.nan)
                boots = []
                for _ in range(n_boot):
                    ia = RNG.integers(0, len(ca), len(ca))
                    ib = RNG.integers(0, len(cb), len(cb))
                    boots.append(gap(cb[ib], db[ib]) - gap(ca[ia], da[ia]))
                lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                out[f"{b} minus {a} @{L}"] = {
                    "recency_gap_diff": (float(gap(cb, db) - gap(ca, da)), float(lo), float(hi)),
                    "recall_diff": float(cb.mean() - ca.mean())}
    SUMMARY["hf_compare"] = out


def hf_summary(d="results/hf"):
    out = {}
    lines = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        name = r["model"].split("/")[-1]
        by_len = {}
        for L in sorted({x["length"] for x in r["rows"]}):
            rows = [x for x in r["rows"] if x["length"] == L]
            c = np.array([x["correct"] for x in rows], float)
            dep = np.array([x["depth_actual"] for x in rows], float)
            q = [c[(dep >= a) & (dep < b)].mean() if ((dep >= a) & (dep < b)).any() else np.nan
                 for a, b in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]]
            targets = sorted({x["depth_target"] for x in rows})
            prof = []
            for t in targets:
                ct = [x["correct"] for x in rows if x["depth_target"] == t]
                lo_t, hi_t = wilson_interval(int(sum(ct)), len(ct))
                prof.append((t, float(np.mean(ct)), lo_t, hi_t))
            sinks = [s for s in r["sinks"] if s["length"] == L]
            lo, hi = wilson_interval(int(c.sum()), len(c))
            entry = {"n": len(c), "recall": float(c.mean()), "ci": (lo, hi),
                     "q": [float(x) for x in q], "recency_gap": float(q[3] - q[0]),
                     "middle_dip": float((q[0] + q[3]) / 2 - (q[1] + q[2]) / 2), "profile": prof,
                     "tokens": float(np.mean([x["n_tokens"] for x in rows]))}
            HF_RAW.setdefault(name, {})[L] = {"correct": c, "depth": dep}
            if len(c) >= 20:
                entry["depth_fit_correct"] = depth_fit(dep, c)
            if all("answer_logprob" in x for x in rows):
                entry["logprob"] = mean_ci([x["answer_logprob"] for x in rows])
                if len(c) >= 20:
                    entry["depth_fit_logprob"] = depth_fit(dep, [x["answer_logprob"] for x in rows])
                entry["profile_logprob"] = [
                    (t, float(np.mean([x["answer_logprob"] for x in rows if x["depth_target"] == t])))
                    for t in targets]
            if sinks:
                for k in ("sink_mass", "sink_ratio", "sink_rate", "entropy_norm"):
                    entry[k] = mean_ci([s[k] for s in sinks])
                entry["layers_used"] = sinks[0]["layers_used"]
                layer_keys = sinks[0]["per_layer"].keys()
                entry["sink_per_layer"] = {k: float(np.mean([s["per_layer"][k]["a0"] for s in sinks
                                                             if k in s["per_layer"]]))
                                           for k in layer_keys}
                gm = [s["gate_mean"] for s in sinks if s.get("gate_mean") is not None]
                entry["gate_mean"] = float(np.mean(gm)) if gm else None
            by_len[L] = entry
            lines.append(f"{name} & {L} & {100 * entry['recall']:.1f} & "
                         + " & ".join(f"{100 * x:.0f}" for x in q)
                         + f" & {entry.get('sink_mass', (float('nan'),))[0]:.3f}"
                         + f" & {entry.get('sink_ratio', (float('nan'),))[0]:.1f} \\\\")
        out[name] = {"by_length": by_len, "gated_layers": r.get("gated_layers")}
    if lines:
        write("paper/generated/table_hf.tex", "\n".join(lines) + "\n")
    SUMMARY["hf"] = out
    return out


# ---------------------------------------------------------------------------
# interventions: attention pushed off position 0
# ---------------------------------------------------------------------------

def bias_label(b):
    return "$-\\infty$" if b == float("-inf") else f"{b:g}"


def load_interventions(d="results/interventions"):
    syn, hfi = None, {}
    path = os.path.join(d, "synthetic.json")
    if os.path.exists(path):
        syn = json.load(open(path, encoding="utf-8"))["rows"]
    for q in sorted(glob.glob(os.path.join(d, "hf_*.json"))):
        r = json.load(open(q, encoding="utf-8"))
        hfi[r["model"].split("/")[-1]] = r
    return syn, hfi


def synthetic_interventions(rows):
    """Paired over seeds: every bias level is compared with bias 0 on the same model."""
    if not rows:
        return
    for r in rows:
        add_depth_fit(r)
    order = LADDER + SINGLE + SIDE
    variants = sorted({r["variant"] for r in rows}, key=lambda v: order.index(v) if v in order else 99)
    lengths = sorted({r["seq_len"] for r in rows})
    biases = sorted({r["bias"] for r in rows}, reverse=True)
    out, lines = {}, []
    for v in variants:
        for b in biases:
            cells = [SHORT.get(v, v), bias_label(b)]
            for L in lengths:
                base = {r["seed"]: r for r in rows if r["variant"] == v and r["seq_len"] == L and r["bias"] == 0.0}
                cur = {r["seed"]: r for r in rows if r["variant"] == v and r["seq_len"] == L and r["bias"] == b}
                seeds = sorted(set(base) & set(cur))
                entry = {"n": len(seeds)}
                for key, scale in [("sink_mass", 1), ("sink_noop", 1), ("recall", 100),
                                   ("recency_gap", 100), ("fit_slope", 100), ("fit_curv", 100)]:
                    vals = np.array([cur[x][key] for x in seeds if cur[x].get(key) is not None], float)
                    diffs = np.array([cur[x][key] - base[x][key] for x in seeds
                                      if cur[x].get(key) is not None and base[x].get(key) is not None], float)
                    entry[key] = mean_ci(vals * scale)
                    entry[key + "_diff"] = mean_ci(diffs * scale)
                out[f"{v}@{L}:bias{b:g}"] = entry
                sm = entry["sink_mass"]
                rc, dr = entry["recall"], entry["recall_diff"]
                cells += [f"{sm[0]:.3f}", f"{rc[0]:.1f}",
                          "--" if b == 0.0 else f"{dr[0]:+.1f}{{\\scriptsize$\\pm${dr[1]:.1f}}}"]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write("paper/generated/table_interventions_synthetic.tex", "\n".join(lines[:-1]) + "\n")
    # Pooled over every trained model, with the NoPE softmax stack reported on its own:
    # it has no other source of order, so its first token can carry position.
    pooled = {}
    removed = min(biases)
    for L in lengths:
        for group, keep in (("with_positional_signal", lambda v: v != "nope"), ("nope_softmax", lambda v: v == "nope")):
            diffs = []
            for v in variants:
                if not keep(v):
                    continue
                base = {r["seed"]: r for r in rows if r["variant"] == v and r["seq_len"] == L and r["bias"] == 0.0}
                cur = {r["seed"]: r for r in rows if r["variant"] == v and r["seq_len"] == L and r["bias"] == removed}
                diffs += [100 * (cur[x]["recall"] - base[x]["recall"]) for x in sorted(set(base) & set(cur))]
            if diffs:
                pooled[f"{group}@{L}"] = {"recall_diff": mean_ci(diffs), "n_models": len(diffs)}
    out["pooled"] = pooled
    SUMMARY["interventions_synthetic"] = out


def hf_interventions(hfi, n_boot=2000):
    """Paired over prompts: each bias level against bias 0 on identical prompts."""
    out = {}
    for name, r in sorted(hfi.items()):
        rows, sinks = r["rows"], r["sinks"]
        biases = sorted({x["bias"] for x in rows}, reverse=True)
        for L in sorted({x["length"] for x in rows}):
            base = {(x["depth_index"], x["trial"]): x for x in rows if x["length"] == L and x["bias"] == 0.0}
            for b in biases:
                cur = {(x["depth_index"], x["trial"]): x for x in rows if x["length"] == L and x["bias"] == b}
                keys = sorted(set(base) & set(cur))
                if not keys:
                    continue
                c = np.array([cur[k]["correct"] for k in keys], float)
                c0 = np.array([base[k]["correct"] for k in keys], float)
                lp = np.array([cur[k]["answer_logprob"] for k in keys], float)
                lp0 = np.array([base[k]["answer_logprob"] for k in keys], float)
                idx = RNG.integers(0, len(keys), size=(n_boot, len(keys)))
                boot = (c[idx] - c0[idx]).mean(1)
                targets = sorted({cur[k]["depth_target"] for k in keys})
                prof = []
                for t in targets:
                    ct = [cur[k]["correct"] for k in keys if cur[k]["depth_target"] == t]
                    lo_t, hi_t = wilson_interval(int(sum(ct)), len(ct))
                    prof.append((t, float(np.mean(ct)), lo_t, hi_t))
                ss = [x for x in sinks if x["length"] == L and x["bias"] == b]
                entry = {"n": len(keys), "recall": float(c.mean()),
                         "recall_ci": wilson_interval(int(c.sum()), len(c)),
                         "recall_diff": [float((c - c0).mean())] + [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                         "logprob": mean_ci(lp), "logprob_diff": mean_ci(lp - lp0), "profile": prof,
                         "sink_mass": mean_ci([x["sink_mass"] for x in ss]) if ss else None}
                if len(keys) >= 20:
                    entry["depth_fit_correct"] = depth_fit([cur[k]["depth_actual"] for k in keys], c)
                    entry["depth_fit_logprob"] = depth_fit([cur[k]["depth_actual"] for k in keys], lp)
                out[f"{name}@{L}:bias{b:g}"] = entry
    SUMMARY["interventions_hf"] = out
    return out


def intervention_figures(syn, hfi_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
    os.makedirs(os.path.join(OUT, "paper/figures"), exist_ok=True)
    cmap = plt.get_cmap("viridis")

    def save(fig, fname):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"paper/figures/{fname}.{ext}"), dpi=200)
        plt.close(fig)

    if syn:
        lengths = sorted({r["seq_len"] for r in syn})
        biases = sorted({r["bias"] for r in syn}, reverse=True)
        variants = [v for v in ["softmax", "gate", "hybrid", "hybrid_nogate", "sinklogit", "attnres"]
                    if any(r["variant"] == v for r in syn)]
        fig, ax = plt.subplots(len(lengths), len(variants), figsize=(6.8, 1.7 * len(lengths) + 0.3),
                               sharey=True, squeeze=False)
        for i, L in enumerate(lengths):
            for j, v in enumerate(variants):
                for k, b in enumerate(biases):
                    rs = [r for r in syn if r["variant"] == v and r["seq_len"] == L and r["bias"] == b]
                    if not rs:
                        continue
                    prof = np.array([[x["acc"] for x in r["profile"]] for r in rs]) * 100
                    centres = [x["centre"] for x in rs[0]["profile"]]
                    ax[i, j].plot(centres, np.nanmean(prof, 0), lw=1.1, color=cmap(k / max(1, len(biases) - 1)),
                                  label=f"bias {bias_label(b).replace('$', '')}")
                ax[i, j].set_title(f"{SHORT.get(v, v)}, {L} tokens")
                if i == len(lengths) - 1:
                    ax[i, j].set_xlabel("depth")
            ax[i, 0].set_ylabel("recall (%)")
        ax[0, -1].legend(frameon=False)
        save(fig, "interventions_synthetic_profiles")

    if hfi_summary:
        names = sorted({k.split("@")[0] for k in hfi_summary})
        lengths = sorted({int(k.split("@")[1].split(":")[0]) for k in hfi_summary})
        fig, ax = plt.subplots(len(names), len(lengths), figsize=(6.8, 1.9 * len(names) + 0.3),
                               sharey=True, squeeze=False)
        for i, name in enumerate(names):
            for j, L in enumerate(lengths):
                keys = [k for k in hfi_summary if k.startswith(f"{name}@{L}:")]
                bs = sorted({float(k.split(":bias")[1]) for k in keys}, reverse=True)
                for n_b, b in enumerate(bs):
                    e = hfi_summary[f"{name}@{L}:bias{b:g}"]
                    prof = e["profile"]
                    sm = e.get("sink_mass")
                    lab = f"bias {bias_label(b).replace('$', '')}" + (f" (sink {sm[0]:.2f})" if sm else "")
                    ax[i, j].plot([x[0] for x in prof], [100 * x[1] for x in prof], marker="o", ms=2,
                                  lw=1.1, color=cmap(n_b / max(1, len(bs) - 1)), label=lab)
                ax[i, j].set_title(f"{name}, {L // 1024}K")
                if i == len(names) - 1:
                    ax[i, j].set_xlabel("depth of the needle")
                if j == len(lengths) - 1:
                    ax[i, j].legend(frameon=False)
            ax[i, 0].set_ylabel("exact match (%)")
        save(fig, "interventions_hf_profiles")


def markdown():
    md = ["# SinkProbe results summary", ""]

    def fmt(t):
        if isinstance(t, (list, tuple)) and len(t) >= 2 and all(isinstance(x, (int, float)) for x in t[:2]):
            return f"{t[0]:.3f} ± {t[1]:.3f}"
        return str(t)
    for section, content in SUMMARY.items():
        md.append(f"## {section}")
        if isinstance(content, dict):
            for k, v in content.items():
                if isinstance(v, dict):
                    md.append(f"- **{k}**: " + "; ".join(f"{kk} {fmt(vv)}" for kk, vv in v.items()
                                                         if kk not in ("profile", "by_length",
                                                                       "profile_logprob",
                                                                       "sink_per_layer",
                                                                       "depth_fit_correct",
                                                                       "depth_fit_logprob")))
                else:
                    md.append(f"- **{k}**: {fmt(v)}")
        md.append("")
    write("results/summary.md", "\n".join(md))


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/main")
    ap.add_argument("--hf", default="results/hf")
    ap.add_argument("--interventions", default="results/interventions")
    ap.add_argument("--min-train-recall", type=float, default=0.9,
                    help="runs below this recall at the training length count as not learned")
    ap.add_argument("--out", default=".", help="base directory for paper/ and results/ outputs")
    a = ap.parse_args()
    OUT = a.out
    global MIN_RECALL
    MIN_RECALL = a.min_train_recall
    runs = load_runs(a.runs)
    global MAIN_WARMUP
    if runs:
        ws = [r["warmup"] for r in runs]
        MAIN_WARMUP = max(set(ws), key=ws.count)
    bench_path = "results/bench/step_times.json"
    bench = json.load(open(bench_path, encoding="utf-8")) if os.path.exists(bench_path) else None
    hf = hf_summary(a.hf)
    hf_compare()
    if runs:
        print(f"{len(runs)} controlled runs, {sum(r['learned'] for r in runs)} learned retrieval")
        table_learnability(runs)
        table_ladder(runs)
        table_steps(runs)
        dose_response(runs)
        skew_effect(runs)
        dissociation(runs)
        noop_enrichment(runs)
        table_mechanisms(runs, bench)
        gate_by_delta(runs)
        warmup_ablation(runs)
    figures(runs, hf) if runs else None
    hf_figures(hf)
    syn_i, hf_i = load_interventions(a.interventions)
    ppl_summary(a.interventions)
    synthetic_interventions(syn_i)
    hfi_summary = hf_interventions(hf_i) if hf_i else None
    intervention_figures(syn_i, hfi_summary)
    write("results/summary.json", json.dumps(SUMMARY, indent=1, default=float))
    markdown()
    print("wrote paper/generated, paper/figures, results/summary.json and results/summary.md")


if __name__ == "__main__":
    main()
