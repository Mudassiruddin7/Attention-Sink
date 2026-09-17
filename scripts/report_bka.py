"""Table and figures for the Bound-Key Attention and layer-placement study.

    python scripts/report_bka.py

Reads   results/runs/main/*.json
        results/evals/<tag>.json for the tags in EVAL_TAGS
Writes  paper/generated/table_bka.tex
        paper/figures/bka_profiles.{pdf,png}, bka_reliability.{pdf,png},
        discovery.{pdf,png}, sink_vs_long_recall.{pdf,png}
        results/summary_bka.json, results/summary_discovery.json, results/summary_sink_recall.json

A run counts as learned when recall at the training length is at least 90%.
Recall and depth-profile statistics use learned runs with the default warm-up;
reliability counts every trained seed, with and without the warm-up. The
discovery figure shows, for each group, the share of seeds whose training
recall has reached 90% by each step, counting only runs learned at the end. The tests are in scripts/stats_placement.py.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import make_report as mr   # noqa: E402

RUNS = "results/runs/main"
EVALS = "results/evals"
LENGTHS = [256, 1024, 2048, 4096]
ROWS = [  # label, variant, source of the non-training lengths
    ("Softmax, RoPE", "softmax", "runs"),
    ("Softmax + output gate", "gate", "runs"),
    ("Softmax, NoPE", "nope", "runs"),
    ("NoPE + length scaling", "nope", "nope_logn"),
    ("BKA with RoPE", "bka_rope", "runs"),
    ("BKA without length scaling", "bka", "bka_nologn"),
    ("BKA", "bka", "runs"),
    ("Hybrid 3:1, NoPE", "hybrid_nope", "runs"),
    ("Hybrid 3:1, NoPE + length scaling", "hybrid_nope", "llls_nope_logn"),
    ("Hybrid 3:1, NoPE, AttnRes", "hybrid_attnres", "runs"),
    ("Hybrid 3:1, NoPE, bound keys", "hybrid_bka", "runs"),
    ("Hybrid with bound keys, no length scaling", "hybrid_bka", "hybrid_bka_nologn"),
    ("Hybrid, NoPE layer first", "hybrid_nope_first", "runs"),
    ("NoPE layer first, no length scaling", "hybrid_nope_first", "hybrid_nope_first_nologn"),
    ("Hybrid, bound-key layer first", "hybrid_bka_first", "runs"),
    ("Bound-key layer first, no length scaling", "hybrid_bka_first", "hybrid_bka_first_nologn"),
]
EVAL_TAGS = ("bka_nologn", "nope_logn", "long_baselines", "hybrid_bka_4096", "hybrid_bka_nologn",
             "llls_nope_logn", "hybrid_nope_first_nologn", "hybrid_bka_first_nologn", "rescore_main")
# Chart ink, and one colour per design that is the same in every figure. The hues
# are Okabe-Ito, in an order that passes the colour-vision checks of the dataviz
# palette validator; orange is below 3:1 contrast, so the table carries its values.
INK, INK_MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COLORS = {"softmax": "#0072B2", "hybrid_nope": "#D55E00", "hybrid_bka": "#E69F00",
          "hybrid_nope_first": "#6A3D9A", "hybrid_bka_first": "#009E73"}
PROFILE_VARIANTS = ["softmax", "hybrid_nope", "hybrid_bka", "hybrid_nope_first", "hybrid_bka_first"]
GROUPS = [  # discovery figure: label, which runs, warm-up steps, colour, line style
    ("Attention only", lambda r: r["layout"] == "S", 1500, COLORS["softmax"], "-"),
    ("Global layer last (LLLS)", lambda r: r["layout"] == "LLLS" and r["delta"] and not r["kv_conv"],
     1500, COLORS["hybrid_nope"], "-"),
    ("Global layer last, bound keys", lambda r: r["variant"] == "hybrid_bka", 1500, COLORS["hybrid_bka"], "-"),
    ("Global layer first (SLLL)", lambda r: r["variant"] == "hybrid_nope_first", 1500,
     COLORS["hybrid_nope_first"], "-"),
    ("Global layer first, bound keys", lambda r: r["variant"] == "hybrid_bka_first", 1500,
     COLORS["hybrid_bka_first"], "-"),
    ("Global layer first, bound keys, no warm-up", lambda r: r["variant"] == "hybrid_bka_first", 0,
     COLORS["hybrid_bka_first"], "--"),
]


def load_runs():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        name = os.path.basename(p)[:-5]
        evals = {e["seq_len"]: e for e in r["evals"]}
        for e in evals.values():
            mr.add_depth_fit(e)
        tl = r["run"]["train_len"]
        mc = r["model_config"]
        runs[name] = {"name": name, "variant": r["run"]["variant"], "seed": r["run"]["seed"],
                      "layout": mc.get("layout", "S"), "delta": mc.get("delta", True),
                      "kv_conv": mc.get("kv_conv", 0), "rope": mc.get("rope", True),
                      "warmup": r["run"].get("copy_warmup", 0), "p_noop": r["task_config"].get("p_noop"),
                      "skew": r["task_config"].get("query_skew", 0.0), "evals": evals, "curve": r["curve"],
                      "learned": evals[tl]["recall"] >= 0.9}
    return runs


def load_eval(tag):
    path = os.path.join(EVALS, f"{tag}.json")
    if not os.path.exists(path):
        return {}
    out = {}
    for row in json.load(open(path, encoding="utf-8"))["rows"]:
        mr.add_depth_fit(row)
        out[(row["name"], row["seq_len"])] = row
    return out


def steps_to(curve, level=0.9):
    for c in curve:
        if c["train_recall"] >= level:
            return c["step"]
    return None


def main():
    runs = load_runs()
    evals = {t: load_eval(t) for t in EVAL_TAGS}
    base = [r for r in runs.values()
            if r["p_noop"] is not None and abs(r["p_noop"] - 0.5) < 1e-9 and abs(r["skew"]) < 1e-9]
    summary, lines = {}, []
    for label, variant, source in ROWS:
        trained = [r for r in base if r["variant"] == variant and r["warmup"] == 1500]
        cold = [r for r in base if r["variant"] == variant and r["warmup"] == 0]
        learned = [r for r in trained if r["learned"]]
        if not trained:
            continue

        def metric(r, L, key):
            if L == 256:
                return r["evals"][256].get(key)
            if source != "runs":
                e = evals[source].get((r["name"], L))
                return None if e is None else e.get(key)
            # The rescoring pass comes first: it has every answer metric and one evaluation
            # protocol for every model (method notes, section 20).
            for e in (evals["rescore_main"].get((r["name"], L)), r["evals"].get(L),
                      evals["long_baselines"].get((r["name"], L)),
                      evals["hybrid_bka_4096"].get((r["name"], L))):
                if e is not None and e.get(key) is not None:
                    return e[key]
            return None

        entry = {"source": source, "learned": len(learned), "trained": len(trained),
                 "learned_no_warmup": sum(r["learned"] for r in cold), "trained_no_warmup": len(cold),
                 "steps_to_90": [steps_to(r["curve"]) for r in learned]}
        cells = [label, f"{len(learned)}/{len(trained)}",
                 f"{entry['learned_no_warmup']}/{len(cold)}" if cold else "--"]
        for L in LENGTHS:
            vals = [100 * v for r in learned if (v := metric(r, L, "recall")) is not None]
            m, h, n = mr.mean_ci(vals)
            entry[f"recall@{L}"] = (m, h, n)
            cells.append("--" if n == 0 else (f"{m:.1f}" + (f"{{\\scriptsize$\\pm${h:.1f}}}" if np.isfinite(h) else "")))
        # Every query but the first: past the training length the first query can fail
        # on where the query block starts rather than on retrieval (method notes, section 13).
        vals = [100 * v for r in learned if (v := metric(r, 4096, "recall_after_first")) is not None]
        m, h, n = mr.mean_ci(vals)
        entry["recall_after_first@4096"] = (m, h, n)
        cells.append("--" if n == 0 else f"{m:.1f}")
        for key in ("recall_no_marker", "marker_rate"):
            vals = [100 * v for r in learned if (v := metric(r, 4096, key)) is not None]
            m, h, n = mr.mean_ci(vals)
            entry[f"{key}@4096"] = (m, h, n)
            cells.append("--" if n == 0 else f"{m:.1f}")
        for key in ("fit_slope", "fit_curv"):
            vals = [100 * v for r in learned if (v := metric(r, 4096, key)) is not None]
            m, h, n = mr.mean_ci(vals)
            entry[f"{key}@4096"] = (m, h, n)
            cells.append("--" if n == 0 else f"{m:+.0f}")
        sink = mr.mean_ci([r["evals"][256]["sink_mass"] for r in learned if "sink_mass" in r["evals"][256]])
        entry["sink_mass@256"] = sink
        cells.append("--" if sink[2] == 0 else f"{sink[0]:.3f}")
        lines.append(" & ".join(cells) + " \\\\")
        summary[label] = entry

    mr.write("paper/generated/table_bka.tex", "\n".join(lines) + "\n")
    with open("results/summary_bka.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)

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
    labels = {variant: label for label, variant, source in ROWS if source == "runs"}

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for j, L in enumerate((2048, 4096)):
        for variant in PROFILE_VARIANTS:
            learned = [r for r in base if r["variant"] == variant and r["warmup"] == 1500 and r["learned"]]
            profs = []
            for r in learned:
                e = r["evals"].get(L) or evals["long_baselines"].get((r["name"], L))
                if e is not None:
                    profs.append([b["acc"] for b in e["profile"]])
            if not profs:
                continue
            arr = 100 * np.array(profs)
            centres = [(i + 0.5) / arr.shape[1] for i in range(arr.shape[1])]
            m = arr.mean(0)
            if len(arr) > 1:
                # Range across seeds; with two to four seeds a t interval is wider than the axis.
                ax[j].fill_between(centres, arr.min(0), arr.max(0), color=COLORS[variant], alpha=0.12, lw=0)
            ax[j].plot(centres, m, marker="o", ms=3.5, mec="white", mew=0.8, lw=1.5, color=COLORS[variant],
                       label=f"{labels[variant]} (n={len(arr)})")
        ax[j].set_title(f"{L} tokens ({L // 256}x the training length)")
        ax[j].set_xlabel("depth of the queried pair")
        ax[j].set_ylim(-3, 103)
        ax[j].set_yticks([0, 25, 50, 75, 100])
    ax[0].set_ylabel("recall (%)")
    ax[1].legend(loc="center right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/bka_profiles.{ext}", dpi=200)
    plt.close(fig)

    # Horizontal bars keep the design names readable; each bar ends in its count,
    # since a share of two to four seeds means little without it.
    rows = [(label, entry) for label, entry in summary.items() if entry["source"] == "runs"]
    fig, ax = plt.subplots(figsize=(6.8, 0.27 * len(rows) + 0.6))
    ys = np.arange(len(rows))[::-1]
    for y, (label, entry) in zip(ys, rows):
        for offset, learned, trained, color in ((0.2, entry["learned"], entry["trained"], "#444444"),
                                                (-0.2, entry["learned_no_warmup"], entry["trained_no_warmup"],
                                                 "#AAAAAA")):
            if not trained:
                continue
            share = 100 * learned / trained
            ax.barh(y + offset, share, 0.36, color=color)
            if share == 0:      # a zero-length bar still shows which condition its count belongs to
                ax.plot([0, 0], [y + offset - 0.18, y + offset + 0.18], color=color, lw=2.5,
                        solid_capstyle="butt")
            ax.text(share + 1.5, y + offset, f"{learned}/{trained}", va="center", fontsize=6, color=INK_MUTED)
    ax.set_yticks(ys, [label for label, _ in rows], fontsize=6.5)
    ax.set_xlim(-1.5, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("seeds that learned (%)")
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    keys = [plt.Rectangle((0, 0), 1, 1, color="#444444"), plt.Rectangle((0, 0), 1, 1, color="#AAAAAA")]
    ax.legend(keys, ["with copy-rich warm-up", "without warm-up"], loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/bka_reliability.{ext}", dpi=200)
    plt.close(fig)

    steps = np.arange(0, 4001, 100)
    discovery = {}
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    for label, keep, warm, color, style in GROUPS:
        group = [r for r in base if keep(r) and r["warmup"] == warm]
        if not group:
            continue
        reached = [steps_to(r["curve"]) if r["learned"] else None for r in group]
        discovery[label] = dict(zip((r["name"] for r in group), reached))
        share = [100 * np.mean([s is not None and s <= x for s in reached]) for x in steps]
        ax.step(steps, share, where="post", color=color, ls=style, lw=1.5, label=f"{label} (n={len(group)})")
    ax.axvline(1500, color=AXIS, lw=0.8, zorder=0)
    ax.text(1500, 1.02, "warm-up ends", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6, color=INK_MUTED)
    ax.set_xlim(0, 4000)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("training step")
    ax.set_ylabel("seeds past 90% recall (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/discovery.{ext}", dpi=200)
    plt.close(fig)
    with open("results/summary_discovery.json", "w", encoding="utf-8") as f:
        json.dump(discovery, f, indent=1)

    # Sink size against recall far past the training length, one point per learned run
    # with the default warm-up. 2048 tokens is the longest length all of them were evaluated at.
    from matplotlib.lines import Line2D
    far = 2048
    families = [("Attention only", lambda r: r["layout"] == "S", COLORS["softmax"]),
                ("Global layer last (LLLS)", lambda r: r["layout"] == "LLLS" and r["delta"] and not r["kv_conv"],
                 COLORS["hybrid_nope"]),
                ("Global layer last, bound keys", lambda r: r["variant"] == "hybrid_bka", COLORS["hybrid_bka"]),
                ("Global layer first (SLLL)", lambda r: r["variant"] == "hybrid_nope_first",
                 COLORS["hybrid_nope_first"]),
                ("Global layer first, bound keys", lambda r: r["variant"] == "hybrid_bka_first",
                 COLORS["hybrid_bka_first"])]
    points = []
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    for label, keep, color in families:
        group = [r for r in base if keep(r) and r["warmup"] == 1500 and r["learned"]
                 and far in r["evals"] and "sink_mass" in r["evals"][256]]
        for rope, marker in ((False, "o"), (True, "^")):
            sub = [r for r in group if r["rope"] == rope]
            if not sub:
                continue
            xs = [r["evals"][256]["sink_mass"] for r in sub]
            ys = [100 * r["evals"][far]["recall"] for r in sub]
            new = label == "Global layer first, bound keys"      # the proposed design, drawn larger
            ax.scatter(xs, ys, s=64 if new else 36, marker=marker, color=color, edgecolors="white",
                       linewidths=0.8, zorder=4 if new else 3)
            points += [{"run": r["name"], "design": label, "rope": rope, "sink_mass": x, f"recall@{far}": y}
                       for r, x, y in zip(sub, xs, ys)]
    rho = mr.spearman_boot(np.array([p["sink_mass"] for p in points]), np.array([p[f"recall@{far}"] for p in points]))
    for p in points:
        if p["run"].startswith("hybrid_nogate__"):
            ax.annotate("gate-free hybrid, RoPE", (p["sink_mass"], p[f"recall@{far}"]), xytext=(-12, 18),
                        textcoords="offset points", ha="right", fontsize=6, color=INK_MUTED,
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    handles = [Line2D([], [], ls="", marker="o", ms=6.5 if l == "Global layer first, bound keys" else 5, color=c,
                      mec="white", label=l) for l, _, c in families if any(p["design"] == l for p in points)]
    handles += [Line2D([], [], ls="", marker="o", ms=5, color="#888888", mec="white",
                       label="global layers without positions"),
                Line2D([], [], ls="", marker="^", ms=5, color="#888888", mec="white",
                       label="global layers with RoPE")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xlim(left=0)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("sink mass at the training length (256 tokens)")
    ax.set_ylabel(f"recall at {far} tokens (%)")
    if rho[1] <= 0 <= rho[2]:
        verdict = "No reliable association"
    else:
        verdict = "Positive association" if rho[0] > 0 else "Negative association"
    ax.set_title(f"{verdict}: Spearman rho {rho[0]:.2f} [{rho[1]:.2f}, {rho[2]:.2f}], {len(points)} runs")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/sink_vs_long_recall.{ext}", dpi=200)
    plt.close(fig)
    with open("results/summary_sink_recall.json", "w", encoding="utf-8") as f:
        json.dump({"spearman": rho, "points": points}, f, indent=1, default=float)

    for label, e in summary.items():
        rec = " ".join(f"{L}:{e[f'recall@{L}'][0]:5.1f}" for L in LENGTHS)
        print(f"{label:28s} learned {e['learned']}/{e['trained']} no-warm-up {e['learned_no_warmup']}/{e['trained_no_warmup']} "
              f"| {rec} | slope@4096 {e['fit_slope@4096'][0]:+.0f} curv {e['fit_curv@4096'][0]:+.0f} | sink {e['sink_mass@256'][0]:.3f}")
    for label, d in discovery.items():
        reached = sorted(s for s in d.values() if s is not None)
        print(f"discovery {label}: {reached} + {sum(s is None for s in d.values())} never")


if __name__ == "__main__":
    main()
