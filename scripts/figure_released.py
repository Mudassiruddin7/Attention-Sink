"""Released checkpoints: sink mass against the probe ladder (method notes, section 24).

    python scripts/figure_released.py

Reads   notebooks/internals_v4.json, notebooks/probes_v4.json,
        notebooks/colab_prodscale_probe_v3.ipynb (executed; its printed output)
Writes  paper/figures/released_sink_vs_probes.pdf (and .png),
        paper/generated/table_released.tex, results/summary_released.json

The v3 numbers are parsed from the executed notebook rather than copied by hand.
From v3 we keep only the probes without a known defect (simple, multikey,
twohop) and the position-0 intervention, whose mask v4 verified. nearkey,
count, paraphrase and update come from v4.
"""

from __future__ import annotations

import itertools
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NB = "notebooks"
V3_NOTEBOOK = f"{NB}/colab_prodscale_probe_v3.ipynb"
ORDER = ["HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
         "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base"]
SHORT = {"HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M", "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
         "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B", "Qwen/Qwen3-0.6B-Base": "Qwen3-0.6B",
         "Qwen/Qwen3-1.7B-Base": "Qwen3-1.7B"}
# Colour carries the family (the confound the text discusses), shape the size.
FAMILY = {"SmolLM2": "#1baf7a", "Qwen2.5": "#2a78d6", "Qwen3": "#eb6834"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def family(name):
    return next(f for f in FAMILY if f in name)


def marker(name):
    return "s" if any(s in name for s in ("1.5B", "1.7B")) else "o"


def parse_v3():
    # Probe lines and the intervention line printed by the v3 run, per model.
    text = []
    for cell in json.load(open(V3_NOTEBOOK, encoding="utf-8"))["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                text.append("".join(out["text"]))
    text = "\n".join(text)
    found, current = {}, None
    for line in text.splitlines():
        head = re.match(r"^(\S+/\S+) \[(\w+)\]: sink", line)
        if head:
            current = head.group(1)
            found[current] = {"dtype": head.group(2)}
            continue
        probe = re.match(r"^\s+(\w+)\s+(\d+): exact\s+([\d.]+)%", line)
        if probe and current:
            found[current][f"{probe.group(1)}@{probe.group(2)}"] = float(probe.group(3))
        removed = re.match(r"^\s+position 0 removed: ([\d.]+)% -> ([\d.]+)%", line)
        if removed and current:
            found[current]["kept"] = float(removed.group(1))
            found[current]["removed"] = float(removed.group(2))
    return found


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r, i = [0.0] * len(v), 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return float("nan") if not va or not vb else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def spearman_exact(x, y):
    rx, ry = ranks(list(x)), ranks(list(y))
    rho = pearson(rx, ry)
    perms = list(itertools.permutations(ry))
    return rho, sum(abs(pearson(rx, list(p))) >= abs(rho) - 1e-12 for p in perms) / len(perms)


def main():
    internals = {r["model"]: r for r in json.load(open(f"{NB}/internals_v4.json", encoding="utf-8"))}
    redone = json.load(open(f"{NB}/probes_v4.json", encoding="utf-8"))["redone_on_v3_models"]
    v3 = parse_v3()

    rows = []
    for name in ORDER:
        s, r, o = internals[name], redone[name], v3[name]
        assert s["mask_verdict"] == "applied", name
        rows.append({
            "model": SHORT[name], "family": family(name),
            "sink": s["sink_mass"], "ratio": s["sink_ratio"], "sink_probe_text": s["sink_mass_task"],
            "hidden_change": 100 * s["mask_moves_output"],
            "simple": o["simple@4096"], "multikey": o["multikey@4096"], "twohop": o["twohop@4096"],
            "nearkey": 100 * r["nearkey"]["exact"],
            "paraphrase_tf": 100 * r["paraphrase"]["exact"], "paraphrase_gen": 100 * r["paraphrase"]["generated"],
            "update_tf": 100 * r["update"]["exact"], "update_gen": 100 * r["update"]["generated"],
            "count_gen": 100 * r["count"]["generated"], "count_numeric": 100 * r["count"]["answer_shaped"],
            "kept": o["kept"], "removed": o["removed"],
            "prompts": len(r["nearkey"]["outcomes"]),
        })

    panels = [("multikey", "Multi-key"), ("nearkey", "Near-duplicate keys"),
              ("paraphrase_gen", "Paraphrased question"), ("twohop", "Two-hop"),
              ("update_gen", "Updated fact")]
    tests = {}
    for key, _ in panels:
        tests[key] = spearman_exact([x["sink"] for x in rows], [x[key] for x in rows])

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.3), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for x in rows:
            ax.scatter(x["sink"], x[key], s=34, marker=marker(x["model"]), color=FAMILY[x["family"]],
                       edgecolors="white", linewidths=1.0, zorder=3)
        rho, p = tests[key]
        ax.set_title(f"{title}\n$\\rho$ = {rho:+.2f}, p = {p:.2f}", fontsize=7, color=INK, pad=3)
        ax.set_ylim(-5, 105)
        ax.set_xlim(0.26, 0.46)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([0.3, 0.35, 0.4, 0.45])
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("sink mass")
    axes[0, 2].tick_params(labelbottom=True)

    # Direct labels once, in the first panel, where the points do not collide.
    first = axes[0, 0]
    for x in rows:
        dy = {"Qwen2.5-0.5B": -12, "Qwen3-0.6B": -12, "Qwen2.5-1.5B": 6, "Qwen3-1.7B": 6}.get(x["model"], 6)
        first.annotate(x["model"], (x["sink"], x["multikey"]), xytext=(0, dy), textcoords="offset points",
                       ha="center", fontsize=5.5, color=INK2)

    legend = axes[1, 2]
    legend.axis("off")
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=6, color=c, markeredgecolor="white",
                          label=f) for f, c in FAMILY.items()]
    handles += [plt.Line2D([], [], linestyle="", marker="o", markersize=6, color=MUTED, label="0.36-0.6B"),
                plt.Line2D([], [], linestyle="", marker="s", markersize=6, color=MUTED, label="1.5-1.7B")]
    legend.legend(handles=handles, loc="center", frameon=False, fontsize=7, labelcolor=INK2,
                  title="family (colour), size (shape)", title_fontsize=6.5)
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/released_sink_vs_probes.{ext}", dpi=300, facecolor="white")

    def cell(v):
        return f"{v:.1f}"

    lines = []
    for x in rows:
        lines.append(" & ".join([
            x["model"], f"{x['sink']:.2f}", f"{x['ratio']:.0f}", cell(x["simple"]), cell(x["multikey"]),
            cell(x["nearkey"]), f"{cell(x['paraphrase_tf'])} / {cell(x['paraphrase_gen'])}", cell(x["twohop"]),
            f"{cell(x['update_tf'])} / {cell(x['update_gen'])}", cell(x["count_gen"]),
            f"{cell(x['kept'])} $\\to$ {cell(x['removed'])}",
        ]) + " \\\\")
    with open("paper/generated/table_released.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open("results/summary_released.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "spearman": {k: {"rho": v[0], "p": v[1]} for k, v in tests.items()}}, f, indent=1)

    for x in rows:
        print(x)
    for k, (rho, p) in tests.items():
        print(f"{k:15s} rho {rho:+.2f} p {p:.3f}")


if __name__ == "__main__":
    main()
