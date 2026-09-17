"""Figures for the ICLR paper, drawn from the saved runs and evaluations.

    python scripts/paper_figures.py            # everything
    python scripts/paper_figures.py --no-maps  # skip the attention maps (they run the models on CPU)

Writes paper/iclr/figures/*.pdf (and .png previews). One colour per design is
used in every figure; the four hues pass the colour-vision checks of the
dataviz palette for all pairs, and attention-only stacks are drawn in grey.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "paper", "iclr", "figures")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
OURS, LLLS, LLLS_BK, SLLL = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
ATTN = "#9a9892"

STYLE = {
    "hybrid_bka_first": dict(color=OURS, marker="o", label="SLLL + bound keys (ours)"),
    "hybrid_bka": dict(color=LLLS_BK, marker="D", label="LLLS + bound keys"),
    "hybrid_nope": dict(color=LLLS, marker="s", label="LLLS (usual hybrid)"),
    "hybrid_nope_first": dict(color=SLLL, marker="v", label="SLLL, no bound keys"),
    "softmax": dict(color=ATTN, marker="^", label="Softmax + RoPE"),
    "bka": dict(color=ATTN, marker="o", label="Attention only + bound keys"),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.axisbelow": True, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "legend.frameon": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=220)
    plt.close(fig)


def rescored():
    table = {}
    for path in ("results/evals/rescore_main.json", "results/evals/rescore_stage5.json",
                 "results/evals/rescore_longer.json"):
        for r in load(path)["rows"]:
            if r.get("warmup", 1500) == 1500:
                table[(r["name"], r["seq_len"])] = r
    return table


def learned_runs(table, variant):
    names = sorted({n for (n, _L) in table if n.split("__")[0] == variant})
    return [n for n in names if table.get((n, 256), {}).get("recall", 0) >= 0.9]


# ---------------------------------------------------------------------------
# Figure 1: the three results in one row
# ---------------------------------------------------------------------------

def panel_sink(ax):
    pts = load("results/summary_sink_recall.json")
    rho, lo, hi = pts["spearman"]
    group = {"Attention only": ("softmax", "bka"), "Global layer last (LLLS)": ("hybrid_nope",),
             "Global layer last, bound keys": ("hybrid_bka",), "Global layer first (SLLL)": ("hybrid_nope_first",),
             "Global layer first, bound keys": ("hybrid_bka_first",)}
    order = ["Attention only", "Global layer last (LLLS)", "Global layer first (SLLL)",
             "Global layer last, bound keys", "Global layer first, bound keys"]
    for g in order:
        ps = [p for p in pts["points"] if p["design"] == g]
        st = STYLE[group[g][0]]
        for rope in (True, False):
            sub = [p for p in ps if p["rope"] == rope]
            if not sub:
                continue
            color = st["color"]
            marker = "^" if rope else ("o" if g != "Attention only" else "o")
            if g == "Attention only":
                color = ATTN
            else:
                marker = "^" if rope else st["marker"]
            ax.scatter([p["sink_mass"] for p in sub], [p["recall@2048"] for p in sub], s=22 if g.endswith("first, bound keys") else 16,
                       color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel("sink mass at the training length")
    ax.set_ylabel("recall at 8$\\times$ train length (%)")
    ax.set_xlim(0, 0.175)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.text(0.172, 58, f"Spearman $\\rho$ = {rho:.2f}\n95% CI [{lo:.2f}, {hi:.2f}]\n{len(pts['points'])} learned runs",
            ha="right", va="center", fontsize=6, color=INK2)
    ax.annotate("smallest sink,\n17% recall", xy=(0.0086, 17.1), xytext=(0.03, 32), fontsize=5.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_title("(a) Sink size vs. recall at 8$\\times$", loc="left")


def panel_discovery(ax):
    disc = load("results/summary_discovery.json")
    groups = [
        ("Attention only", ATTN, "-", "Attention only"),
        ("Global layer last (LLLS)", LLLS, "-", "LLLS"),
        ("Global layer last, bound keys", LLLS_BK, "-", "LLLS + bound keys"),
        ("Global layer first (SLLL)", SLLL, "-", "SLLL, no bound keys"),
        ("Global layer first, bound keys", OURS, "-", "ours"),
        ("Global layer first, bound keys, no warm-up", OURS, (0, (2.2, 1.4)), "ours, no warm-up"),
    ]
    steps_grid = np.arange(0, 4001, 50)
    for key, color, ls, label in groups:
        vals = list(disc[key].values())
        n = len(vals)
        frac = [100 * sum(1 for v in vals if v is not None and v <= s) / n for s in steps_grid]
        ax.step(steps_grid, frac, where="post", color=color, ls=ls, lw=1.6 if "ours" in label else 1.3,
                label=f"{label} ({n})", zorder=3 if "ours" in label else 2)
    ax.axvline(1500, color=AXIS, lw=0.7, zorder=1)
    ax.text(1580, 58, "warm-up ends", fontsize=5.8, color=MUTED, ha="left", va="center", rotation=90)
    ax.set_xlabel("training step")
    ax.set_ylabel("runs past 90% training recall (%)")
    ax.set_xlim(0, 4000)
    ax.set_ylim(-4, 104)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("(b) When retrieval is learned", loc="left")


def length_curve(table, variant, metric="recall"):
    runs = learned_runs(table, variant)
    xs, ys = [], []
    for L in (256, 1024, 2048, 4096, 8192, 16384):
        vals = [100 * table[(n, L)][metric] for n in runs if (n, L) in table]
        if vals:
            xs.append(L // 256)
            ys.append(float(np.mean(vals)))
    return xs, ys, len(runs)


def panel_length(ax, table):
    for variant in ("softmax", "bka", "hybrid_nope_first", "hybrid_nope", "hybrid_bka", "hybrid_bka_first"):
        st = STYLE[variant]
        xs, ys, n = length_curve(table, variant)
        ours = variant == "hybrid_bka_first"
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], ms=4.2 if ours else 3.4, mec="white", mew=0.6,
                lw=1.8 if ours else 1.3, label=f"{st['label']} ({n})", zorder=4 if ours else 3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1$\\times$", "4$\\times$", "8$\\times$", "16$\\times$", "32$\\times$", "64$\\times$"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 75)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("evaluation length / training length")
    ax.set_ylabel("recall, exact match (%)")
    ax.set_title("(c) Recall at longer contexts", loc="left")


def figure_teaser(table):
    small = {"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
             "ytick.labelsize": 6, "legend.fontsize": 6}
    with plt.rc_context(small):
        _teaser(table)


def _teaser(table):
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.85), gridspec_kw=dict(wspace=0.42))
    panel_sink(axes[0])
    panel_discovery(axes[1])
    panel_length(axes[2], table)
    handles = [
        Line2D([], [], color=OURS, marker="o", mec="white", lw=1.8, ms=4.5, label="SLLL + bound keys (ours)"),
        Line2D([], [], color=LLLS_BK, marker="D", mec="white", lw=1.3, ms=4, label="LLLS + bound keys"),
        Line2D([], [], color=LLLS, marker="s", mec="white", lw=1.3, ms=4, label="LLLS, the usual hybrid"),
        Line2D([], [], color=SLLL, marker="v", mec="white", lw=1.3, ms=4, label="SLLL, no bound keys"),
        Line2D([], [], color=ATTN, marker="o", mec="white", lw=1.3, ms=4, label="attention only"),
        Line2D([], [], color=INK2, marker="^", lw=0, ms=4, label="triangles: RoPE"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, handlelength=1.8,
               columnspacing=1.6)
    save(fig, "teaser")


# ---------------------------------------------------------------------------
# Depth profiles and attention on the value
# ---------------------------------------------------------------------------

def figure_profiles(table):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95), sharey=True, gridspec_kw=dict(wspace=0.08))
    centres = [(j + 0.5) / 10 for j in range(10)]
    for ax, L in zip(axes, (4096, 16384)):
        for variant in ("softmax", "hybrid_nope_first", "hybrid_nope", "hybrid_bka", "hybrid_bka_first"):
            runs = [n for n in learned_runs(table, variant) if (n, L) in table]
            if not runs:
                continue
            arr = 100 * np.array([[b["acc"] for b in table[(n, L)]["profile"]] for n in runs], dtype=float)
            st = STYLE[variant]
            ours = variant == "hybrid_bka_first"
            if len(runs) > 1:
                ax.fill_between(centres, arr.min(0), arr.max(0), color=st["color"], alpha=0.10, lw=0)
            ax.plot(centres, arr.mean(0), color=st["color"], marker=st["marker"], ms=3.4 if not ours else 3.8,
                    mec="white", mew=0.5, lw=1.7 if ours else 1.2, label=f"{st['label']} ({len(runs)})",
                    zorder=4 if ours else 3)
        ax.set_title(f"{L:,} tokens ({L // 256}$\\times$ the training length)")
        ax.set_xlabel("depth of the queried pair in the context")
        ax.set_xlim(0, 1)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("recall (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, handlelength=1.8,
               columnspacing=1.2)
    save(fig, "profiles")


def figure_attention_depth():
    std = load("results/evals/attention_standard.json")["rows"]
    centres = [(j + 0.5) / 10 for j in range(10)]

    def best(row):
        li = max(range(len(row["layers"])),
                 key=lambda j: row["layers"][j]["value_mass"][row["layers"][j]["best_head"]])
        return li, row["layers"][li]["best_head"]

    first = {r["name"]: best(r) for r in std if r["seq_len"] == 256}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, L in zip(axes, (256, 4096)):
        for variant in ("hybrid_nope_first", "hybrid_nope", "hybrid_bka", "hybrid_bka_first"):
            rows = [r for r in std if r["variant"] == variant and r["seq_len"] == L and r["name"] in first
                    and r["learned"]]
            if variant == "hybrid_nope_first":
                rows = [r for r in std if r["variant"] == variant and r["seq_len"] == L and r["name"] in first]
            if not rows:
                continue
            arr = np.array([r["layers"][first[r["name"]][0]]["value_mass_by_depth"][first[r["name"]][1]] for r in rows])
            st = STYLE[variant]
            ours = variant == "hybrid_bka_first"
            if len(arr) > 1:
                ax.fill_between(centres, arr.min(0), arr.max(0), color=st["color"], alpha=0.10, lw=0)
            ax.plot(centres, arr.mean(0), color=st["color"], marker=st["marker"], ms=3.4, mec="white", mew=0.5,
                    lw=1.7 if ours else 1.2, label=f"{st['label']} ({len(arr)})", zorder=4 if ours else 3)
        ax.set_title(f"{L:,} tokens")
        ax.set_xlabel("depth of the queried pair in the context")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("attention on the value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, handlelength=1.8,
               columnspacing=1.0)
    save(fig, "attention_depth")


# ---------------------------------------------------------------------------
# Images for the architecture figure: a real sequence and real attention rows
# ---------------------------------------------------------------------------

def load_model(path):
    import torch
    from sinkprobe.data import TaskConfig
    from sinkprobe.model import ModelConfig, TinyLM
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyLM(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tc = TaskConfig(**{k: v for k, v in ck["task_config"].items() if k != "vocab_size"})
    return model, tc


def token_kinds(seq, types, answer_pos, keys):
    """0 filler, 1 copied filler, 2 key, 3 value, 4 marker, 5 query key, 6 answer slot.

    keys are all key tokens of the sequence; each occurs once in the haystack,
    directly before its value.
    """
    from sinkprobe.data import COPY
    t = len(seq)
    kinds = np.zeros(t, dtype=int)
    kinds[1:][types[:-1] == COPY] = 1
    body_end = int(answer_pos[0]) - 1
    keyset = set(int(k) for k in keys)
    for i in range(1, body_end):
        if int(seq[i]) in keyset:
            kinds[i], kinds[i + 1] = 2, 3
    kinds[0] = 4
    for p in answer_pos:
        kinds[p - 1], kinds[p], kinds[p + 1] = 4, 5, 6
    return kinds


def figure_sequence_strip(seed=104, shown=48, cols=16, n_queries_shown=4):
    """TikZ tiles for a real training sequence: its first tokens and first queries.

    Written as paper/iclr/figures/arch_tokens.tex so the tiles stay vector and
    the architecture figure shows an actual sequence from the generator. The
    keys of a sequence are the generator's first random draw, so replaying that
    draw recovers every pair, not only the queried ones.
    """
    from sinkprobe.data import HaystackTask, TaskConfig
    cfg = TaskConfig()
    task = HaystackTask(cfg)
    keys = np.random.default_rng(seed).choice(cfg.key_range, size=cfg.n_pairs, replace=False) + cfg.key_base
    d = task.build(1, 256, np.random.default_rng(seed), skew=0.0)
    seq, types, ans = d["tokens"][0].numpy(), d["types"][0].numpy(), d["answer_pos"][0].numpy()
    kinds = token_kinds(seq, types, ans, keys)
    style = {0: "tokFill", 1: "tokCopy", 2: "tokKey", 3: "tokVal", 4: "tokMark", 5: "tokQuery", 6: "tokAns"}
    dx, dy = 0.19, 0.2
    lines = [f"% Generated by scripts/paper_figures.py: HaystackTask(TaskConfig()), seed {seed}.",
             f"% First {shown} tokens of the haystack, then the first {n_queries_shown} queries."]
    for i in range(shown):
        r, c = divmod(i, cols)
        lines.append(rf"\node[{style[kinds[i]]}] (tok{i}) at ({c * dx:.2f},{-r * dy:.2f}) {{}};")
    qrow = -(shown // cols + 0.8) * dy
    first_key = int(seq[ans[0]])
    for i in range(3 * n_queries_shown):
        pos = int(ans[0]) - 1 + i
        lines.append(rf"\node[{style[kinds[pos]]}] (qtok{i}) at ({i * dx:.2f},{qrow:.2f}) {{}};")
    hit = [i for i in range(shown) if int(seq[i]) == first_key]
    if hit:
        lines.append(rf"\coordinate (hitkey) at (tok{hit[0]}.center);")
        lines.append(rf"\coordinate (hitval) at (tok{hit[0] + 1}.center);")
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "arch_tokens.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    counts = {name: int((kinds[:shown] == k).sum()) for k, name in ((0, "filler"), (1, "copy"), (2, "key"), (3, "value"))}
    print("sequence strip", counts, "first query's pair shown at", hit)


def attention_rows(layer, x, rows):
    import torch
    q, k, _ = layer.project(x)
    b, h, t, dh = q.shape
    qr = q.gather(2, rows[:, None, :, None].expand(b, h, rows.shape[1], dh))
    scores = (qr.float() @ k.float().transpose(-1, -2)) / math.sqrt(dh)
    visible = torch.arange(t).view(1, 1, 1, t) <= rows.view(b, 1, -1, 1)
    return torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)


def capture_rows(model, x, ans, layer_index):
    import torch
    store = {}
    mix = model.blocks[layer_index].mix
    hook = mix.register_forward_pre_hook(lambda m, args: store.__setitem__("x", args[0]))
    with torch.no_grad():
        logits, _ = model(x)
        p = attention_rows(mix, store["x"], ans)
    hook.remove()
    return p[0].numpy(), logits


def figure_attention_maps():
    """Real attention rows around the gold value, with and without bound keys.

    Each image row is one query of the query block; columns are positions
    relative to the value of the queried pair (0) and its key (-1). The head is
    the one with the most attention on the value (or, for the second global
    layer of the model without bound keys, on the key) over the same rows.
    """
    import torch
    from sinkprobe.data import HaystackTask
    specs = [
        ("hybrid_bka_first__p0.5__g0__s0", 0, 256, "value", "attn_ours_256"),
        ("hybrid_bka_first__p0.5__g0__s0", 0, 4096, "value", "attn_ours_4096"),
        ("hybrid_nope_first__p0.5__g0__s2", 0, 256, "value", "attn_nobk_l0_256"),
        ("hybrid_nope_first__p0.5__g0__s2", 4, 256, "key", "attn_nobk_l4_256"),
    ]
    half, n_seqs = 4, 2
    info = {}
    for name, layer_index, L, target, tag in specs:
        model, tc = load_model(os.path.join(ROOT, "results", "runs", "main", f"{name}.pt"))
        task = HaystackTask(tc)
        rng = np.random.default_rng(5)
        windows, on_val, on_key, correct = [], [], [], []
        for _ in range(n_seqs):
            d = task.build(1, L, rng, skew=0.0)
            x, ans = d["tokens"], d["answer_pos"]
            p, logits = capture_rows(model, x, ans, layer_index)          # (H, Q, T)
            seq = x[0].numpy()
            body_end = int(ans[0, 0]) - 1
            vpos = np.array([int(np.nonzero(seq[1:body_end] == k)[0][0]) + 2 for k in seq[ans[0].numpy()]])
            idx = vpos[:, None] + np.arange(-half, half + 1)[None, :]
            idx = np.clip(idx, 0, p.shape[-1] - 1)
            windows.append(np.stack([p[:, q, idx[q]] for q in range(len(vpos))], axis=1))  # (H, Q, W)
            on_val.append(p[:, np.arange(len(vpos)), vpos])
            on_key.append(p[:, np.arange(len(vpos)), vpos - 1])
            pred = logits[0].gather(0, ans[0][:, None].expand(-1, logits.size(-1))).argmax(-1)
            correct.append((pred == x[0, ans[0] + 1]).float().numpy())
        win = np.concatenate(windows, axis=1)
        on_val, on_key = np.concatenate(on_val, axis=1), np.concatenate(on_key, axis=1)
        score = on_val if target == "value" else on_key
        head = int(score.mean(1).argmax())
        fig, ax = plt.subplots(figsize=(0.95, 1.05))
        ax.imshow(win[head], aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(AXIS)
            sp.set_linewidth(0.5)
        save(fig, tag)
        info[tag] = {"run": name, "layer": layer_index, "head": head, "length": L, "queries": int(win.shape[1]),
                     "mean_on_value": float(on_val[head].mean()), "mean_on_key": float(on_key[head].mean()),
                     "best_value_head_mass": float(on_val.mean(1).max()),
                     "recall_these_queries": float(np.concatenate(correct).mean())}
        print(tag, info[tag])
    with open(os.path.join(FIG, "attention_maps.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    table = rescored()
    figure_teaser(table)
    figure_profiles(table)
    figure_attention_depth()
    figure_sequence_strip()
    if not a.no_maps:
        figure_attention_maps()


if __name__ == "__main__":
    main()
