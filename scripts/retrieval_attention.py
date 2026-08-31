"""Where the global layers look when they answer a query, by depth and length.

    python scripts/retrieval_attention.py --glob "results/runs/main/hybrid_bka_first__p0.5__g0__s?.pt" \
        --lengths 256 4096 --tag standard --plot

For each softmax layer of a saved model, the attention rows of the answer
positions (the queried key in the query block, which must predict its value)
are recomputed exactly from the layer's own queries and keys and read at:

    value    the position of the value that follows the queried key in the
             haystack, where a retrieval head has to look
    key      the position of the queried key itself
    sink     position 0
    recent   the last 64 haystack positions

Only the answer rows are computed, so the T x T map is never stored. Mass on
the value position is reported for every head, overall and in ten depth bins.
The plot follows, in each run, the head that reads the value best at the first
length, so the longer lengths show whether that same head still finds the
value at every depth. --set-task reads the same models on another data regime,
for example p_noop=0.0, the copy-rich data of the warm-up.

Writes results/evals/attention_<tag>.json and, with --plot,
paper/figures/attention_<tag>.{pdf,png}.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.data import HaystackTask, TaskConfig      # noqa: E402
from sinkprobe.layers import LOGN, SoftmaxAttention      # noqa: E402
from sinkprobe.model import ModelConfig, TinyLM          # noqa: E402
from sinkprobe.train import parse_kv                     # noqa: E402

N_BINS = 10
RECENT = 64
MEASURES = ("value", "key", "sink", "recent", "top1_value")
# The same colour per design as scripts/report_bka.py.
COLORS = {"softmax": "#0072B2", "hybrid_nope": "#D55E00", "hybrid_bka": "#E69F00",
          "hybrid_nope_first": "#6A3D9A", "hybrid_bka_first": "#009E73"}
LABELS = {"softmax": "Softmax, RoPE", "hybrid_nope": "Global layer last, NoPE",
          "hybrid_bka": "Global layer last, bound keys", "hybrid_nope_first": "Global layer first, NoPE",
          "hybrid_bka_first": "Global layer first, bound keys"}


def gold_positions(tokens: torch.Tensor, answer_pos: torch.Tensor, body_end: int):
    """Haystack positions of each queried key and of its value, both (B, Q).

    Keys never occur in the filler or as values, so the only occurrence of a
    queried key before the query block is its pair.
    """
    keys = tokens.gather(1, answer_pos)
    hit = tokens[:, None, 1:body_end] == keys[:, :, None]
    if not bool(hit.any(-1).all()):
        raise ValueError("a queried key does not occur in the haystack")
    key_pos = hit.float().argmax(-1) + 1
    return key_pos, key_pos + 1


def attention_rows(layer: SoftmaxAttention, x: torch.Tensor, rows: torch.Tensor):
    """Attention of the query positions rows (B, Q) over every position, as (B, H, Q, T)."""
    if layer.sink is not None or layer.norm != "softmax":
        raise ValueError("only plain softmax layers are supported")
    q, k, _ = layer.project(x)
    b, h, t, dh = q.shape
    qr = q.gather(2, rows[:, None, :, None].expand(b, h, rows.shape[1], dh))
    scores = (qr.float() @ k.float().transpose(-1, -2)) / math.sqrt(dh)
    visible = torch.arange(t, device=x.device).view(1, 1, 1, t) <= rows.view(b, 1, -1, 1)
    return torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)


@torch.no_grad()
def read_model(model, task, length, n_seqs, token_budget, rng, dev):
    layers = [(i, blk.mix) for i, blk in enumerate(model.blocks) if isinstance(blk.mix, SoftmaxAttention)]
    inputs = {}
    hooks = [mix.register_forward_pre_hook(lambda m, args, i=i: inputs.__setitem__(i, args[0]))
             for i, mix in layers]
    nq, heads = task.cfg.n_queries, layers[0][1].h
    body_end = length - 3 * nq
    sums = {m: np.zeros((len(layers), heads)) for m in MEASURES}
    by_depth = np.zeros((len(layers), heads, N_BINS))
    count_depth, correct_depth = np.zeros(N_BINS), np.zeros(N_BINS)
    done = 0
    try:
        while done < n_seqs:
            bs = min(max(1, token_budget // length), n_seqs - done)
            d = task.build(bs, length, rng, skew=0.0)
            x, ans = d["tokens"].to(dev), d["answer_pos"].to(dev)
            logits, _ = model(x)
            pred = logits.gather(1, ans[..., None].expand(-1, -1, logits.size(-1))).argmax(-1)
            correct = (pred == x.gather(1, ans + 1)).cpu().numpy()
            key_pos, val_pos = gold_positions(x, ans, body_end)
            bins = np.clip((d["answer_depth"].numpy() * N_BINS).astype(int), 0, N_BINS - 1)
            for j in range(N_BINS):
                count_depth[j] += (bins == j).sum()
                correct_depth[j] += correct[bins == j].sum()
            for li, (i, mix) in enumerate(layers):
                p = attention_rows(mix, inputs[i], ans)

                def at(pos):
                    return p.gather(3, pos.view(bs, 1, nq, 1).expand(bs, heads, nq, 1)).squeeze(-1)

                val = at(val_pos).cpu().numpy()                                  # (B, H, Q)
                parts = {"value": val, "key": at(key_pos).cpu().numpy(), "sink": p[..., 0].cpu().numpy(),
                         "recent": p[..., body_end - RECENT:body_end].sum(-1).cpu().numpy(),
                         "top1_value": (p.argmax(-1) == val_pos.view(bs, 1, nq)).float().cpu().numpy()}
                for m, arr in parts.items():
                    sums[m][li] += arr.sum(axis=(0, 2))
                per_head = val.transpose(1, 0, 2)                                # (H, B, Q)
                for j in range(N_BINS):
                    if (bins == j).any():
                        by_depth[li, :, j] += per_head[:, bins == j].sum(-1)
            done += bs
    finally:
        for hk in hooks:
            hk.remove()
    total = count_depth.sum()
    rows = []
    for li, (i, _) in enumerate(layers):
        depth = by_depth[li] / np.maximum(count_depth, 1)
        rows.append({"layer": i, **{f"{m}_mass" if m != "top1_value" else m: (sums[m][li] / total).tolist()
                                    for m in MEASURES},
                     "value_mass_by_depth": depth.tolist(), "best_head": int(np.argmax(sums["value"][li]))})
    recall = {"recall": float(correct_depth.sum() / total),
              "recall_by_depth": (correct_depth / np.maximum(count_depth, 1)).tolist(), "n_queries": int(total)}
    return rows, recall


def best_reader(row):
    """(index into row['layers'], head) of the head with the most attention on the value."""
    li = max(range(len(row["layers"])),
             key=lambda j: row["layers"][j]["value_mass"][row["layers"][j]["best_head"]])
    return li, row["layers"][li]["best_head"]


def plot(rows, tag, lengths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.axisbelow": True, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e1e0d9",
                         "grid.linewidth": 0.6, "grid.linestyle": "-", "legend.frameon": False})
    first = {r["name"]: best_reader(r) for r in rows if r["seq_len"] == lengths[0]}
    centres = [(j + 0.5) / N_BINS for j in range(N_BINS)]
    fig, axes = plt.subplots(1, len(lengths), figsize=(6.8, 2.3), sharey=True, squeeze=False)
    for c, length in enumerate(lengths):
        ax = axes[0][c]
        for variant, color in COLORS.items():
            runs = [r for r in rows if r["variant"] == variant and r["seq_len"] == length
                    and r["learned"] and r["name"] in first]
            if not runs:
                continue
            arr = np.array([r["layers"][first[r["name"]][0]]["value_mass_by_depth"][first[r["name"]][1]]
                            for r in runs])
            if len(arr) > 1:
                ax.fill_between(centres, arr.min(0), arr.max(0), color=color, alpha=0.12, lw=0)
            ax.plot(centres, arr.mean(0), marker="o", ms=3.5, mec="white", mew=0.8, lw=1.5, color=color,
                    label=f"{LABELS[variant]} (n={len(arr)})")
        ax.set_title(f"{length} tokens")
        ax.set_xlabel("depth of the queried pair")
        ax.set_ylim(-0.03, 1.03)
    axes[0][0].set_ylabel("attention on the value")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/attention_{tag}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 4096])
    ap.add_argument("--seqs", type=int, default=32, help="sequences per length")
    ap.add_argument("--tokens", type=int, default=32768, help="tokens per batch")
    ap.add_argument("--set-task", nargs="*", help="task overrides, e.g. p_noop=0.0")
    ap.add_argument("--no-logn", action="store_true", help="read the models with the length scaling off")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda when available)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="results/evals")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGN["enabled"] = not a.no_logn
    overrides = parse_kv(a.set_task)
    out = []
    for path in sorted({p for g in a.glob for p in glob.glob(g)}):
        name = os.path.basename(path)[:-3]
        trained = json.load(open(path[:-3] + ".json", encoding="utf-8"))
        ck = torch.load(path, map_location=dev, weights_only=False)
        model = TinyLM(ModelConfig(**ck["model_config"])).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        task_cfg = {k: v for k, v in ck["task_config"].items() if k != "vocab_size"}
        task_cfg.update(overrides)
        task = HaystackTask(TaskConfig(**task_cfg))
        seed = trained["run"]["seed"]
        train_recall = {e["seq_len"]: e["recall"] for e in trained["evals"]}.get(trained["run"]["train_len"], 0.0)
        for i, length in enumerate(a.lengths):
            rng = np.random.default_rng(90_000 + 131 * i + seed)
            layers, recall = read_model(model, task, length, a.seqs, a.tokens, rng, dev)
            row = {"name": name, "variant": trained["run"]["variant"], "seed": seed,
                   "warmup": trained["run"].get("copy_warmup"), "learned": train_recall >= 0.9,
                   "seq_len": length, "logn": LOGN["enabled"], "task_overrides": overrides, **recall,
                   "layers": layers}
            out.append(row)
            li, hd = best_reader(row)
            best = layers[li]
            prof = best["value_mass_by_depth"][hd]
            print(f"{name:32s} len {length:5d} recall {recall['recall']:.3f} | layer {best['layer']} head {hd}: "
                  f"value {best['value_mass'][hd]:.3f} (first bin {prof[0]:.3f}, last {prof[-1]:.3f}), "
                  f"sink {best['sink_mass'][hd]:.3f}, recent {best['recent_mass'][hd]:.3f}", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"attention_{a.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "rows": out}, f)
    print(f"wrote {path} with {len(out)} rows")
    if a.plot and out:
        plot(out, a.tag, a.lengths)


if __name__ == "__main__":
    main()
