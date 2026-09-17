"""Push attention off position 0 in trained controlled models and re-measure.

    python scripts/intervene.py --biases 0 -1 -2 -4 -inf

Loads the weights saved by the sweep for the base condition (p_noop = 0.5,
query_skew = 0, default warm-up), adds a bias to the logit of position 0 in
every softmax layer, and re-runs the full evaluation. Each bias level sees the
same evaluation sequences, so the comparison across levels is paired.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.data import HaystackTask, TaskConfig                    # noqa: E402
from sinkprobe.interventions import SINK_BIAS, install_synthetic_bias  # noqa: E402
from sinkprobe.metrics import evaluate                                 # noqa: E402
from sinkprobe.model import ModelConfig, TinyLM                        # noqa: E402
from sinkprobe.train import autocast                                   # noqa: E402

KEEP = ["recall", "recency_gap", "middle_dip", "acc_q1", "acc_q2", "acc_q3", "acc_q4",
        "sink_mass", "sink_ratio", "sink_noop", "sink_copy", "sink_answer", "ce_copy",
        "ce_answer", "ce_noop", "entropy_norm", "act_max", "gate_mean", "virtual_sink", "recall_no_marker", "marker_rate"]


def parse_biases(items):
    """'off' (or -inf) removes position 0 entirely; commas may separate values."""
    out = []
    for item in items:
        for tok in str(item).split(","):
            tok = tok.strip().lower()
            if tok:
                out.append(float("-inf") if tok in ("off", "-inf", "ninf") else float(tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="results/runs/main")
    ap.add_argument("--variants", nargs="+",
                    default=["softmax", "gate", "hybrid", "hybrid_nogate", "sinklogit", "attnres"])
    ap.add_argument("--biases", nargs="+", default=["0", "-1", "-2", "-4", "off"],
                    help="logit shifts for position 0; 'off' removes it entirely (-inf)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--eval-seqs", type=int, default=256)
    ap.add_argument("--min-recall", type=float, default=0.9,
                    help="skip models whose recall at the training length is below this")
    ap.add_argument("--out", default="results/interventions/synthetic.json")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda when available)")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    install_synthetic_bias()
    biases = parse_biases(a.biases)
    rows, skipped = [], []
    paths = [p for p in sorted(glob.glob(os.path.join(a.runs_dir, "*__p0.5__g0__s*.pt")))
             if "__w" not in os.path.basename(p)]
    for path in paths:
        name = os.path.basename(path)[:-3]
        variant, seed = name.split("__")[0], int(name.split("__s")[-1])
        if variant not in a.variants:
            continue
        with open(path[:-3] + ".json", encoding="utf-8") as f:
            trained = json.load(f)
        recall = min(e["recall"] for e in trained["evals"] if e["seq_len"] == trained["run"]["train_len"])
        if recall < a.min_recall:
            print(f"{name:28s} skipped: recall {recall:.3f} at the training length", flush=True)
            skipped.append({"name": name, "recall": recall})
            continue
        ck = torch.load(path, map_location=dev, weights_only=False)
        model = TinyLM(ModelConfig(**ck["model_config"])).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        task = HaystackTask(TaskConfig(**{k: v for k, v in ck["task_config"].items()
                                          if k != "vocab_size"}))
        for bias in biases:
            SINK_BIAS["value"] = bias
            for i, L in enumerate(a.lengths):
                rng = np.random.default_rng(70_000 + 97 * i + seed)
                with autocast(dev, True):
                    res = evaluate(model, task, L, a.eval_seqs, rng, dev)
                row = {"name": name, "variant": variant, "seed": seed, "bias": bias, "seq_len": L}
                row.update({k: res[k] for k in KEEP if k in res})
                row["profile"] = res["profile"]
                rows.append(row)
                print(f"{name:28s} bias {bias:>5} len {L:5d} recall {res['recall']:.3f} "
                      f"sink {res.get('sink_mass', float('nan')):.4f} "
                      f"noop {res.get('sink_noop', float('nan')):.4f} gap {res['recency_gap']:+.3f}",
                      flush=True)
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None
    SINK_BIAS["value"] = 0.0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "rows": rows, "skipped": skipped}, f)
    print(f"wrote {a.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
