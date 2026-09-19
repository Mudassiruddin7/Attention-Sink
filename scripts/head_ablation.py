"""Causal test of where retrieval happens: remove the output of a whole global layer, or of
one head, at test time and measure recall on the same test inputs.

    python scripts/head_ablation.py --glob "results/runs/main/hybrid_bka_first__p0.5__g0__s*.pt" \
        --tag ablation_bkf

Removing an output means zeroing the part of the attention output that belongs to a head
(after the output gate), just before the output projection. Nothing else in the network
changes, and every condition of a model is scored on the same test sequences, so the
comparison between conditions is paired. Only runs with the standard warm-up that learned
the task are used. Writes results/evals/<tag>.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.data import HaystackTask, TaskConfig      # noqa: E402
from sinkprobe.metrics import evaluate                   # noqa: E402
from sinkprobe.model import ModelConfig, TinyLM          # noqa: E402


def remove_heads(attn, heads):
    """Zero the output slices of the given heads of one attention layer."""
    dh = attn.dh

    def hook(_module, args):
        (o,) = args
        o = o.clone()
        for h in heads:
            o[..., h * dh:(h + 1) * dh] = 0
        return (o,)

    return attn.out.register_forward_pre_hook(hook)


def conditions(model):
    """Intact model, each global layer removed, and each head of the first global layer removed."""
    glob_layers = [i for i, blk in enumerate(model.blocks) if blk.kind == "S"]
    n_heads = model.blocks[glob_layers[0]].mix.h
    out = [("intact", {})]
    out += [(f"layer{i}", {i: list(range(n_heads))}) for i in glob_layers]
    out += [(f"layer{glob_layers[0]}_head{h}", {glob_layers[0]: [h]}) for h in range(n_heads)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 4096])
    ap.add_argument("--seqs", type=int, nargs="+", default=[64, 32], help="test inputs per length")
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="results/evals")
    a = ap.parse_args()
    assert len(a.seqs) == len(a.lengths)
    torch.set_grad_enabled(False)

    paths = sorted({p for g in a.glob for p in glob.glob(g)})
    rows = []
    for path in paths:
        name = os.path.basename(path)[:-3]
        with open(path[:-3] + ".json", encoding="utf-8") as f:
            trained = json.load(f)
        if trained["run"].get("copy_warmup", 1500) != 1500:
            continue
        train_len = trained["run"]["train_len"]
        train_recall = min(e["recall"] for e in trained["evals"] if e["seq_len"] == train_len)
        if train_recall < a.min_recall:
            print(f"{name:32s} skipped: not learned", flush=True)
            continue
        ck = torch.load(path, map_location=a.device, weights_only=False)
        model = TinyLM(ModelConfig(**ck["model_config"])).to(a.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        task = HaystackTask(TaskConfig(**{k: v for k, v in ck["task_config"].items() if k != "vocab_size"}))
        seed = trained["run"]["seed"]
        for cond, removed in conditions(model):
            hooks = [remove_heads(model.blocks[i].mix, heads) for i, heads in removed.items()]
            for i, (L, n) in enumerate(zip(a.lengths, a.seqs)):
                t0 = time.time()
                rng = np.random.default_rng(90_000 + 97 * i + seed)
                res = evaluate(model, task, L, n, rng, a.device, collect_seqs=0)
                rows.append({"name": name, "variant": trained["run"]["variant"], "seed": seed,
                             "condition": cond, "removed": {str(k): v for k, v in removed.items()},
                             "seq_len": L, "n_seqs": n, "recall": res["recall"],
                             "recall_no_marker": res.get("recall_no_marker"),
                             "recall_after_first": res.get("recall_after_first")})
                print(f"{name:32s} {cond:14s} len {L:5d} recall {res['recall']:.3f} "
                      f"no-marker {res.get('recall_no_marker', float('nan')):.3f} ({time.time() - t0:.0f}s)",
                      flush=True)
            for hk in hooks:
                hk.remove()
        del model
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "rows": rows}, f, indent=1)
    print(f"wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
