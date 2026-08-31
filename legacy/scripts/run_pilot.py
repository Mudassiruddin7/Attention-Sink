"""Train the four pilot variants and write every diagnostic to results/.

Usage
    python scripts/run_pilot.py --steps 2500 --seeds 3

The models are small enough to train on a laptop CPU. Nothing here needs a
GPU, network access or downloaded weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.data import NeedleTask, TaskConfig          # noqa: E402
from sinkprobe.metrics import evaluate, wilson_interval    # noqa: E402
from sinkprobe.model import ModelConfig, PilotLM           # noqa: E402

VARIANTS = ["dense", "dense_gated", "hybrid", "hybrid_ar"]
LABELS = {
    "dense": "Softmax + RoPE",
    "dense_gated": "Softmax + RoPE + output gate",
    "hybrid": "Hybrid 3:1 KDA style, NoPE",
    "hybrid_ar": "Hybrid 3:1 + Block AttnRes",
}


def train_one(variant, seed, args, task, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ModelConfig(variant=variant, vocab_size=task.cfg.vocab_size,
                      d_model=args.d_model, n_layers=args.n_layers,
                      n_heads=args.n_heads, max_len=args.max_len,
                      d_ff=args.d_ff, chunk=args.chunk)
    model = PilotLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    rng = np.random.default_rng(1000 + seed)

    t0 = time.time()
    curve = []
    model.train()
    for step in range(args.steps):
        x, y, _ = task.build(args.batch, args.train_len, rng=rng)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        if args.aux_lm_weight > 0:
            # Auxiliary next token objective over the whole sequence. Most of
            # the filler is unpredictable, so this term mainly supplies the
            # many query positions with nothing useful to retrieve. Those are
            # the positions that produce an attention sink in ordinary
            # language model training.
            aux = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                  x[:, 1:].reshape(-1))
            loss = loss + args.aux_lm_weight * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % args.log_every == 0 or step == 0:
            curve.append({"step": step + 1, "loss": float(loss.item())})
            print(f"  [{variant} s{seed}] step {step + 1:5d}  loss {loss.item():.4f}",
                  flush=True)
    train_secs = time.time() - t0

    out = {"variant": variant + args.tag, "label": LABELS[variant] + args.tag,
           "seed": seed,
           "params": model.n_params(), "train_seconds": round(train_secs, 1),
           "train_curve": curve, "evals": []}

    eval_rng = np.random.default_rng(90000 + seed)
    for L in args.eval_lens:
        res = evaluate(model, task, L, args.eval_batches, args.batch,
                       eval_rng, device)
        hits = int(round(res["accuracy"] * res["n_examples"]))
        lo, hi = wilson_interval(hits, res["n_examples"])
        res["ci_low"], res["ci_high"] = lo, hi
        out["evals"].append(res)
        print(f"  [{variant} s{seed}] len {L:5d}  acc {res['accuracy']:.3f} "
              f"sink {res['sink_mass']:.3f}  recency {res['recency_gap']:+.3f}",
              flush=True)

    # Per depth profile at the training length and at the longest length.
    prof = {}
    for L in (args.train_len, args.eval_lens[-1]):
        rows = []
        for d in np.linspace(0.05, 0.95, 10):
            hits, n = 0, 0
            r = np.random.default_rng(int(7777 + seed * 100 + d * 1000))
            for _ in range(args.profile_batches):
                x, y, _ = task.build(args.batch, L, depth=float(d), rng=r)
                with torch.no_grad():
                    logits, _ = model(x.to(device))
                pred = logits[:, -1, :].argmax(-1).cpu()
                hits += int((pred == y).sum())
                n += len(y)
            rows.append({"depth": round(float(d), 2), "acc": hits / n, "n": n})
        prof[str(L)] = rows
    out["depth_profile"] = prof
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-len", type=int, default=256)
    ap.add_argument("--eval-lens", type=int, nargs="+",
                    default=[256, 512, 1024, 2048])
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--profile-batches", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--aux-lm-weight", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-keys", type=int, default=24)
    ap.add_argument("--n-values", type=int, default=24)
    ap.add_argument("--n-distractors", type=int, default=6)
    ap.add_argument("--filler-period", type=int, default=24)
    ap.add_argument("--filler-noise", type=float, default=0.5)
    ap.add_argument("--variants", type=str, nargs="+", default=VARIANTS)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out", type=str, default="results/pilot.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("SINKPROBE_THREADS", os.cpu_count())))
    task = NeedleTask(TaskConfig(n_keys=args.n_keys, n_values=args.n_values,
                                 n_distractors=args.n_distractors,
                                 filler_period=args.filler_period,
                                 filler_noise=args.filler_noise))
    print(f"device {device}  vocab {task.cfg.vocab_size}  "
          f"train len {args.train_len}  steps {args.steps}", flush=True)

    runs = []
    for variant in args.variants:
        for seed in range(args.seeds):
            print(f"training {variant} seed {seed}", flush=True)
            runs.append(train_one(variant, seed, args, task, device))

    payload = {"config": vars(args), "device": device,
               "torch": torch.__version__, "runs": runs}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
