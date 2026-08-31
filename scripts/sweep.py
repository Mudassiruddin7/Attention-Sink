"""Launch every training run behind the paper, a few at a time on one GPU.

    python scripts/sweep.py --workers 2 --cuda-graph        # run everything missing
    python scripts/sweep.py --only factorial --dry-run

All experiments share one pool of runs, so a configuration that appears in
two experiments is trained once. Runs are ordered seed by seed, so a partial
sweep is still balanced across conditions. A run whose JSON already exists is
skipped, so the sweep can be stopped and resumed at any time.

ladder     softmax -> gate -> hybrid -> hybrid_nope -> hybrid_attnres, plus the
           side branches sinklogit, softpick and hybrid_nodelta, at the base
           setting p_noop = 0.5, query_skew = 0
factorial  each mechanism added alone to the softmax baseline
           (gate, hybrid_nogate, nope, attnres); with softmax, gate and hybrid
           this includes a 2x2 of output gate by delta-rule layers
noop       p_noop in {0, 0.25, 0.5, 0.75, 1} for softmax, gate and hybrid
skew       query_skew in {-4, 0, 4} for softmax, gate and hybrid
warmup     softmax, gate and hybrid trained without the copy-rich warm-up

Short plan, for a laptop GPU (--only core dose2 skew2):
core       softmax, gate, hybrid_nogate, hybrid, nope, attnres, sinklogit and
           hybrid_attnres at the base setting
dose2      p_noop 0 and 1 for softmax and gate (0.5 comes from core)
skew2      query_skew 4 for softmax and gate

Runs with the default warm-up are named variant__p..__g..__s..; any other
warm-up is appended as __w<steps>.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.model import MAIN_LADDER, SIDE_BRANCHES   # noqa: E402

KNOB_VARIANTS = ["softmax", "gate", "hybrid"]
FACTOR_VARIANTS = ["hybrid_nogate", "nope", "attnres"]    # gate is already in the ladder
WARMUP_VARIANTS = ["softmax", "gate", "hybrid"]
# Short plan (about an hour on a laptop GPU): every mechanism alone plus the
# full hybrid recipe at the base setting, and the no-op dose and query skew on
# the two fast softmax variants.
CORE_VARIANTS = ["softmax", "gate", "hybrid_nogate", "hybrid", "nope", "attnres",
                 "sinklogit", "hybrid_attnres"]
SHORT_PLAN = ["core", "dose2", "skew2"]
EXPERIMENTS = ["ladder", "factorial", "noop", "skew", "warmup"]
# Variants whose evaluation holds several full attention maps get the GPU alone.
SOLO_VARIANTS = {"softpick"}


def run_name(variant, p_noop, skew, seed, warmup=None):
    name = f"{variant}__p{p_noop:g}__g{skew:g}__s{seed}"
    return name if warmup is None else name + f"__w{warmup}"


def jobs_for(experiments, n_seeds_main, n_seeds_side):
    """name -> (variant, p_noop, skew, seed, warmup or None for the default)."""
    jobs = {}

    def add(v, p, g, seed, w=None):
        jobs.setdefault(run_name(v, p, g, seed, w), (v, p, g, seed, w))

    for seed in range(max(n_seeds_main, n_seeds_side)):
        if "ladder" in experiments:
            for v in MAIN_LADDER:
                if seed < n_seeds_main:
                    add(v, 0.5, 0.0, seed)
            for v in SIDE_BRANCHES:
                if seed < n_seeds_side:
                    add(v, 0.5, 0.0, seed)
        if "factorial" in experiments and seed < n_seeds_main:
            for v in ["softmax", "gate", "hybrid"] + FACTOR_VARIANTS:
                add(v, 0.5, 0.0, seed)
        if "noop" in experiments and seed < n_seeds_side:
            for p in (0.0, 0.25, 0.5, 0.75, 1.0):
                for v in KNOB_VARIANTS:
                    add(v, p, 0.0, seed)
        if "skew" in experiments and seed < n_seeds_side:
            for g in (-4.0, 0.0, 4.0):
                for v in KNOB_VARIANTS:
                    add(v, 0.5, g, seed)
        if "warmup" in experiments and seed < n_seeds_side:
            for v in WARMUP_VARIANTS:
                add(v, 0.5, 0.0, seed, 0)
        if "core" in experiments and seed < n_seeds_main:
            for v in CORE_VARIANTS:
                add(v, 0.5, 0.0, seed)
        if "dose2" in experiments and seed < n_seeds_side:
            for p in (0.0, 1.0):
                for v in ("softmax", "gate"):
                    add(v, p, 0.0, seed)
        if "skew2" in experiments and seed < n_seeds_side:
            for v in ("softmax", "gate"):
                add(v, 0.5, 4.0, seed)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=EXPERIMENTS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds-main", type=int, default=3)
    ap.add_argument("--seeds-side", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-seqs", type=int, default=512)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--copy-warmup", type=int, default=1500)
    ap.add_argument("--hybrid-copy-warmup", type=int, default=None,
                    help="default warm-up for variants with delta rule layers, if different")
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--runs-dir", default="results/runs/main")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = jobs_for(a.only, a.seeds_main, a.seeds_side)
    todo = [(n, spec) for n, spec in jobs.items()
            if not os.path.exists(os.path.join(a.runs_dir, n + ".json"))]
    print(f"{len(jobs)} runs in {', '.join(a.only)}; {len(todo)} still to do")
    if a.dry_run:
        for n, _ in todo:
            print("  ", n)
        return 0

    os.makedirs(a.runs_dir, exist_ok=True)
    log_dir = os.path.join("results", "logs", "runs")
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    running, failed = [], []
    t0 = time.time()
    queue = list(todo)
    while queue or running:
        while queue and len(running) < a.workers:
            solo_running = any(n.split("__")[0] in SOLO_VARIANTS for n, _, _ in running)
            if solo_running or (queue[0][1][0] in SOLO_VARIANTS and running):
                break
            name, (v, p, g, seed, w) = queue.pop(0)
            warm = a.copy_warmup if w is None else w
            if w is None and v.startswith("hybrid") and a.hybrid_copy_warmup is not None:
                warm = a.hybrid_copy_warmup
            cmd = [sys.executable, "-m", "sinkprobe.train", "--variant", v, "--seed", str(seed),
                   "--steps", str(a.steps), "--batch", str(a.batch),
                   "--eval-seqs", str(a.eval_seqs), "--eval-lens", *[str(x) for x in a.eval_lens],
                   "--copy-warmup", str(warm),
                   "--task", f"p_noop={p}", f"query_skew={g}",
                   "--out", os.path.join(a.runs_dir, name + ".json"), "--save-model"]
            if a.cuda_graph:
                cmd.append("--cuda-graph")
            log = open(os.path.join(log_dir, name + ".log"), "w", encoding="utf-8")
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
            print(f"[{time.time() - t0:7.0f}s] start {name}", flush=True)
        time.sleep(5)
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = "done" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode != 0:
                    failed.append(name)
                left = len(queue) + len(running)
                print(f"[{time.time() - t0:7.0f}s] {status} {name}  ({left} left)", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min; {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
