#!/usr/bin/env bash
# Every step behind the current results, in the order it was run, with the exact settings.
# Measured GPU time on a laptop RTX 3050 (4 GB): checkpoint probes about 1 h 15 min,
# seed-0 controlled runs about 1 h 30 min, short-plan stage 1 about 1 h 45 min.
# One GPU job at a time: on a 4 GB card, jobs that together exceed memory spill
# into host memory and slow to a crawl instead of failing.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
mkdir -p results/logs results/interventions

# 1. Properties the results rest on (CPU, about a minute).
CUDA_VISIBLE_DEVICES="" "$PY" tests/test_sinkprobe.py
CUDA_VISIBLE_DEVICES="" "$PY" tests/test_interventions.py

# 2. Released checkpoints: sink diagnostics and 16-needle retrieval by depth, 8 trials per depth.
"$PY" -m sinkprobe.hf_probe --model Qwen/Qwen3-0.6B-Base --needles 16 \
    --lengths 2048 4096 8192 16384 30000 --depths 11 --trials 8 --sink-trials 2 \
    --out results/hf/Qwen3-0.6B-Base.json | tee results/logs/hf_Qwen3-0.6B-Base.log
"$PY" -m sinkprobe.hf_probe --model Qwen/Qwen3.5-0.8B-Base --needles 16 \
    --lengths 2048 4096 8192 16384 --depths 11 --trials 8 --sink-trials 2 \
    --out results/hf/Qwen3.5-0.8B-Base.json | tee results/logs/hf_Qwen3.5-0.8B-Base.log

# 3. Seed 0 of the ladder and side branches, and of the delta-rule hybrid without the gate
#    (4000 steps, 1500 of them copy-rich warm-up; 512 evaluation sequences at four lengths).
"$PY" scripts/sweep.py --only ladder --seeds-main 1 --seeds-side 1 --steps 4000 \
    --copy-warmup 1500 --batch 32 --eval-seqs 512 --cuda-graph | tee results/logs/sweep_seed0.log
"$PY" -m sinkprobe.train --variant hybrid_nogate --seed 0 --steps 4000 --copy-warmup 1500 \
    --batch 32 --eval-seqs 512 --task p_noop=0.5 query_skew=0.0 --cuda-graph --save-model \
    --out results/runs/main/hybrid_nogate__p0.5__g0__s0.json

# 4. Short plan, stage 1: seed 1 of the key comparisons, NoPE and AttnRes alone,
#    sink interventions (controlled models that learned, and both checkpoints),
#    perplexity checks and the report.
bash scripts/run_short.sh 1

# 5. The first pilot re-read against the correct null model, the cache arithmetic,
#    and every table, figure and quoted number.
"$PY" scripts/reanalyze_pilot_v1.py
"$PY" scripts/run_costmodel.py
"$PY" scripts/make_report.py
