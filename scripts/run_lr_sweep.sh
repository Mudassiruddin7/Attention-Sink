#!/usr/bin/env bash
# Learning-rate check for the placement result: BKF (bound-key global layer first) and the
# same layer placed last, at learning rates 1e-3 and 1e-2 (the main runs use 3e-3), seeds 0
# and 1, with the standard recipe otherwise. A design and its matched control share the GPU.
# Runs go to results/runs/lr_sweep so that the main tables are unchanged.
#
#   PY=path/to/python bash scripts/run_lr_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8 PY RUNS=results/runs/lr_sweep
mkdir -p "$RUNS" results/logs/runs

JOBS=${JOBS:-"hybrid_bka_first:0:1e-3 hybrid_bka:0:1e-3 hybrid_bka_first:1:1e-3 hybrid_bka:1:1e-3 \
hybrid_bka_first:0:1e-2 hybrid_bka:0:1e-2 hybrid_bka_first:1:1e-2 hybrid_bka:1:1e-2"}

train_one() {
  IFS=: read -r v s lr <<< "$1"
  name="${v}__p0.5__g0__s${s}__lr${lr}"
  if [ -e "$RUNS/$name.json" ]; then echo "skip  $name"; return 0; fi
  echo "start $name $(date +%H:%M:%S)"
  "$PY" -m sinkprobe.train --variant "$v" --seed "$s" --steps 4000 --copy-warmup 1500 --batch 32 \
      --lr "$lr" --eval-seqs 256 --eval-lens 256 1024 2048 4096 --task p_noop=0.5 query_skew=0.0 \
      --cuda-graph --save-model --out "$RUNS/$name.json" > "results/logs/runs/$name.log" 2>&1
  echo "done  $name exit $? $(date +%H:%M:%S)"
}
export -f train_one

echo "learning-rate sweep started $(date +%H:%M:%S)"
echo $JOBS | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'train_one {}'
echo "learning-rate sweep finished $(date +%H:%M:%S)"
