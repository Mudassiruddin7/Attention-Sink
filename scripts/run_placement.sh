#!/usr/bin/env bash
# Layer placement in the NoPE hybrid: the global layer first in each block
# (SLLL) instead of last (LLLS). Two runs at a time on a laptop RTX 3050, about
# 22 minutes per pair.
#
#   bash scripts/run_placement.sh 1   # bound-key layer first, seeds 0-3, standard recipe
#   bash scripts/run_placement.sh 2   # the same without warm-up (seeds 0, 1), and the
#                                     # global layer first without bound keys (seeds 0, 1)
#   bash scripts/run_placement.sh 4   # seeds 6-8 of the bound-key layer first and of the same
#                                     # layer placed last, for the success-rate comparison
#   bash scripts/run_placement.sh 3   # confirmation: seeds 4-5 of the bound-key layer first
#                                     # and of the same layer placed last, run side by side,
#                                     # and seeds 2-3 of the layer first without bound keys
#
# Stage 1 prediction, fixed before its runs (method notes, section 15): every
# seed passes 90% training recall before the warm-up ends at step 1500, and
# recall after the first query slot stays at or above 85% at 4096 tokens.
# Stage 2 separates where the global layer sits from what its keys contain,
# and tests learning without the copy-rich warm-up. Stage 3 adds seeds to the
# two comparisons that carry the claims. Finished runs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8 PY RUNS=results/runs/main
mkdir -p "$RUNS" results/logs/runs results/evals

STAGE=${1:-1}
# variant:seed:warm-up steps. Jobs start two at a time in this order, so a design
# and its matched control share the GPU.
case "$STAGE" in
  1) DEFAULT="hybrid_bka_first:0:1500 hybrid_bka_first:1:1500 hybrid_bka_first:2:1500 hybrid_bka_first:3:1500" ;;
  2) DEFAULT="hybrid_bka_first:0:0 hybrid_nope_first:0:1500 hybrid_bka_first:1:0 hybrid_nope_first:1:1500" ;;
  3) DEFAULT="hybrid_bka_first:4:1500 hybrid_bka:4:1500 hybrid_nope_first:2:1500 hybrid_nope_first:3:1500 hybrid_bka_first:5:1500 hybrid_bka:5:1500" ;;
  4) DEFAULT="hybrid_bka_first:6:1500 hybrid_bka:6:1500 hybrid_bka_first:7:1500 hybrid_bka:7:1500 hybrid_bka_first:8:1500 hybrid_bka:8:1500" ;;
  *) DEFAULT="hybrid_bka_first:9:1500 hybrid_bka:9:1500 hybrid_bka_first:10:1500 hybrid_bka:10:1500" ;;
esac
JOBS=${JOBS:-$DEFAULT}

train_one() {
  IFS=: read -r v s w <<< "$1"
  name="${v}__p0.5__g0__s${s}"
  [ "$w" = 0 ] && name="${name}__w0"
  if [ -e "$RUNS/$name.json" ]; then echo "skip  $name"; return 0; fi
  echo "start $name $(date +%H:%M:%S)"
  "$PY" -m sinkprobe.train --variant "$v" --seed "$s" --steps 4000 --copy-warmup "$w" --batch 32 \
      --eval-seqs 256 --eval-lens 256 1024 2048 4096 --task p_noop=0.5 query_skew=0.0 \
      --cuda-graph --save-model --out "$RUNS/$name.json" > "results/logs/runs/$name.log" 2>&1
  echo "done  $name exit $? $(date +%H:%M:%S)"
}
export -f train_one

echo "placement stage $STAGE started $(date +%H:%M:%S)"
echo $JOBS | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'train_one {}'

if [ "$STAGE" = 1 ]; then
  "$PY" scripts/eval_ckpt.py --glob "$RUNS/hybrid_bka_first__p0.5__g0__s?.pt" --lengths 4096 --no-logn \
      --tag hybrid_bka_first_nologn > results/logs/eval_hybrid_bka_first_nologn.log 2>&1
  echo "bound-key layer first without length scaling exit $?"
elif [ "$STAGE" = 2 ]; then
  "$PY" scripts/eval_ckpt.py --glob "$RUNS/hybrid_nope_first__p0.5__g0__s?.pt" --lengths 4096 --no-logn \
      --tag hybrid_nope_first_nologn > results/logs/eval_hybrid_nope_first_nologn.log 2>&1
  echo "NoPE layer first without length scaling exit $?"
  # The usual layout with the same length scaling; it has no parameters, so no retraining.
  "$PY" scripts/eval_ckpt.py --glob "$RUNS/hybrid_nope__p0.5__g0__s?.pt" --lengths 4096 --set logn_ref=256 \
      --tag llls_nope_logn > results/logs/eval_llls_nope_logn.log 2>&1
  echo "NoPE hybrid with length scaling exit $?"
fi
"$PY" scripts/report_bka.py > results/logs/report_bka.log 2>&1
echo "report exit $? $(date +%H:%M:%S)"
"$PY" scripts/stats_placement.py > results/logs/stats_placement.log 2>&1
echo "statistics exit $? $(date +%H:%M:%S)"
