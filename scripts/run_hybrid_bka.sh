#!/usr/bin/env bash
# Confirmation runs for the NoPE hybrid with bound keys (hybrid_bka), after its
# two-seed pilot. About 45 minutes on a laptop RTX 3050, two runs at a time.
#
#   bash scripts/run_hybrid_bka.sh
#
# Adds seeds 2 and 3 of hybrid_bka (reliability with the standard recipe),
# seed 0 of hybrid_bka without the copy-rich warm-up (reliability without it),
# and seed 2 of the plain NoPE hybrid (the baseline it is compared with). Then
# evaluates the length-scaling ablation without retraining and writes the
# report. Finished runs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8 PY RUNS=results/runs/main
mkdir -p "$RUNS" results/logs/runs results/evals

# The design being confirmed, e.g. VARIANT=hybrid_bka_first bash scripts/run_hybrid_bka.sh
export V=${VARIANT:-hybrid_bka}
# variant:seed:warm-up steps
JOBS="${V}:2:1500 ${V}:3:1500 ${V}:0:0 hybrid_nope:2:1500"

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

echo "confirmation runs started $(date +%H:%M:%S)"
echo $JOBS | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'train_one {}'

"$PY" scripts/eval_ckpt.py --glob "$RUNS/${V}__p0.5__g0__s?.pt" "$RUNS/${V}__p0.5__g0__s?__w0.pt" \
    --lengths 4096 --no-logn --tag "${V}_nologn" > "results/logs/eval_${V}_nologn.log" 2>&1
echo "${V} without length scaling exit $?"
"$PY" scripts/eval_ckpt.py --glob "$RUNS/hybrid_nope__p0.5__g0__s?.pt" "$RUNS/hybrid_attnres__p0.5__g0__s?.pt" \
    "$RUNS/softmax__p0.5__g0__s?.pt" "$RUNS/nope__p0.5__g0__s?.pt" \
    --lengths 4096 --tag long_baselines > results/logs/eval_long_baselines.log 2>&1
echo "baselines at 4096 exit $?"
"$PY" scripts/report_bka.py > results/logs/report_bka.log 2>&1
echo "report exit $? $(date +%H:%M:%S)"
