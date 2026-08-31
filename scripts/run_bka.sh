#!/usr/bin/env bash
# Bound-Key Attention study on a laptop GPU (about 1 hour after the pilot).
#
#   bash scripts/run_bka.sh
#
# Trains the runs the comparison needs with the same 4000-step recipe as every
# other run, then evaluates two ablations without retraining (the length
# scaling has no parameters and is the identity at the training length) and the
# baselines at 16x the training length, and writes the report. Finished runs
# are skipped, so the script can be stopped and restarted.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8 PY RUNS=results/runs/main
mkdir -p "$RUNS" results/logs/runs results/evals

# variant:seed:warm-up steps (1500 is the default recipe; 0 trains without warm-up)
JOBS="bka:1:1500 bka:1:0 bka_rope:0:1500 softmax:0:0 bka:2:1500 bka:2:0 bka_rope:1:1500 softmax:1:0 \
bka_rope:2:1500 softmax:2:0 softmax:2:1500 nope:1:1500 nope:2:1500"

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

echo "training started $(date +%H:%M:%S)"
echo $JOBS | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'train_one {}'

echo "evaluation-only ablations started $(date +%H:%M:%S)"
"$PY" scripts/eval_ckpt.py --glob "$RUNS/bka__p0.5__g0__s?.pt" "$RUNS/bka__p0.5__g0__s?__w0.pt" \
    --lengths 1024 2048 4096 --no-logn --tag bka_nologn > results/logs/eval_bka_nologn.log 2>&1
echo "bka without length scaling exit $?"
"$PY" scripts/eval_ckpt.py --glob "$RUNS/nope__p0.5__g0__s?.pt" \
    --lengths 1024 2048 4096 --set logn_ref=256 --tag nope_logn > results/logs/eval_nope_logn.log 2>&1
echo "nope with length scaling exit $?"
"$PY" scripts/eval_ckpt.py --glob "$RUNS/softmax__p0.5__g0__s?.pt" "$RUNS/gate__p0.5__g0__s?.pt" \
    "$RUNS/nope__p0.5__g0__s?.pt" "$RUNS/attnres__p0.5__g0__s?.pt" "$RUNS/sinklogit__p0.5__g0__s?.pt" \
    "$RUNS/hybrid*__p0.5__g0__s?.pt" --lengths 4096 --tag long_baselines > results/logs/eval_long_baselines.log 2>&1
echo "baselines at 4096 exit $?"

"$PY" scripts/report_bka.py > results/logs/report_bka.log 2>&1
echo "report exit $? $(date +%H:%M:%S)"
