#!/usr/bin/env bash
# Short GPU plan for a laptop.
#
#   bash scripts/run_short.sh 1    # about 1 h 45 min on a laptop RTX 3050 (measured):
#                                  # seed 1 of the key comparisons, NoPE and AttnRes alone
#                                  # (83 min), sink interventions, perplexity checks, report
#   bash scripts/run_short.sh 2    # optional, about 1 h 30 min: brings them to 3 seeds
#
# Uses the same 4000-step recipe as the seed-0 runs already in results/runs/main,
# so nothing finished is wasted. New runs evaluate 256 sequences at 256, 1024 and
# 2048 tokens. A run whose JSON exists is skipped, so the script can be stopped
# with Ctrl+C and started again.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8 PY RUNS=results/runs/main
STAGE=${1:-1}
mkdir -p "$RUNS" results/logs/runs results/interventions

if [ "$STAGE" = 1 ]; then
  JOBS="hybrid:1 softmax:1 hybrid_nogate:1 gate:1 hybrid_attnres:1 nope:0 hybrid_nope:1 attnres:0"
else
  JOBS="hybrid:2 softmax:2 hybrid_nogate:2 gate:2 hybrid_attnres:2 nope:1 attnres:1 nope:2 attnres:2"
fi

train_one() {
  v=${1%%:*}; s=${1##*:}
  name="${v}__p0.5__g0__s${s}"
  if [ -e "$RUNS/$name.json" ]; then echo "skip  $name"; return 0; fi
  echo "start $name $(date +%H:%M:%S)"
  "$PY" -m sinkprobe.train --variant "$v" --seed "$s" --steps 4000 --copy-warmup 1500 --batch 32 \
      --eval-seqs 256 --eval-lens 256 1024 2048 --task p_noop=0.5 query_skew=0.0 \
      --cuda-graph --save-model --out "$RUNS/$name.json" > "results/logs/runs/$name.log" 2>&1
  echo "done  $name exit $? $(date +%H:%M:%S)"
}
export -f train_one

echo "stage $STAGE training started $(date +%H:%M:%S)"
echo $JOBS | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'train_one {}'

echo "synthetic interventions started $(date +%H:%M:%S)"
"$PY" scripts/intervene.py --runs-dir "$RUNS" \
    --variants softmax gate nope attnres hybrid hybrid_nogate hybrid_nope hybrid_attnres sinklogit \
    --biases 0 off --lengths 256 1024 --eval-seqs 256 --out results/interventions/synthetic.json \
    > results/logs/intervene_synthetic.log 2>&1
echo "synthetic interventions exit $?"

if [ "$STAGE" = 1 ]; then
  for m in Qwen/Qwen3-0.6B-Base Qwen/Qwen3.5-0.8B-Base; do
    tag=${m#*/}
    echo "$tag intervention started $(date +%H:%M:%S)"
    "$PY" -m sinkprobe.hf_intervene --model "$m" --biases 0 off --lengths 2048 4096 --depths 11 --trials 4 \
        --out "results/interventions/hf_$tag.json" > "results/logs/intervene_$tag.log" 2>&1
    echo "$tag intervention exit $?"
    "$PY" scripts/hf_ppl_check.py --model "$m" --biases 0 off --windows 16 --length 2048 \
        --out "results/interventions/ppl_$tag.json" > "results/logs/ppl_$tag.log" 2>&1
    echo "$tag perplexity check exit $?"
  done
fi

"$PY" scripts/make_report.py > results/logs/report.log 2>&1
echo "report exit $? $(date +%H:%M:%S)"
