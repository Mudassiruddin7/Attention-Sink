#!/usr/bin/env bash
# Natural-text check for the layer-placement study: does the bound-key global
# layer first cost language modelling, and does its copying hold past the
# training length on real text? One run at a time by default: two byte-level
# runs at 512 tokens do not fit on a 4 GB card (set PAR=2 on a larger one).
#
#   bash scripts/run_text.sh
#
# Byte-level models trained on WikiText-2 with an ordinary next-byte objective
# and no warm-up, then scored on copying a 32-byte span repeated from a known
# depth of held-out WikiText-103 text, at 1x, 2x and 4x the training length.
# The control probe repeats a span that is not in the context, which gives the
# accuracy reachable from language statistics alone.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8 PY RUNS=results/runs/text
mkdir -p "$RUNS" results/logs/runs results/evals

# The new design and the same layer placed last, seed 0: a sanity check of the
# language-modelling cost, not a study. To add seed 1:
#   JOBS="hybrid_bka_first:0 hybrid_bka:0 hybrid_bka_first:1 hybrid_bka:1" bash scripts/run_text.sh
JOBS=${JOBS:-"hybrid_bka_first:0 hybrid_bka:0"}
export STEPS=${STEPS:-3000}
PAR=${PAR:-1}

train_one() {
  IFS=: read -r v s <<< "$1"
  name="text_${v}__s${s}"
  if [ -e "$RUNS/$name.json" ]; then echo "skip  $name"; return 0; fi
  extra=()
  case "$v" in *bka*) extra=(--model logn_ref=512) ;; esac
  echo "start $name $(date +%H:%M:%S)"
  "$PY" -m sinkprobe.train --task-kind text --variant "$v" --seed "$s" --steps "$STEPS" --copy-warmup 0 \
      --batch 32 --train-len 512 --eval-seqs 256 --eval-lens 512 1024 2048 --cuda-graph --save-model \
      "${extra[@]}" --out "$RUNS/$name.json" > "results/logs/runs/$name.log" 2>&1
  echo "done  $name exit $? $(date +%H:%M:%S)"
}
export -f train_one

echo "text training started $(date +%H:%M:%S)"
echo $JOBS | tr ' ' '\n' | xargs -P "$PAR" -I{} bash -c 'train_one {}'

"$PY" scripts/eval_ckpt.py --glob "$RUNS/text_*.pt" --lengths 512 1024 2048 --set-task control=True \
    --min-recall 0 --tag text_control > results/logs/eval_text_control.log 2>&1
echo "control probe exit $?"
"$PY" scripts/report_text.py > results/logs/report_text.log 2>&1
echo "text report exit $? $(date +%H:%M:%S)"
