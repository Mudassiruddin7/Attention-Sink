#!/usr/bin/env bash
# Everything still to run, most valuable first. Resumable: each step skips work
# whose output already exists. Jobs run one at a time, for a 4 GB laptop GPU.
#
#   bash scripts/run_pending.sh                                  # every step, about 1 h 45 min
#   STEPS="regime attention longer pairs sink reports" bash scripts/run_pending.sh
#
#   text       WikiText sanity check: seed 0 of the bound-key layer first and of
#              the same layer placed last (about 35 minutes)
#   regime     the placement models on copy-rich data (p_noop 0) at 256 and 4096
#              tokens, without retraining (about 15 minutes)
#   attention  where the global layers look when they answer, by depth and length,
#              on the standard and the copy-rich data (about 10 minutes)
#   longer     learned models at 8192 and 16384 tokens (32x and 64x), as trained and,
#              for the NoPE hybrid, with the length scaling on; the attention check
#              at 16384 tokens (about 20 minutes)
#   pairs      16, 32 and 64 key-value pairs at 256 and 2048 tokens: retrieval by
#              attention should barely notice, a recurrent memory should (about 10 minutes)
#   sink       attention to position 0 removed in the placement models, at 256 and
#              2048 tokens (about 10 minutes)
#   rescore    every placement model re-evaluated at 256 to 16384 tokens with all three
#              answer metrics (exact match, markers excluded, marker rate), so no table
#              mixes numbers from different evaluation passes (about 45 minutes)
#   reports    tables, figures, statistics and the scored predictions (CPU, about a minute)
#
# The predictions for these steps were written before they ran (method notes,
# sections 17 and 18); scripts/table_predictions.py scores them.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
STEPS=${STEPS:-"text regime attention longer pairs sink rescore reports"}
RUNS=results/runs/main
# The four cells of the placement study: global layer first or last, with or without bound keys.
MODELS=("$RUNS/hybrid_bka_first__p0.5__g0__s?.pt" "$RUNS/hybrid_bka__p0.5__g0__s?.pt"
        "$RUNS/hybrid_nope_first__p0.5__g0__s?.pt" "$RUNS/hybrid_nope__p0.5__g0__s?.pt")
mkdir -p results/logs results/evals results/interventions

wanted() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
already() { [ -e "$1" ] && echo "skip  $1"; }

echo "pending runs started $(date +%H:%M:%S)"
if wanted text; then
  bash scripts/run_text.sh
fi

if wanted regime && ! already results/evals/regime_copyrich.json; then
  "$PY" scripts/eval_ckpt.py --glob "${MODELS[@]}" --lengths 256 4096 --eval-seqs 128 \
      --set-task p_noop=0.0 --tag regime_copyrich > results/logs/eval_regime_copyrich.log 2>&1
  echo "regime evaluation exit $? $(date +%H:%M:%S)"
fi

if wanted attention; then
  for regime in standard copyrich; do
    already "results/evals/attention_${regime}.json" && continue
    extra=()
    [ "$regime" = copyrich ] && extra=(--set-task p_noop=0.0)
    "$PY" scripts/retrieval_attention.py --glob "${MODELS[@]}" --lengths 256 4096 --seqs 32 \
        ${extra[@]+"${extra[@]}"} --tag "$regime" --plot > "results/logs/attention_${regime}.log" 2>&1
    echo "attention ${regime} exit $? $(date +%H:%M:%S)"
  done
fi

if wanted longer; then
  # No attention diagnostics here: their full attention maps do not fit at 16K tokens on 4 GB.
  if ! already results/evals/longer.json; then
    "$PY" scripts/eval_ckpt.py --glob "${MODELS[@]}" --lengths 8192 16384 --eval-seqs 48 --batch 2 \
        --collect-seqs 0 --learned-only --tag longer > results/logs/eval_longer.log 2>&1
    echo "longer contexts exit $? $(date +%H:%M:%S)"
  fi
  if ! already results/evals/longer_llls_nope_logn.json; then
    "$PY" scripts/eval_ckpt.py --glob "$RUNS/hybrid_nope__p0.5__g0__s?.pt" --lengths 8192 16384 \
        --eval-seqs 48 --batch 2 --collect-seqs 0 --learned-only --set logn_ref=256 \
        --tag longer_llls_nope_logn > results/logs/eval_longer_llls_nope_logn.log 2>&1
    echo "longer contexts, NoPE hybrid with length scaling exit $? $(date +%H:%M:%S)"
  fi
  if ! already results/evals/attention_longer.json; then
    "$PY" scripts/retrieval_attention.py --glob "$RUNS/hybrid_bka_first__p0.5__g0__s?.pt" \
        "$RUNS/hybrid_bka__p0.5__g0__s?.pt" --lengths 256 16384 --seqs 16 --tokens 16384 \
        --tag longer --plot > results/logs/attention_longer.log 2>&1
    echo "attention at 16384 tokens exit $? $(date +%H:%M:%S)"
  fi
fi

if wanted pairs; then
  for n in 16 32 64; do
    already "results/evals/pairs${n}.json" && continue
    "$PY" scripts/eval_ckpt.py --glob "${MODELS[@]}" --lengths 256 2048 --eval-seqs 64 \
        --set-task "n_pairs=$n" --tag "pairs${n}" > "results/logs/eval_pairs${n}.log" 2>&1
    echo "${n} key-value pairs exit $? $(date +%H:%M:%S)"
  done
fi

if wanted sink && ! already results/interventions/placement.json; then
  "$PY" scripts/intervene.py --variants hybrid_bka_first hybrid_bka hybrid_nope_first hybrid_nope \
      --biases 0 off --lengths 256 2048 --eval-seqs 128 --out results/interventions/placement.json \
      > results/logs/intervene_placement.log 2>&1
  echo "sink removal exit $? $(date +%H:%M:%S)"
fi

if wanted rescore; then
  if ! already results/evals/rescore_main.json; then
    "$PY" scripts/eval_ckpt.py --glob "$RUNS/*__p0.5__g0__s?.pt" --lengths 256 1024 2048 4096 --eval-seqs 128 --tag rescore_main > results/logs/eval_rescore_main.log 2>&1
    echo "rescore, 1x to 16x exit $? $(date +%H:%M:%S)"
  fi
  if ! already results/evals/rescore_longer.json; then
    "$PY" scripts/eval_ckpt.py --glob "${MODELS[@]}" --lengths 8192 16384 --eval-seqs 48 --batch 2 --collect-seqs 0 --learned-only --tag rescore_longer > results/logs/eval_rescore_longer.log 2>&1
    echo "rescore, 32x and 64x exit $? $(date +%H:%M:%S)"
  fi
  for n in 16 64; do
    already "results/evals/pairs${n}_rescored.json" && continue
    "$PY" scripts/eval_ckpt.py --glob "${MODELS[@]}" --lengths 256 2048 --eval-seqs 64 --set-task "n_pairs=$n" --tag "pairs${n}_rescored" > "results/logs/eval_pairs${n}_rescored.log" 2>&1
    echo "rescore, ${n} pairs exit $? $(date +%H:%M:%S)"
  done
fi

if wanted reports; then
  for s in report_bka stats_placement robustness_summary more_findings table_predictions report_text; do
    "$PY" "scripts/$s.py" > "results/logs/$s.log" 2>&1
    echo "$s exit $?"
  done
fi
echo "pending runs finished $(date +%H:%M:%S)"
