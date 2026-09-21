# SinkProbe

**Project page:** https://mudassiruddin7.github.io/Attention-Sink/

Code for measuring attention sinks and position bias separately, in small
models trained under controlled conditions and in released checkpoints.

SinkProbe was introduced by Sara Rizwan, Samaanah Abdus Salam and Mohammed
Mudassir Uddin in
[*Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token
Context?*](https://arxiv.org/abs/2609.08574) (arXiv:2609.08574, 2026; code at
[sararizwan7/Attention-Mechanisms-in-1M-Context-Window](https://github.com/sararizwan7/Attention-Mechanisms-in-1M-Context-Window)).
Their first pilot is kept in `legacy/` and `results/pilot_v1/`. This
repository extends that code with a new testbed, the layer-placement and
bound-key experiments, and the probes of released checkpoints described below.

Tables and figures are rebuilt from the saved evaluations by
`scripts/paper_tables.py` and `scripts/paper_figures.py`, which write into
`paper/` (not tracked).

| Result | Where |
| --- | --- |
| Sink mass is positively rank-correlated with recall at 8 times the training length across 34 learned runs (0.49, 95% confidence interval 0.20 to 0.70) | `results/summary_sink_recall.json` |
| Attention-only stacks learn retrieval early (9 of 9 runs) and fail beyond twice the training length; hybrids with the global layer last hold retrieval to 16 times that length and learn late or never (13 of 20 runs) | `results/logs/stats_placement.log` |
| BKF (bound keys, global layer first) learns by step 300 on 11 of 11 seeds and keeps 98.0% exact-match recall at 64 times the training length (100.0% with the marker tokens excluded) | `results/summary_paper.json` |

Long-context models are now built around mechanisms that are said to remove
the attention sink: an output gate on softmax attention, a learned sink logit,
rectified softmax, and hybrid stacks of delta-rule linear attention with a few
global layers. Whether removing the sink also makes a model read its context
evenly is a separate question, and it is the one this code is built to test.

---

## What is measured

| Quantity | Definition |
| --- | --- |
| `sink_mass` | Mean attention on position 0 over softmax layers, heads and queries 1..T-1 (the first-token attention, F-Attn, of Qiu et al., 2025) |
| `sink_ratio` | `sink_mass` divided by the same quantity for uniform causal attention, mean over t of 1/(t+1). This reference is about ln(T)/T, not 1/T |
| `sink_rate` | Share of (layer, head) pairs whose mean attention on position 0 exceeds 0.3 (Gu et al., 2025) |
| `sink_noop`, `sink_copy`, `sink_answer` | Sink mass restricted to queries whose next token is unpredictable, copyable, or a retrieval answer |
| `act_max`, `act_ratio` | Largest hidden-state magnitude per layer and its ratio to the median (Sun et al., 2024) |
| `gate_*`, `virtual_sink` | Mean output-gate value and mass on the learned sink logit, where the layer has one |
| `recall`, `profile` | Retrieval accuracy overall and in ten depth bins, with Wilson intervals |
| `recency_gap`, `middle_dip` | Last-quarter minus first-quarter recall; outer quarters minus inner quarters |

---

## The controlled testbed

**Data** (`sinkprobe/data.py`). A haystack of filler segments, each either
fresh random tokens or a verbatim copy of an earlier segment, with key-value
pairs at segment boundaries and a block of queries at the end. Every position
is labelled by what it has to predict (no-op, copy, answer). Two knobs change
one thing each:

- `p_noop`, the share of segments that are fresh random tokens, sets how many
  query positions have nothing worth reading.
- `query_skew` tilts which pairs are asked during training toward the end or
  the start of the context. Evaluation always asks uniformly.

**Models** (`sinkprobe/model.py`, `sinkprobe/layers.py`). Eight-layer,
width-128 language models (about 1.9 million parameters) that differ in one mixing
rule per step:

| Variant | Change from the row above |
| --- | --- |
| `softmax` | causal softmax attention, rotary positions |
| `gate` | sigmoid output gate on every attention layer |
| `hybrid` | three gated delta-rule layers per gated attention layer |
| `hybrid_nope` | no positional encoding on the attention layers |
| `hybrid_attnres` | Block Attention Residuals over depth |

Side branches: `sinklogit` (learned per-head sink logit), `softpick`
(rectified softmax), `hybrid_nodelta` (delta correction removed).

Each mechanism is also added alone to the softmax baseline (`gate`,
`hybrid_nogate`, `nope`, `attnres`). With `softmax`, `gate`, `hybrid_nogate`
and `hybrid` this gives a 2x2 of output gate by delta-rule layers, so the
effect of one mechanism is never read off a comparison that changed two.

## Experiments

| Name | What varies | What it tests |
| --- | --- | --- |
| ladder | mechanisms added cumulatively | what the full recipe does to each diagnostic |
| factorial | one mechanism at a time on the same baseline | which mechanism moves which failure mode |
| noop | share of positions with nothing worth reading | whether the sink tracks no-op demand |
| skew | where retrieval is asked for during training | whether position bias tracks retrieval demand |
| warmup | with and without the copy-rich warm-up | that the training protocol, not luck, makes retrieval learnable |
| interventions | a bias on the logit of position 0 at evaluation | whether retrieval runs through the sink, in trained and released models |

## Proposed design: a bound-key global layer first

The diagnostics leave two things unsolved: retrieval that holds at every
depth and length, and training that succeeds on every seed. Global layers
without positions extrapolate only inside a hybrid, and the usual hybrid
layout (three delta-rule layers, then one global layer) learns retrieval late
or not at all.

`hybrid_bka_first` in `sinkprobe/model.py` is that hybrid with the global
layer first in each block (SLLLSLLL, where S is the global softmax layer and L
a linear delta-rule layer). In the global layer, keys and values are
bound to their preceding tokens by a residual causal convolution of width 4,
matching uses no positional rotation, and logits are multiplied by
max(1, log(visible keys) / log(training length)), which is exactly 1 during
training. It has the same parameter count as the usual layout. Controls:
`hybrid_bka` (the same layer placed last), `hybrid_nope_first` (placed first,
without bound keys) and `bka` (bound keys in an attention-only stack).

```bash
bash scripts/run_placement.sh 1    # bound-key layer first, seeds 0-3 (~45 min)
bash scripts/run_placement.sh 2    # no warm-up, and placement without bound keys (~45 min)
bash scripts/run_hybrid_bka.sh     # the same layer placed last, and baselines at 16x length
bash scripts/run_bka.sh            # bound keys in an attention-only stack
bash scripts/run_text.sh           # byte-level WikiText check of copying by depth and length
python scripts/report_bka.py       # table_bka.tex; depth-profile, reliability and discovery figures
python scripts/stats_placement.py  # the tests behind method notes, section 15
python scripts/report_text.py
python scripts/sink_vs_step.py     # sink mass against the learning step
python scripts/head_ablation.py --tag ablation_layers --glob \
    "results/runs/main/hybrid_bka_first__p0.5__g0__s*.pt" "results/runs/main/hybrid_bka__p0.5__g0__s*.pt" \
    "results/runs/main/hybrid_nope__p0.5__g0__s*.pt" "results/runs/main/hybrid_nope_first__p0.5__g0__s*.pt"
                                   # remove one global layer or one head at test time (runs on a CPU)
bash scripts/run_lr_sweep.sh       # both placements at learning rates 1e-3 and 1e-2, seeds 0-1 (~1.5 h)
python scripts/lr_sweep_summary.py # learning step and recall of that check
```

What the last three checks found, with the numbers the paper quotes, is in
section 25 of `results/logs/method_notes.md`: the lookup of the bound-key
layout lives in one global layer (removing it drops recall at 4,096 tokens from
99.0% to 0.2%, while the second global layer can go), sink mass is no
consistent guide to the learning step, and at learning rates 1e-3 and 1e-2 the
global-first layout still learned first on both seeds.

The reasoning between runs, including the predictions fixed before each one,
is logged in `results/logs/method_notes.md` (sections 11 to 17). Notes on the
DeepSeek-V4.1 and Artificial Intelligence (AI) Index 2026 reports are in `results/logs/deepseek_v41_facts.md` and
`results/logs/ai_index_2026_facts.md`.

`scripts/reanalyze_pilot_v1.py` re-reads the first pilot against the correct
null model for sink mass: its "30x an even split" becomes 7.2x, and its
answer-only control (1.11x uniform) turns out to measure attention that never
received a gradient.

The delta-rule layer is computed in exact chunkwise form. The test suite
checks the outputs and the memory against the token-by-token recurrence in
64-bit arithmetic at chunk sizes 1, 8, 16 and 64, within 1e-10.

**Training** (`sinkprobe/train.py`). Answer loss plus next-token loss on every
other position, both as means over their own positions. Runs start with 1,500
steps on copy-rich data (`p_noop = 0`), identical for every condition,
followed by 2,500 steps on the condition being tested; runs whose names end in
`__w0` skip the warm-up to test it. Without it, the first models trained on
noisy haystacks stayed at chance for more than 2,000 steps (see
`results/logs/learn_*.log`). Training uses bfloat16 (16-bit brain floating
point) autocast and a captured CUDA (Compute Unified Device Architecture)
graph step; eager and graph training give matching loss curves.

---

## Released checkpoints

`sinkprobe/hf_probe.py` registers an attention function with `transformers`
that computes the layer output with scaled dot-product attention (SDPA) and,
during prefill only, measures
attention on position 0 from the post-rotary queries and keys, with
subsampled queries and a chunked logsumexp so the full attention map is never
stored. Linear-attention layers are excluded and counted. Retrieval uses the
multi-key needle task of RULER over WikiText-103 text.

The pair used is Qwen3-0.6B-Base (softmax attention in all 28 layers) and
Qwen3.5-0.8B-Base (Gated DeltaNet and gated attention in a 3:1 pattern).

---

## Reproducing

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

python scripts/fetch_wikitext.py --crlf           # data/ for the text probes (same MD5 checksums as the original runs)
bash scripts/run_pipeline.sh                       # every step below, in order
```

Tables and figures from the saved evaluations (no GPU needed; run
`scripts/stats_placement.py` first on a fresh clone):

```bash
python scripts/paper_tables.py                     # LaTeX tables and results/summary_paper.json
python scripts/paper_figures.py                    # figures; the attention maps run the saved models on CPU
```

or step by step:

```bash
python tests/test_sinkprobe.py                     # delta rule, causality, task, metrics
python tests/test_interventions.py                 # exactness of the sink intervention
python tests/test_stats.py                         # statistics behind the placement study
python tests/test_retrieval_attention.py           # exact attention rows for the attention check

bash scripts/run_short.sh 1                        # short plan behind the current results (~1 h 45 min)
python scripts/sweep.py --workers 2 --cuda-graph   # full design, 96 runs (~15 h on a laptop graphics processing unit, GPU)
python scripts/intervene.py                        # sink intervention on the trained models

python -m sinkprobe.hf_probe --model Qwen/Qwen3-0.6B-Base --needles 16 \
    --lengths 2048 4096 8192 16384 30000 --trials 16 --out results/hf/Qwen3-0.6B-Base.json
python -m sinkprobe.hf_intervene --model Qwen/Qwen3-0.6B-Base --biases 0 off \
    --out results/interventions/hf_Qwen3-0.6B-Base.json   # 'off' removes attention to position 0
python scripts/hf_ppl_check.py --model Qwen/Qwen3-0.6B-Base --biases 0 off \
    --out results/interventions/ppl_Qwen3-0.6B-Base.json  # does that hurt language modelling?

python scripts/reanalyze_pilot_v1.py               # first pilot against the correct null model
python scripts/make_report.py                      # tables, figures, results/summary.{json,md}
python scripts/bench_step.py                       # step times for the compute appendix
python scripts/run_costmodel.py                    # cache arithmetic for the Kimi K3 layer mix

bash scripts/run_pending.sh                        # still to run: WikiText, data regime, attention, 64x, pairs, sink removal
python scripts/stats_placement.py                  # placement statistics and their LaTeX tables
python scripts/table_predictions.py                # predictions fixed before runs, scored from the results
python scripts/robustness_summary.py               # usable context at 80% recall; recall across the data change
python scripts/more_findings.py                    # discovery thresholds, sink timing, uniformity; longer, pairs, sink steps
python scripts/retrieval_attention.py --help       # where the global layers look, by depth and length
```

Every reported number is computed from the run files by the scripts above
(`results/summary*.json`). Uncertainty is across independently trained seeds
(mean and 95% Student t interval); differences use
Welch intervals; rank correlations carry bootstrap intervals.

A run counts as having learned retrieval when its recall at the training
length is at least 90%. Sink and position-bias comparisons use learned runs
only, since a model that never retrieves has no meaningful position profile,
and how often each variant learns is reported as a result of its own
(the learnability table written by `make_report.py`).

Hardware used: one NVIDIA RTX 3050 laptop GPU with 4 gigabytes (GB) of memory
and an AMD Ryzen 7 7435HS processor.

---

## Layout

```
sinkprobe/
  data.py        haystack generator with position labels and the two knobs
  layers.py      softmax variants, gated delta rule (chunked and recurrent)
  residuals.py   residual sum and Block Attention Residuals
  model.py       the variant ladder
  metrics.py     sink, activation and position-bias diagnostics
  train.py       one run, JSON out
  hf_probe.py    diagnostics on released checkpoints
  costmodel.py   cache growth for the Kimi K3 layer mix (93 layers: 69 Kimi Delta
                 Attention, KDA, and 24 multi-head latent attention, MLA)
scripts/         sweep, report, benchmarks
tests/           properties the results rest on
results/         run files, logs, summaries
legacy/          code of the first pilot, kept for provenance of results/pilot_v1
```

MIT licence.
