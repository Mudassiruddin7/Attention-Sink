# Method notes: decisions and the runs behind them

Every number below was read from a log or result file in this repository.
These notes are the source for the appendix on training and measurement
choices.

## 1. What was wrong with the first pilot (results/pilot_v1)

- The no-auxiliary-loss control left most query positions without any
  gradient, so their attention stayed near its initial, nearly uniform state.
  Uniform causal attention at T = 96 gives a sink mass of mean over t of
  1/(t+1) = 0.044; the control measured 0.049. The control therefore did not
  isolate the objective as the cause of the sink.
- The "even reader" reference was given as 1/T. The correct reference for
  the averaged first-token attention is mean over t of 1/(t+1), about
  ln(T)/T (0.0201 at T = 256), which is 4 to 5 times larger at the lengths used.
- The hybrid models scored 6.0 and 5.7 percent recall against 8.3 percent blind chance.
  Their linear layer had no delta rule and no short convolution, and its
  decay was floored at exp(-0.25) = 0.78 per step.
- The hybrid rows changed two things at once relative to the gated softmax
  row (linear layers and removal of RoPE).

## 2. Why the new task shares one vocabulary

Learnability probes, softmax, 8 layers, batch 32, 800 steps
(`results/logs/learn_*.log`), separate key, value and filler vocabularies:

| Probe | Copy-position CE at step 799 (chance ln 256 = 5.55) | Recall |
| --- | --- | --- |
| base, p_noop = 0.5 | 5.587 | 4.3% |
| lr 1e-2 | 5.592 | 3.5% |
| 2 layers | 5.592 | 2.7% |
| length 64 | 5.595 | 6.2% |
| filler vocabulary 32 | 3.503 (chance 3.47) | 7.4% |
| p_noop = 0 (copy-rich) | 0.008 | 2.3% |

With copy-rich data the copy circuit formed by step 400, yet retrieval stayed
at chance because keys and values lived in their own vocabulary. Keys and
values are now ordinary filler tokens, and the keys of a sequence are removed
from its filler and from its values, so every key occurs once before its
query and retrieval is the copy operation itself.

## 3. Why every run starts with a copy-rich warm-up

- No warm-up, p_noop = 0.5: copy CE stayed at 5.59 through step 2,000
  (`pilot_softmax.log`, `diag_softmax_noqk.log`). QK-norm on or off made no
  difference.
- 300-step warm-up: copy CE reached 5.08 and returned to about 5.4 once the noisy
  data began (`recipe_V2_softmax_warm_noqk.log`).
- 2048-token vocabulary, no warm-up: copy CE 7.62 (= ln 2048) at step 500
  (`recipe_W1_softmax_v2048_qk.log`), so spurious token matches are not the
  cause.
- 800-step warm-up, shared vocabulary: softmax reached 91% recall by step 600
  and 100% by step 1,600, and kept it on noisy data
  (`recipe_X1_softmax_warm800.log`). The hybrid began learning only after
  step 1,000 and reached 92.6% training recall by step 2,000
  (`recipe_X2_hybrid_warm800.log`).

Final recipe for every variant and condition: 4,000 steps, the first 1,500
on p_noop = 0 with the condition's query skew, then 2,500 on the condition.
Our reading is that irreducible loss on unpredictable positions dominates the
gradient early and delays the formation of induction circuits; the warm-up is
identical across conditions, so it cannot create a difference between them.

## 4. Numerical checks

- Chunked gated delta rule against the token-by-token recurrence, float64:
  max error 1e-15 (outputs and state) and 5e-14 (gradients) at chunk sizes
  1, 8, 32, 128, with and without the delta term.
- CUDA-graph training against eager training, 400 steps, same seed
  (`bench_step` session): hybrid loss 10.914 vs 10.914 at step 399, softmax
  10.322 vs 10.299; 3.09x faster for the hybrid (310.7 to 100.7 ms per step)
  and 1.42x for softmax (52.2 to 36.8 ms).

## 5. Released checkpoints on a 4 GB card

- This PyTorch build (2.11.0+cu128, Windows) has no flash attention. Its
  memory-efficient kernel does not accept grouped-query inputs, so the
  library's GQA path fell back to the math kernel and allocated a full
  16 x 8192 x 8192 map (4.00 GiB) at 8K tokens. Repeating key and value heads
  before calling SDPA keeps the memory-efficient kernel: an 8K causal pass
  takes 35 ms and 96 MiB.
- Generation with a key value cache needs about 0.94 GB at 8K and would need
  about 3.7 GB at 32K for Qwen3-0.6B. Scoring is therefore teacher-forced
  exact match, which gives the same verdict as greedy exact match in one
  forward pass without a cache. Peak memory is 2.07 GB at 29K tokens for
  Qwen3-0.6B and 3.25 GB at 16K for Qwen3.5-0.8B, whose reference Gated
  DeltaNet implementation runs out of memory at 30K.
- The question ends in "is:" to match the needle format "is: N"; with "is"
  the model's first predicted token was the colon and every trial scored 0.

## 6. Kimi K3 facts

See `kimi_k3_facts.md`: 93 layers (69 KDA + 24 Gated MLA), NoPE on MLA
layers, pre-training at 8K extended to 64K, and no sink or position-resolved
retrieval diagnostics anywhere in the 47-page report.

## 7. GPU budget actually used

The full 96-run design was stopped after 9 runs (seed 0 of the ladder, side
branches and the gate-free hybrid) because it would have taken about 15 hours
with two jobs sharing the laptop GPU (7 min per softmax run, 15 to 21 min per
hybrid or explicit-attention run). The short plan (`scripts/run_short.sh 1`)
added 8 runs with the same 4000-step recipe and a lighter evaluation (256
sequences at 256, 1024 and 2048 tokens instead of 512 sequences at four
lengths). Its training took 83 minutes, not the 50 estimated beforehand.

## 8. Which runs count as having learned

A run counts as learned when recall at the training length is at least 90%.
12 of 17 runs pass. Every softmax-layout run learned (softmax 2/2, gate 2/2,
NoPE 1/1, AttnRes 1/1, sink logit 1/1, softpick 1/1). The hybrids learned
about half the time within 4000 steps: gated hybrid 1/2 (the other reached
69%), gate-free hybrid 1/2, NoPE hybrid 1/2, full recipe with AttnRes 1/2,
hybrid without the delta rule 0/1. Sink and position-bias comparisons use
learned runs only; the success rate is reported as a result
(`paper/generated/table_learnability.tex`). The seed-0 impression that the
full recipe with AttnRes is uniquely robust did not replicate at seed 1; what
did replicate is that each NoPE hybrid that learned kept 87 to 95% recall at
8x the training length with a flat depth profile.

## 9. Interventions: attention pushed off position 0

- A bug first: `--biases 0 -inf` was rejected because argparse reads `-inf` as
  an option name. Both intervention scripts now take `off` for minus infinity.
- Controlled models that learned (paired over the same evaluation sequences):
  removing attention on position 0 changed recall by -2.3 +/- 2.0 points at
  the training length and -0.2 +/- 0.6 points at 4x, pooled over 10 models
  with a positional signal. The NoPE softmax model is the exception: recall
  fell from 100% to 0%, consistent with its first token carrying position.
- Qwen3-0.6B-Base, 16 needles, 44 paired prompts per length: removing the sink
  (0.47 -> 0.00) raised exact match by +9.1 points [+2.3, +18.2] at 2K and
  +13.6 points [+4.5, +25.0] at 4K; answer log probability rose by 0.03 and
  0.08 nats. The gains sit at the edges of the context.
- Language modelling check, 16 WikiText windows of 2048 tokens scored after
  token 64, paired: Qwen3-0.6B changed by +0.0000 +/- 0.0013 nats per token;
  Qwen3.5-0.8B changed by +0.0155 +/- 0.0027. A CPU diagnostic confirmed the
  intervention is active on that path (sink 0.53 -> 0.00, logits moved by up
  to 14.5). For Qwen3-0.6B the loss change is confined to the start of the
  context: +1.06 nats on tokens 1-3, +0.48 on tokens 4-15, +0.004 on 16-63 and
  +0.001 on 64-382 (4 windows of 384 tokens).
- Control, Qwen3.5-0.8B-Base (gated hybrid, sink already small), same prompts:
  removing its sink (0.039 -> 0.00 at 2K, 0.033 -> 0.00 at 4K) left exact match
  at 100.0% at 2K and moved it from 95.5% to 93.2% at 4K (-2.3 [-6.8, 0.0]);
  answer log probability changed by -0.008 +/- 0.014 and -0.066 +/- 0.044.
  The retrieval gain from removing the sink appears only in the dense model
  whose sink is large.

## 10. Sink at no-op queries

Across the 12 learned runs, first-token attention at no-op queries exceeds
that at copy queries in 11 (sign test p = 0.0063, Wilcoxon p = 0.001; mean
difference +0.040 +/- 0.021; median ratio 2.7).

## 11. Bound-Key Attention (proposed mechanism)

Motivation from sections 8 to 10: removing the sink does not change retrieval
in models with a positional signal; position-uniform retrieval at 8x length
appeared only in NoPE hybrids, and only when they trained (about half the
seeds). The mechanism targets the two things that failed: uniform
retrievability across depth and length, and reliable training.

Design, in `sinkprobe/layers.py` (`kv_conv`, `logn_ref`), variant `bka`:
- keys and values pass through a residual depthwise causal convolution of
  width 4, so a key carries its preceding tokens and one attention hop can
  find "the position after the queried content";
- no positional rotation in the match (content-only);
- logits of query t are multiplied by max(1, log(t + 1) / log(256)); this is
  exactly 1 during training and only acts when the context grows past the
  training length.

Ablations without retraining (`scripts/eval_ckpt.py`): the same BKA weights
with the length scaling off, and NoPE softmax weights with it on. Ablation
with retraining: `bka_rope` (rotary positions kept). Reliability: BKA and
softmax trained with and without the copy-rich warm-up.

Prior art that reviewers will compare against: KV-shifting attention and
Canon layers (convolutional mixing of keys and values), NoPE, and log-length
or scalable-softmax logit scaling. The claim is the link from the diagnosis
to the design and the evidence on uniformity and reliability, not any single
component.

Natural-text check (`sinkprobe/text_task.py`, `scripts/run_text.sh`):
byte-level models trained on WikiText-2 with no warm-up, scored on copying a
32-byte span repeated from a known depth of held-out WikiText-103 text, with a
control in which the repeated span is not in the context.

## 12. BKA pilot: the extrapolation prediction failed

Pilot, seed 0, 4000 steps (`results/runs/main/bka__p0.5__g0__s0*.json`):

| Run | Recall at 256 / 1024 / 2048 / 4096 | First step with train recall >= 90% |
| --- | --- | --- |
| BKA, copy-rich warm-up | 100 / 0 / 0 / 0 | 300 |
| BKA, no warm-up | 100 / 0 / 0 / 0 | 1300 |
| Softmax RoPE, seeds 0 and 1 | 100 / 47 / 7 and 11 / not run | 600 |
| Softmax NoPE, seed 0 | 100 / 6 / 0 / not run | 1900 |
| NoPE hybrid, seed 1 | 100 / 89 / 87 / not run | 1800 |
| NoPE hybrid with AttnRes, seed 0 | 100 / 99 / 95 / not run | 2700 |

Binding keys made the retrieval hop fast to learn, with and without the
warm-up, but the pure attention stack without positions collapsed to 0%
recall at every depth past the training length, as NoPE softmax does. The
prediction that BKA keeps a flat, high depth profile at 4x to 16x is
falsified for this design. Reading: in a pure attention stack without
positions the other circuits still depend on implicit, length-dependent
position; the NoPE hybrids extrapolate because their delta-rule layers carry
order through recurrence. The planned BKA seeds and the BKA-with-RoPE
ablation were cancelled after this result (two seed-1 runs were stopped
partway and left no result files).

Where the collapse starts (seed 0 checkpoint, 128 evaluation sequences,
`results/evals/pilot_*.json`): with length scaling, recall was 85.8% at 320
tokens (1.25x), 11.8% at 512 and 0.0% at 1024; without it, 79.7%, 0.0% and
0.0%. NoPE softmax on the same lengths kept 93.0% at 320 and 55.8% at 512. The
scaling helps a little but does not prevent the collapse, and bound keys in a
pure attention stack degrade sooner than NoPE alone.

Revised design, variant `hybrid_bka`: the NoPE hybrid (three gated
delta-rule layers per gated global layer) with bound keys and length
scaling in its two global layers, so that order comes from the recurrent
layers and the retrieval hop is easy to learn. Pilot: two seeds with the
standard recipe. The length-scaling ablation for this design needs no
retraining.

## 13. hybrid_bka pilot, and what the 87.5% ceiling is

Pilot, seeds 0 and 1, standard recipe (`results/runs/main/hybrid_bka__*`):

| Run | Recall at 256 / 1024 / 2048 / 4096 | Depth profile at 4096 | First step with train recall >= 90% |
| --- | --- | --- | --- |
| hybrid_bka, seed 0 | 100 / 87.5 / 87.5 / 87.5 | flat, 84 to 91 across bins | 1900 |
| hybrid_bka, seed 1 | 100 / 100 / 100 / 100 | flat, 100 in every bin | 1900 |
| NoPE hybrid, seed 0 | did not learn (0.9%) | | never |
| NoPE hybrid, seed 1 | 100 / 88.9 / 87.4 / not run | flat | 1800 |
| NoPE hybrid with AttnRes, seed 0 | 99.9 / 98.6 / 95.3 / not run | flat | 2700 |
| NoPE hybrid with AttnRes, seed 1 | did not learn (0.8%) | | never |

On the same two seeds the bound-key hybrid learned twice and each plain NoPE
hybrid once. Two seeds cannot establish a difference in reliability; the
confirmation run adds seeds 2 and 3, a run without warm-up and a third
plain NoPE hybrid seed. The bound keys did not make learning faster.

The 87.5% ceiling (48 sequences per length, CPU check of the checkpoints):
at 1024 tokens both hybrid_bka seed 0 and NoPE hybrid seed 1 answer every
query slot but the first at 100%. The first slot scores 0% and 10%, and every
error is the query marker token, not a wrong value. At 256 tokens all slots
are 100%. The first answer is where the query block starts, which training
always places at the same absolute offset; past the training length the
model repeats the marker instead of answering. Retrieval itself holds at
every depth. `metrics.evaluate` now records `acc_by_slot` and
`recall_after_first`, and both should be reported.

## 14. Confirmation of hybrid_bka, and the verdict on the mechanism

Standard recipe unless noted; recall in percent at 256 / 1024 / 2048 / 4096.

| Run | Learned | Recall | First step with train recall >= 90% |
| --- | --- | --- | --- |
| hybrid_bka seed 0 | yes | 100 / 87.5 / 87.5 / 87.5 (first query slot fails past 256) | 1900 |
| hybrid_bka seed 1 | yes | 100 / 100 / 100 / 100 | 1900 |
| hybrid_bka seed 2 | yes | 99.9 / 87.1 / 87.0 / 87.0, flat depth profile | 3200 |
| hybrid_bka seed 3 | no | 54.4 at 256 with a recency-shaped profile, near 0 beyond | never |
| hybrid_bka seed 0, no warm-up | no | 0.4 / 0.3 / 0.5 / 0.6 | never |
| NoPE hybrid seed 2 | yes | 100 / 97.5 / 92.6 / 78.5, flat depth profile | 2700 |

At 4096 tokens (16x), evaluation without retraining (`results/evals/`):
- length scaling off vs on for hybrid_bka: seed 0 87.4 vs 87.5, seed 1 99.9 vs
  100, seed 2 47.7 vs 87.0;
- baselines: NoPE hybrid seeds 1 and 2 86.8 and 78.4 (seed 0 did not learn),
  NoPE hybrid with AttnRes seed 0 88.2 (seed 1 did not learn), softmax RoPE
  seeds 0 and 1 0.6 and 0.7, NoPE softmax 0.0.

Verdict:
- Reliability: bound keys learned on 3 of 4 seeds, plain NoPE hybrids on 3 of
  5 (Fisher exact p = 1.0). Bound keys did not speed learning and failed
  without the warm-up. The reliability target is not met.
- Position-uniform retrieval far past the training length is a property of
  the NoPE hybrid family rather than of bound keys: every NoPE hybrid that
  learned (6 of 6, three variants) kept 78 to 100% recall at 16x with no
  depth trend (every depth bin within 10 points of the run mean), against 0
  to 0.7% for the attention-only models (Mann-Whitney exact p = 0.004). Among
  learned NoPE hybrids, bound keys did not change recall at 16x (91.5 vs
  84.5, p = 0.4).
- Length scaling helped one of three learned seeds by 39 points and was
  neutral for the other two.

Bound keys alone are therefore a negative result: in a pure attention stack
they learn quickly but fail past the training length, and in the usual
hybrid layout they add nothing measurable. What the runs support at this
point is that uniform retrieval at long length comes from content-only global
attention over a length-invariant order carrier, and that the open problem of
that family is training reliability (section 15).

## 15. Reliability depends on where the global layer sits

Diagnosis, from the runs with p_noop 0.5, no skew and the copy-rich warm-up
(statistics from `scripts/stats_placement.py`):

| Layout | Learned | Steps to 90% training recall |
| --- | --- | --- |
| Attention only, 8 global layers (RoPE, NoPE, output gate, sink logit, softpick, AttnRes, BKA) | 9 of 9 | 300, 500, 600, 600, 600, 600, 700, 700, 1900 |
| Gated delta-rule hybrids, three recurrent layers then one global layer (LLLS) | 8 of 13 | 1800, 1800, 1900, 1900, 2700, 2700, 3200, 3600; 5 not within 4000 |

- At step 1400, 12 of the 13 LLLS runs had at most 14% training recall (the
  other had 50%). At step 1500, answer loss was still 5.41 to 5.56 nats in 11
  of them, about where training started (ln 258 = 5.55); the other two were
  at 5.02 and 4.11. Every attention-only run except NoPE was at or below 0.19
  nats.
- Discovery time differs between the layouts (log-rank p = 8e-6; exact
  permutation test of the rank sum p = 3e-5). The share that learned within
  4000 steps differs less clearly (Fisher p = 0.054).
- Two kinds of failure. Three runs never moved off chance on the answers
  (answer loss 5.51 to 5.55 nats at step 3900). Two were still improving when
  training stopped: the gated RoPE hybrid, seed 1, at 68% training recall, and
  the bound-key hybrid, seed 3, at 48%, whose recall rose from 27% for the
  earliest pairs to 93% for the latest.
- Bound keys shorten discovery when the global layer reads token embeddings
  but not after three recurrent layers: attention-only NoPE 1900 steps, BKA
  300; LLLS NoPE hybrids 1800, 2700 and not within 4000, LLLS with bound keys
  1900, 1900, 3200 and not within 4000.

Hypothesis: in LLLS the global layer matches on features that three
delta-rule layers have already mixed, so the retrieval hop has to be learned
together with them. A bound-key global layer placed first reads token
embeddings, where "the position after the queried key" is a single bilinear
match, as in BKA.

Test: `hybrid_bka_first` (layers SLLLSLLL, same parameter count as
`hybrid_bka`), `scripts/run_placement.sh 1`, seeds 0 to 3, standard recipe.
Causality and the agreement of the fast and explicit attention paths (to
2.5e-6) were checked on CPU first. Predictions, written into the script
before the runs: (1) every seed passes 90% training recall by step 1500;
(2) recall after the first query slot at 4096 tokens is at least 85% for
every learned seed.

| Seed | Steps to 90% (LLLS with bound keys, same seed) | Recall at 256 / 1024 / 2048 / 4096 | After first slot, 4096 | 4096, length scaling off |
| --- | --- | --- | --- | --- |
| 0 | 300 (1900) | 100 / 100 / 100 / 100 | 100 | 31.3 |
| 1 | 300 (1900) | 100 / 100 / 99.9 / 97.4 | 97.0 | 9.0 |
| 2 | 300 (3200) | 100 / 100 / 100 / 100 | 100 | 75.0 |
| 3 | 300 (not within 4000) | 100 / 100 / 99.9 / 99.7 | 100 | 60.5 |

Both predictions held.
- Discovery was faster than in the LLLS hybrids (log-rank p = 6e-5, rank sum
  p = 4e-4) and matched the fastest attention-only run (BKA, 300 steps).
  Paired by seed with the LLLS bound-key hybrid it was faster 4 times out of
  4; four pairs cannot give a sign-test p below 0.125.
- Learned 4 of 4 (Clopper-Pearson 95% interval 40 to 100%) against 8 of 13
  (32 to 86%), Fisher p = 0.26: a higher success rate is not established.
- At 16x the training length, among learned runs: recall 99.3% against 88.0%
  for the learned LLLS NoPE hybrids (exact permutation p = 0.043), and 99.2%
  against 94.3% after the first query slot (p = 0.08). Every depth bin at
  4096 lies between 95 and 100%. The first-slot failure seen in LLLS is
  absent (first slot 98 to 100%).
- Length scaling is required in this design: switched off at evaluation,
  recall at 4096 falls from 99.3% to 44.0% on average (by 25 to 88 points per
  seed). In LLLS it mattered for one seed of three.
- Sink mass at 256 tokens is 0.087 (2.5 to 5.7 times the uniform reference),
  the largest of the designs in `table_bka.tex`, while retrieval is the most
  uniform.
- Training took 1146 to 1220 s per run against 1133 to 1150 s for LLLS with
  bound keys (two runs sharing the GPU in both cases).

Stage 2 (`scripts/run_placement.sh 2`): seeds 0 and 1 without the warm-up,
seeds 0 and 1 of the global layer first without bound keys
(`hybrid_nope_first`), and the LLLS NoPE hybrids evaluated with the same
length scaling (no retraining).

| Run | Steps to 90% | Recall at 256 / 4096 | After first slot, 4096 |
| --- | --- | --- | --- |
| Bound-key layer first, no warm-up, seed 0 | 1100 | 100 / 99.5 | 99.4 |
| Bound-key layer first, no warm-up, seed 1 | 1200 | 100 / 100 | 100 |
| Global layer first without bound keys, seed 0 | not within 4000 | 74.0 / 29.2 | 29.0 |
| Global layer first without bound keys, seed 1 | not within 4000 | 55.0 / 0.1 | 0.0 |

- Without the warm-up the bound-key layer first learned on both seeds, with
  flat depth profiles at 4096 (98 to 100% in every bin). The same layer placed
  last did not learn without the warm-up (one seed, 0.4%); bound keys in an
  attention-only stack did (step 1300) but scored 0% past the training length.
  Two runs against one give no test (Fisher p = 0.33).
- Placement alone is not enough. Without bound keys the global-first hybrid
  learned copying during the warm-up (training recall 69% and 68% at step
  1200, copy loss 0.23 and 0.58) and lost it when the data changed at step
  1500 (copy loss 3.92 and 3.52, training recall 17% and 11%), then recovered
  only partly (72% and 52% at step 3900). At 4096 tokens seed 0 kept a recency
  profile (under 10% for pairs in the first half of the context, 91% for the
  latest) and seed 1 fell to 0.1%. Against the bound-key version: 0 of 2
  against 4 of 4 learned (Fisher p = 0.067), discovery log-rank p = 0.025.
- Length scaling also helps the usual layout at 16x: the learned LLLS NoPE
  hybrids move from 80.0% to 92.4% on average (NoPE hybrid seed 2 from 78.5%
  to 99.3%).

Correction to the stage 1 comparison at 16x. With the length scaling on for
every run, recall is 99.3% for the bound-key layer first against 92.4% for the
learned LLLS NoPE hybrids (p = 0.13), and 99.2% against 99.8% after the first
query slot (p = 0.41). The earlier 99.3% against 88.0% (p = 0.043) compared
designs with and without the scaling. Once they have learned, both layouts
retrieve equally well at 16x apart from the first query slot; what the
placement changes is whether and when retrieval is learned, and the
first-slot failure.

What the evidence supports (4 seeds with warm-up and 2 without, 1.9M
parameters, synthetic task):
1. With the global layer last, discovery is late and depends on the seed;
   with a bound-key global layer first it is early on every seed tested
   (log-rank p = 6e-5 against 13 LLLS runs), including without the warm-up.
2. Both parts are needed: bound keys with the global layer last, and the
   global layer first without bound keys, each fail on some seeds.
3. Length scaling is required at 16x in this design (99.3% against 44.0%
   without it) and helps the usual layout as well.

Not supported: higher long-context accuracy than the usual layout once that
layout has learned and has the same scaling; anything at larger scale or on
natural text.

## 16. Claims ledger

Framing for the paper: the main contribution is the phenomenon, a
training-versus-extrapolation trade-off in long-context retrieval that sink
diagnostics do not capture. The bound-key global layer first, with length
scaling, is a targeted fix for that trade-off, tested in the synthetic
setting only. Updated after stage 3 and the WikiText check.

| Claim | Evidence | Status |
| --- | --- | --- |
| Sink mass and position bias are distinct | corrected null model; rank correlations across runs include 0 (`paper/figures/dissociation.pdf`) | supported |
| Sink size does not predict retrieval far past the training length | `paper/figures/sink_vs_long_recall.pdf`, 25 learned runs: rank correlation between sink mass at 256 tokens and recall at 2048 of 0.32 [-0.08, 0.62]. Recall is high at sink 0.016 to 0.029 (NoPE hybrids) and at 0.024 to 0.122 (bound keys, either placement); the lowest sink (0.009, gated RoPE hybrid) keeps 17.1%, the gate-free RoPE hybrid at 0.085 keeps 19.9%, and attention-only models at 0.035 to 0.071 keep 0 to 11.6% | supported as "no reliable relation, and larger sinks do not go with worse recall in these runs" |
| Sink mass concentrates at no-op queries | 24 of 25 learned runs, including the placement designs, sign test p = 1.5e-6 (sections 10 and 18) | supported |
| Removing the sink rarely changes retrieval in controlled models | -2.3 +/- 2.0 and -0.2 +/- 0.6 points; NoPE softmax is the exception (section 9) | supported |
| Removing the sink never helps retrieval | contradicted by Qwen3-0.6B, +9.1 and +13.6 points (section 9) | not claimed; no title saying sinks do not matter |
| Attention-only models learn early and fail past the training length; NoPE LLLS hybrids hold at 16x but learn late or not at all | discovery log-rank p = 4e-6 (9 against 15 runs); 0 to 0.7% against 78 to 100% at 16x | supported |
| The bound-key layer first learns early on every seed tested | 8 of 8: 6 with warm-up, all by step 300; 2 without, by 1100 and 1200. Log-rank p = 8e-6 against 15 LLLS runs; faster than the same layer placed last on 6 of 6 matched seeds (sign test p = 0.031) | supported, "every seed tested" |
| A higher success rate | 9/9 against 11/18, Fisher p = 0.059 (section 23) | not supported at the 5% level |
| Higher accuracy at 16x than the usual layout | 98.7 against 91.6% (p = 0.10), and 98.9 against 99.8% after the first slot (p = 0.78), same scaling for both | not supported |
| Both components were required | global layer first without bound keys: 1 of 4 learned (Fisher p = 0.033 against 6 of 6), all 4 lost copying when the warm-up ended, and none kept uniform recall at 16x (0.1 to 64.4%, rising with depth). Bound keys placed last: 4 of 6 learned, at step 1100 to 3200 | supported for early learning and uniform recall at 16x, "in the tested runs" |
| Bound keys keep what the first global layer learned when the data change | training recall across the change at step 1500: 100 to 100 on 6 of 6 seeds with bound keys; 70 points lost on average on 4 of 4 without (exact permutation p = 0.005) | supported, "in the tested runs" |
| Usable context at 80% recall | attention only 1 to 2x, RoPE hybrids 2x, NoPE hybrids with the global layer last 8x to at least 16x, bound-key layer first at least 16x on 6 of 6 (coarse length grid) | supported as a description |
| Learning speed does not depend on the threshold for learned | log-rank p = 8e-6 at 50, 80, 90 and 95% training recall (section 18) | supported |
| Recall at 16x is high at every depth for the bound-key layer first | lowest 95% lower bound over the depth bins 90.2 to 98.0% on 6 of 6 seeds (section 18) | supported |
| Sink size is not set by retrieval | hybrids that never learned end with sinks of 0.016 to 0.248, learned ones 0.009 to 0.122; no consistent change at discovery, probes every 500 steps (section 18) | supported as a description |
| Retrieval holds at 64x the training length | markers excluded, 100.0% at 8192 and at 16384 tokens on 6 of 6 seeds (exact match 98.8 and 99.1%); the same layer placed last reaches 91.4 and 92.2%, and the NoPE hybrid falls to 75.7 and 54.6% (section 21) | supported |
| Retrieval survives four times as many distractors | 16 to 64 key-value pairs at 256 tokens: 0.0 points lost on 6 of 6 seeds, against 6.4 to 38.3 points for the global layer first without bound keys (section 19) | supported |
| The first global layer reads the value only when its keys are bound | at 4096 tokens one head of layer 0 puts 0.57 to 0.69 on the value with bound keys, no depth bin below half its mean; without them no head of layer 0 exceeds 0.004 (section 19) | supported |
| The sink is not used for retrieval in the new design | with the markers excluded, recall at 2048 tokens is 100.0% on all six seeds with position 0 kept and with it removed; seed 1's 28.4-point loss is entirely marker emission, its marker rate going 0.2 to 28.6% (section 21) | supported |
| Both failures of section 19 are the query marker, not retrieval | with the two reserved tokens excluded from the answer, seed 5 on copy-rich data at 4096 goes 25.0 to 98.4%, seed 4 85.9 to 98.4%, and seed 1 with position 0 removed at 2048 goes 70.3 to 100.0% (section 20) | supported |
| The first-slot failure is a placement effect | at 1024 tokens the first query slot scores 0% with the global layer placed last (a marker on 100% of queries, 0 to 6% when markers are excluded) and 100% with it placed first; both are 100% at the training length; slots 1 to 7 are 100% in both (section 22) | supported |
| Length scaling is part of the design | 99.3 to 44.0% at 16x without it (seeds 0 to 3) | stated in the method, not in a footnote |
| No language-modelling cost | WikiText, seed 0: +0.009 bits per byte at 512 bytes against the same layer placed last (+0.013 and +0.009 at 1024 and 2048) | supported as a sanity check (one seed, 1.9M parameters) |
| A copying advantage on natural text | gain from the context 14.7 against 16.3 points at 2048 bytes, within 1.6 points at every length (seed 0) | not supported |
| Larger models, 1M tokens, natural text beyond this sanity check | not tested | out of scope; 1M only as motivation |

Wording rules: "on every seed tested", never "reliably", until a rate test
supports it. The 99.3 against 88.0% comparison stays in this log (section 15)
and out of the paper. No result is claimed beyond 16x the training length.
Discovery counts only runs that are learned at the end; this was changed after
stage 3, when a run passed 90% on warm-up batches and then lost it, and no
conclusion changed with it.

### Predictions for stage 3 and the WikiText check

Written at 2026-09-15 19:49, after stage 3 had started. The only stage-3 output seen
before writing was step 200 of its first pair.

Stage 3 (`scripts/run_placement.sh 3`):
1. `hybrid_bka_first` seeds 4 and 5 pass 90% training recall by step 1500
   and keep at least 85% recall after the first query slot at 4096 tokens.
2. `hybrid_bka` seeds 4 and 5, trained side by side with them, do not pass
   90% before step 1500 (no LLLS run has so far).
3. `hybrid_nope_first` seeds 2 and 3 do not reach 90% within 4000 steps.

WikiText check (`scripts/run_text.sh`, the bound-key layer first against the
same layer placed last, seeds 0 and 1, trained side by side):
4. No language-modelling cost: held-out loss at 512 bytes differs by less
   than 0.05 bits per byte on each seed.
5. Copy accuracy minus the control at 2048 bytes (4x) is at least that of the
   same layer placed last on each seed.

A failed prediction is reported as a failure, with the numbers.

### Stage 3 outcome

| Run | Steps to 90% | Recall at 256 / 4096 | After first slot, 4096 |
| --- | --- | --- | --- |
| Bound-key layer first, seed 4 | 300 | 100 / 98.2 | 100 |
| Bound-key layer first, seed 5 | 300 | 100 / 97.0 | 96.6 |
| Same layer placed last, seed 4 | 1100 | 100 / 87.5 | 100 |
| Same layer placed last, seed 5 | not learned | 18.3 / 1.4 | 1.6 |
| Global layer first without bound keys, seed 2 | 2600 | 98.4 / 41.4 | 43.2 |
| Global layer first without bound keys, seed 3 | not learned (90% on warm-up batches at step 1100, 25% at step 1500) | 82.8 / 64.4 | 64.1 |

1. Held. Both seeds passed 90% at step 300 and kept 100% and 96.6% after the
   first query slot at 4096.
2. Failed for seed 4: the same layer placed last passed 90% at step 1100, the
   first LLLS run to do so before step 1500. Seed 5 did not learn.
3. Failed as worded. Seed 2 learned (98.4% at 256, 90% at step 2600) and seed
   3 passed 90% on warm-up batches at step 1100. Both lost copying when the
   warm-up ended (training recall 80% to 15% and 90% to 25% between steps 1200
   and 1500), as seeds 0 and 1 had. Neither kept uniform recall at 16x: 41.4%
   (9 to 12% in bins 2 to 4, 88% in the last) and 64.4% (7% in the first bin,
   98% in the last).

What stage 3 changes: the matched comparison reaches significance (6 of 6
seeds faster, p = 0.031); the global layer first without bound keys can learn
the task at the training length but not uniform retrieval at 16x; the usual
layout can occasionally learn early (seed 4).

### WikiText check, interrupted

Two byte-level runs at 512 tokens did not fit together on the 4 GB card (3.86
of 4.09 GB in use, no progress past step 0 in five minutes); they were stopped
and restarted one at a time (2.9 GB, about 30 s per 100 steps). The restarted
run was stopped again at step 1800 of 3000 to free the laptop, before any
evaluation, so no WikiText result exists yet. Decision recorded before any
result: the check runs seed 0 only, as a sanity check of language-modelling
cost, so predictions 4 and 5 will be scored on one seed.

## 17. What the DeepSeek-V4.1 and AI Index 2026 reports add, and three readings of the saved runs

Report notes: `results/logs/deepseek_v41_facts.md` and
`results/logs/ai_index_2026_facts.md`.

- DeepSeek-V4.1-Flash (1M-token context) reads compressed global keys that
  summarize adjacent tokens, pairs global attention with a 128-token window in
  every layer from the third, and trains sparse attention from scratch at 64K
  without a dense warm-up. The report has no sink or position-resolved
  retrieval analysis, names sparse retrieval over long contexts as an untested
  boundary, and its base model scores 45.2 on LongBench-V2, 6.3 points below
  V4-Pro.
- The AI Index 2026 reports context windows growing about 30x per year since
  mid-2023, stresses the gap between accepted and usable context, and asks for
  evaluations that isolate long-context ability.

### Usable context

`scripts/robustness_summary.py`, after the AI Index's 80%-accuracy length: the
longest evaluated length with recall of at least 80% (and at least 80% at every
shorter evaluated length), as a multiple of the training length, for learned
runs with the default warm-up.

| Design | Usable context |
| --- | --- |
| Attention only (softmax RoPE, output gate, sink logit, softpick, AttnRes, NoPE, BKA) | 1 to 2x |
| Hybrids with RoPE (gated and gate-free) | 2x |
| NoPE hybrids with the global layer last | 8x to at least 16x (at least 16x in 6 of 7) |
| Global layer first without bound keys (1 learned run) | 1x |
| Bound-key layer first | at least 16x on 6 of 6 seeds |

Runs were not all evaluated at the same lengths (some only at 256, 1024 and
2048, some also at 320 and 512), so the attention-only multiples are coarse.

### Recall across the change of data

Same script. Training recall at step 1400 (copy-rich data) against the lowest
between steps 1500 and 1800 (condition data), for runs at 50% or more by step
1400:

- bound-key layer first: 100 to 100 on 6 of 6 seeds;
- global layer first without bound keys: 84 to 17, 80 to 11, 88 to 15 and 95
  to 25, a mean drop of 70 points (against the bound-key version, exact
  permutation p = 0.005);
- attention-only models: 94 to 100 before, 96 to 100 at the lowest after (no
  drop); gate-free RoPE hybrid 50 to 14; LLLS bound-key hybrid seed 4, 100 to 99.

Reading: without bound keys, what the first global layer learned on copy-rich
data does not survive the change of data; with them nothing is lost. This
supports the refined hypothesis that bound keys stabilize retrieval across the
change, in the tested runs; it does not show why.

### Attention check, preliminary

`scripts/retrieval_attention.py` on CPU: seed 0 of three designs, 8 sequences,
256 and 1024 tokens. A check of the code, not a result.

- LLLS bound-key hybrid: three heads of its first global layer put 0.96 to
  0.997 of their attention on the value at both lengths, flat across depth, yet
  recall at 1024 is 87.5%. The first-slot failure is not a failure of retrieval
  attention.
- Bound-key layer first: the heads of layer 0 put 0.27 to 0.69 on the value
  (the most attended position for 56 to 100% of queries, by head), flat across
  depth and length; recall 100%.
- Global layer first without bound keys: no head puts more than 0.005 on the
  value, while one head of the second global layer puts 0.55 (256 tokens) and
  0.60 (1024) on the queried key itself. The model finds the key but does not
  read the value after it.

### Predictions for the pending steps

`scripts/run_pending.sh`, written before its regime and attention steps were
run; seed 0 of the attention check had been seen as above.

6. Data regime: every learned bound-key-layer-first model scores within 2
   points of its standard-data recall on copy-rich data (p_noop 0), at 256 and
   at 4096 tokens.
7. Data regime: every global-layer-first model without bound keys scores
   higher at 256 tokens on copy-rich data than on the standard data.
8. Attention: in every learned bound-key-layer-first run, the head that reads
   the value best at 256 tokens still puts at least 0.25 of its attention on it
   at 4096, with no depth bin below half of that head's mean.
9. Attention: in all four global-layer-first runs without bound keys, no head
   of layer 0 puts more than 0.05 of its attention on the value, at 256 or 4096
   tokens.

## 18. More findings from the saved runs, and three more pending steps

`scripts/more_findings.py`; the parts below need no GPU.

- Learning speed does not depend on where learned is drawn. Counting discovery
  at 50, 80, 90 or 95% training recall, all six bound-key-layer-first seeds reach
  the level by step 300, the LLLS hybrids reach it at steps 1000 to 3700 or not
  at all, and the log-rank p is 8e-6 at every threshold (6 against 15 runs).
- Sink size is not set by retrieval. Hybrids that never learned end with sink
  mass from 0.016 to 0.248; learned hybrids from 0.009 to 0.122. From the probe
  before to the probe after an LLLS hybrid learned (probes every 500 steps), six
  of nine changes are about 0.005 or smaller; the others are falls of 0.029 and
  0.041 and a rise of 0.017. In the bound-key layer first the sink keeps growing
  after retrieval is learned: 0.014 to 0.058 at step 500, 0.050 to 0.114 at step
  4000.
- The no-op finding of section 10 extends to the new designs: sink mass at no-op
  queries exceeds that at copy queries in 24 of 25 learned runs (sign test
  p = 1.5e-6); the exception is the output-gate model, seed 1.
- Recall at 16x is high at every depth only for the bound-key layer first. The
  lowest 95% lower bound over the ten depth bins at 4096 tokens is 90.2 to 98.0%
  on its six seeds, against 74.9 to 97.3% for the same layer placed last (whose
  recall includes the failing first query slot), 67.0 to 78.6% for the NoPE
  hybrids as trained (79.7 and 95.6% with the length scaling), and 5.8% for the
  one learned run of the global layer first without bound keys.

Three evaluation-only steps were added to `scripts/run_pending.sh`, on the saved
models:

- longer: learned models at 8192 and 16384 tokens (32x and 64x), as trained and,
  for the NoPE hybrid, with the length scaling on; the attention check at 16384
  tokens for both bound-key placements;
- pairs: 16, 32 and 64 key-value pairs at 256 and 2048 tokens, which should
  barely affect retrieval by attention and should hurt retrieval from a
  fixed-size recurrent state;
- sink: attention to position 0 removed in every learned placement model, at 256
  and 2048 tokens.

`scripts/eval_ckpt.py` gained `--collect-seqs 0` (the attention diagnostics hold
full attention maps, which do not fit at 16K tokens on a 4 GB card),
`--learned-only` and `--batch`; `scripts/intervene.py` gained `--device`.

Predictions for these steps, written before they were run. CPU checks of the
code had already shown, at 256 tokens with 8 sequences, 100% recall for all six
bound-key-layer-first seeds with position 0 kept and with it removed, and 100%
for seed 0 with 64 pairs. Those checks are too small to score anything, but they
were seen before predictions 11 and 12 were written.

10. Longer: every learned bound-key-layer-first model keeps at least 90% recall
    after the first query slot at 8192 tokens and at least 80% at 16384.
11. Pairs: from 16 to 64 pairs at 256 tokens, every learned bound-key-layer-first
    model loses at most 5 points, and the global-layer-first models without
    bound keys lose at least 10 points more on average.
12. Sink: removing attention to position 0 changes recall of every learned
    bound-key-layer-first model by less than 5 points, at 256 and at 2048 tokens.

### WikiText check outcome

`scripts/run_text.sh`, seed 0 of each design: 3000 steps on WikiText-2 bytes
without warm-up, scored on held-out WikiText-103 passages (predictions 4 and 5,
section 16). The control repeats a span that is not in the context.

| Design | Loss at 512 / 1024 / 2048 bytes (bits per byte) | Copy minus control at 512 / 1024 / 2048 (points) | Copy accuracy across depth |
| --- | --- | --- | --- |
| Bound-key layer first | 1.740 / 1.653 / 1.603 | +18.6 / +17.2 / +14.7 | 72 to 85% |
| Same layer placed last | 1.731 / 1.640 / 1.594 | +19.7 / +16.6 / +16.3 | 74 to 83% |

4. Held: the loss differs by +0.009 bits per byte at 512 bytes (+0.013 and
   +0.009 at 1024 and 2048).
5. Failed: at 2048 bytes the bound-key layer first gains 14.7 points from the
   context against 16.3 for the same layer placed last.

Reading, from one seed per design: on natural text both placements learn to
copy from the context without a warm-up, keep most of that gain at 4x the
training length with no depth trend, and model language about equally well.
The placement's advantage on the synthetic task, early discovery on every
seed, does not show here, where both designs learned. The check finds no
language-modelling cost and no copying advantage.

## 19. Outcome of the pending steps

`bash scripts/run_pending.sh`, every step exit 0, 00:29 to 01:52. Predictions
6 to 12 of sections 17 and 18, scored by `scripts/table_predictions.py`.

6. Failed for seeds 0, 1, 4 and 5. On copy-rich data (p_noop 0) the bound-key
   layer first is unchanged at 256 tokens (+0.0 on every seed) but lower at
   4096: -3.9, -4.1, -0.7, +0.2, -12.1 and -63.9 points. Retrieval at the
   training length does not depend on the data regime; retrieval far past it
   does, and on one seed heavily.
7. Held. Every global-layer-first model without bound keys scores higher at 256
   tokens on copy-rich data: 74.0 to 98.1, 55.0 to 90.0, 98.4 to 99.9 and 82.8
   to 99.2%. What they learned during the warm-up is still there; the noisy
   data is what they cannot handle.
8. Held. At 4096 tokens the head that read the value best at 256 still puts
   0.57 to 0.69 of its attention on it, with no depth bin below half of that
   head's mean, on all six seeds; it is a head of layer 0 in every run.
9. Held. Without bound keys no head of layer 0 exceeds 0.004 on the value, at
   either length, in all four runs.
10. Held. After the first query slot, recall is 97.3 to 100% at 8192 tokens
    (32x) and 98.5 to 100% at 16384 (64x) on all six seeds.
11. Held. From 16 to 64 key-value pairs at 256 tokens the bound-key layer first
    loses 0.0 points on every seed, while the global layer first without bound
    keys loses 38.3, 29.1, 6.4 and 23.8 points, a difference of 24.4 points in
    the mean.
12. Failed for seed 1. Removing attention to position 0 moves recall by at most
    0.6 points at 256 tokens on every seed, and at 2048 tokens by -0.6, +0.0,
    +0.0, +1.1 and +2.0 points on five seeds but -28.4 on seed 1.

Readings. Predictions 8, 9 and 11 together say what bound keys do: they let the
first global layer match the position after the queried key, that match is what
carries retrieval (no head finds the value without them), and it is content
matching rather than memory, since four times as many pairs cost nothing.
Prediction 10 extends position-uniform retrieval from 16x to 64x. Predictions 6
and 12 are the honest limits: past the training length the design is sensitive
to the data regime on some seeds, and one seed of six does lean on position 0
at 2048 tokens, so the sink claim holds design-wide but not run by run.

## 20. Debugging the two failures of section 19

Both failures are the readout artefact of section 13, not retrieval.

Evidence, in order.

1. The retrieval head is unaffected by the data regime. On copy-rich data at
   4096 tokens the layer-0 head of each bound-key-layer-first seed puts *more*
   attention on the value than on the standard data (seed 5: 0.634 to 0.678;
   seed 0: 0.606 to 0.642), with the value the most attended position on 100%
   of queries, while recall falls (seed 5: 96.1 to 19.1%).
2. The errors follow how often the value token repeats. On copy-rich data at
   4096, seed 5 answers correctly when the value occurs about once in the
   context (mean 1.2 occurrences) and fails when it recurs (mean 44.3), and its
   wrong answers are tokens that appear nowhere in the context.
3. Those wrong answers are the reserved tokens. At 1024 and 4096 tokens, 100%
   of seed 5's wrong answers are the sequence marker or the query marker, and
   the gold value is ranked second (median rank 1).
4. Excluding the two reserved tokens from the answer restores retrieval:

   | Case | Recall | Reserved tokens excluded |
   | --- | --- | --- |
   | Seed 5, copy-rich, 4096 | 25.0% | 98.4% |
   | Seed 4, copy-rich, 4096 | 85.9% | 98.4% |
   | Seed 0, copy-rich, 4096 | 95.3% | 98.4% |
   | Seed 1, position 0 removed, 2048 | 70.3% | 100.0% |

   The same holds at 1024 tokens, where every copy-rich seed returns to 100%.

Why copy-rich data makes it worse: with p_noop 0 every filler segment is a copy,
so the marker-versus-answer decision at the query block is made under statistics
the model never saw at that length, and the marker wins more often. Removing
position 0 does the same thing to seed 1 by raising attention entropy (0.343 to
0.469 at 2048) without moving the retrieval head.

Resolution. `sinkprobe/metrics.py` now also reports `recall_no_marker`, the
answer accuracy with the tokens that can never be an answer excluded; the tasks
name them (`reserved_tokens`, the sequence and query markers for the haystack,
the separator and sequence marker for the text probe) and both evaluation
scripts keep the field. Plain recall stays the primary number, so nothing in
sections 1 to 19 changes; the new field separates a readout artefact of the task
from retrieval. Predictions 6 and 12 stay failed as they were written.

## 21. One evaluation pass, three answer metrics

Every placement model was re-evaluated in a single pass (`rescore` in
`scripts/run_pending.sh`), so no table mixes evaluations. Three quantities are
reported, following the decomposition the debugging of section 20 forced:
exact match, the same with the tokens that can never be an answer excluded,
and the share of answers that were one of those tokens.

Standard data, learned runs, mean over seeds (exact / markers excluded / marker rate):

| Design | 256 | 4096 | 8192 | 16384 |
| --- | --- | --- | --- | --- |
| Bound-key layer first (6) | 100 / 100 / 0.0 | 98.7 / 100.0 / 1.3 | 98.8 / 100.0 / 1.2 | 99.1 / 100.0 / 0.9 |
| Bound keys, layer last (4) | 100 / 100 / 0.0 | 90.6 / 91.6 / 9.4 | 90.4 / 91.4 / 9.4 | 90.5 / 92.2 / 9.4 |
| Layer first, no bound keys (1) | 98.2 / 98.2 / 0.0 | 42.4 / 44.0 / 4.8 | 42.7 / 46.4 / 8.1 | 40.1 / 43.0 / 9.6 |
| Layer last, NoPE (2) | 100 / 100 / 0.0 | 83.2 / 89.2 / 6.2 | 70.6 / 75.7 / 6.2 | 50.9 / 54.6 / 6.2 |

Copy-rich data at 4096 tokens: the bound-key layer first is 84.6 / 99.3 / 14.9,
so its whole regime sensitivity is marker emission; the other designs move by
about a point between the two metrics.

Sink removal at 2048 tokens, bound-key layer first: with markers excluded every
seed scores 100.0% both with position 0 and without it. Seed 1's 28.4-point
loss in exact match is entirely marker emission (marker rate 0.2 to 28.6%).

What this changes.
- Retrieval itself is perfect to 64x for the bound-key layer first (100.0%
  markers excluded on all six seeds) and is the only design that holds: the
  same layer placed last stays near 92%, and the NoPE hybrid falls from 89.2%
  at 16x to 54.6% at 64x. The 64x claim is stronger than section 19 stated.
- The first-slot shortfall of the layer-last design is not only marker
  emission: markers excluded it still loses about 8 points, so that design
  answers the first query with a wrong value as well.
- Predictions 6 and 12 stay failed as they were written, against exact match.
  The decomposition explains them; it does not rescore them.
- Marker rate is now a reported quantity, so the artefact is visible rather
  than hidden in a difference between two numbers.

### Predictions for stage 4

Written before the runs. Stage 4 trains seeds 6, 7 and 8 of the bound-key layer
first and of the same layer placed last, side by side, for the success-rate
comparison that 6 of 6 against 9 of 15 left unsettled (Fisher p = 0.12).

13. Every new bound-key-layer-first seed passes 90% training recall by step 500.
14. Each new bound-key-layer-first seed reaches 90% earlier than the layer-last
    seed trained beside it.
15. With the markers excluded, every new bound-key-layer-first seed keeps at
    least 95% recall at 4096 tokens.

## 22. What the 87.5% ceiling actually is

The ceiling of section 13 and the residual gap of section 21 are the same
thing, and it belongs to the placement, not to the task.

At 1024 tokens, first query slot only, 16 sequences per run
(exact match / markers excluded / marker rate):

| Design | 256 tokens | 1024 tokens |
| --- | --- | --- |
| Bound-key layer first, seeds 0 and 2 | 100 / 100 / 0 | 100 / 100 / 0 |
| Same layer placed last, seed 0 | 100 / 100 / 0 | 0 / 6.2 / 100 |
| Same layer placed last, seed 2 | 100 / 100 / 0 | 0 / 0 / 100 |

Slots 1 to 7 are 100% in both designs at both lengths, so the whole 8-point
shortfall of the layer-last design is its first query, and it is not a wrong
value: the model emits a marker on every first query, and excluding markers it
has no answer either (0 to 6%).

Reading. The query block is announced by a marker token. A global layer placed
first sees that token in the embeddings, so the block is detectable by content
at any length. Placed last, the global layer sees features three delta-rule
layers have already mixed, and the only stable cue for where the block starts
is its absolute offset, which training fixes at 256 tokens. That is why the
layer-last design answers every query but the first, and why the layer-first
design has no ceiling. It also means section 21's residual gap is explained,
not merely bounded.

## 23. Stage 4: three more matched seed pairs

Seeds 6, 7 and 8 of the bound-key layer first and of the same layer placed last,
trained side by side, plus WikiText seed 1 and a rescoring of every design.

| Seed | Bound-key layer first | Same layer placed last |
| --- | --- | --- |
| 6 | step 300, 100% at 4096 | step 1900, 89.3% |
| 7 | step 300, 99.5% | never learned, 0.7% at 256 |
| 8 | step 300, 97.4% | step 1400, 100% |

Predictions 13 to 15 all held: every new seed passed 90% by step 300 (predicted
500), each was earlier than the layer-last seed beside it, and all three keep
100% at 4096 tokens with markers excluded (predicted 95%).

Where the claims stand now.
- Learning speed: 9 of 9 seeds at step 300; paired against the matched control
  it is earlier on 9 of 9 (sign test p = 0.004, against 0.031 at six pairs), and
  the log-rank against the 18 LLLS runs is 3e-07.
- Success rate: 9 of 9 against 11 of 18, Fisher p = 0.059. Still not established
  at the 5% level; three more pairs moved it from 0.12 to 0.059 only.
- Retrieval with markers excluded, over nine seeds: 100.0% at 4096, 8192 and
  16384 tokens; the same layer placed last reaches 92.9, 92.5 and 92.9%.
- WikiText on two seeds: the design costs 0.009 and 0.004 bits per byte. The
  copying gain is 14.7 against 16.3 points on seed 0 and 13.8 against 13.6 on
  seed 1, so prediction 5 stays failed and the honest reading is parity.

The paper and the abstract were updated from these numbers.

### Prediction for stage 5

Written before the runs. Stage 5 adds seeds 9 and 10 of both designs, the two
pairs that take the success-rate comparison past p = 0.05 if the layer-last
design keeps its historical rate (11 of 18). This is the last training run.

16. Both new bound-key-layer-first seeds pass 90% training recall by step 500,
    and the success-rate comparison against the layer-last design reaches
    Fisher p < 0.05.

## 24. Stage 5 outcome, and released checkpoints with valid sink numbers

### Stage 5

| Seed | Bound-key layer first | Same layer placed last |
| --- | --- | --- |
| 9 | step 300, 99.9% at 4096 | step 1500, 99.9% |
| 10 | step 300, 100.0% | step 1500, 98.2% |

Prediction 16 held on both parts, and `scripts/table_predictions.py` now scores
it from the result files: 18 predictions, 13 held, 5 failed.

Where the claims stand (`results/logs/stats_placement.log`).
- Learning speed: 11 of 11 seeds at step 300. Paired with the matched control,
  earlier on 11 of 11 (sign test p = 0.001); log-rank against the 20 layer-last
  hybrids 4.3e-08, unchanged at the 50, 80, 90 and 95% thresholds.
- Success rate: 11/11 against 13/20 layer-last hybrids, Fisher p = 0.033.
  Against the matched control alone (bound keys, layer last) it is 11/11
  against 8/11, p = 0.21, which is not established. The paper states both.
- Accuracy once learned: 99.0 against 91.2% at 4096 as trained (p = 0.062);
  with the length scaling on for every run 99.0 against 93.7% (p = 0.19), and
  99.4 against 99.9% after the first slot (p = 0.61). Parity, as before.

### Released checkpoints, notebook v4

`notebooks/colab_prodscale_probe.ipynb`, executed on a T4 on 16 Sep 2026
(transformers 5.16.1); outputs in `notebooks/internals_v4.json` and
`notebooks/probes_v4.json`. The v3 run is kept as
`notebooks/colab_prodscale_probe_v3.ipynb`.

Sink mass, eager attention in float32, 1024 tokens of filler (probe text in
brackets): SmolLM2-360M 0.396 (0.403), 62x uniform; Qwen2.5-0.5B 0.280
(0.286), 44x; Qwen3-0.6B-Base 0.437 (0.439), 69x; Qwen2.5-1.5B 0.303 (0.311),
48x; Qwen3-1.7B-Base 0.408 (0.415), 64x. v3's `nan` was SDPA returning no
weights.

Mask check, all five `applied`: attention to position 0 is exactly zero with
the mask; the mask moves the final hidden states by 8.1 to 16.9% (relative
L2, 256 tokens); SDPA's masked output sits 0.18 to 0.27% from eager's, the
same as the float16 noise without a mask (0.16 to 0.24%). So v3's intervention
stands: on the paraphrase probe at about 4K tokens, masking position 0 changed
12.5 to 8.3, 0 to 0, 66.7 to 62.5, 45.8 to 45.8 and 79.2 to 79.2%, at most one
prompt of 24 per model, although the representations moved.

Probe fixes and their effect, 4096 tokens, 24 prompts per cell:
- `nearkey` without the duplicate-key bug: 12.5, 75.0, 75.0, 91.7, 95.8% (v3:
  20.8, 70.8, 50.0, 66.7, 62.5). Against `multikey` the near-duplicate keys
  cost 0 to 17 points (-8.3, +4.2, -16.7, -4.1, 0.0); most of the v3 drops were
  the bug.
- `count` with a varying answer: 0% for all five; every generation contains a
  number, never the right one (typical answers 1, 2, 10, 11, 12).
- Generation (32 tokens) against teacher forcing on the free-form probes,
  exact / generated / answer-shaped:
  paraphrase 8.3/12.5/100, 0.0/25.0/41.7, 66.7/66.7/100, 70.8/75.0/100,
  75.0/75.0/100; update 16.7/33.3/100, 0.0/25.0/54.2, 83.3/83.3/100,
  25.0/58.3/100, 91.7/91.7/100 (same model order). Qwen2.5 often answers in a
  full sentence, and the 0.5B model often just continues the filler, so teacher
  forcing undercounts it; Qwen3 answers with the code.
  The Qwen2.5-against-Qwen3 gap on `update` halves under generation (58.3
  against 91.7 at 1.5-1.7B) but does not close.
- The same probe drawn twice differs by up to 25 points (Qwen2.5-1.5B
  paraphrase 45.8 in v3, 70.8 in v4; the length cap changed the prompt draw).
  At 24 prompts a cell, differences of that size are within noise.

Sink mass against probe accuracy across the five models (Spearman, exact
permutation p; the smallest attainable p is 0.017): multikey +0.21 (0.73),
nearkey +0.10 (0.90), paraphrase +0.50 (0.45; generated +0.21), twohop 0.00
(1.00), update +0.80 (0.13; generated +0.80). The update association follows
the family, both Qwen3 models having larger sinks and better updates than both
Qwen2.5 models, so it cannot be separated from the training recipe.

Not run: Llama-3.2-1B and Gemma-2-2b (403, no access to the gated
repositories). Hosted models: 11 calls on the Hugging Face credit before 402
(Llama-3.3-70B simple 5/5 and multikey 4/4, Gemma-4-31B simple 2/2), no free
keys set; not evidence, not reported as results.

Paper wording: "five released checkpoints from two families", descriptive
statistics only, no claim about larger models.

## 25. Layer removal, sink against learning step, and a learning-rate check

Three checks run after stage 5, for the paper's mechanism and robustness
claims. None of them changes a training recipe; the first two only read saved
models.

### Which layer does the lookup (`scripts/head_ablation.py`)

A forward pre-hook on `blk.mix.out` zeroes the attention output of one global
layer, or of one head, before the output projection. Every condition of a model
sees the same inputs (64 at 256 tokens, 48 at 4,096), scored on the CPU in
float32. Results in `results/evals/ablation_layers.json`, 308 rows.

- BKF (11 runs): removing layer 0 drops recall at 4,096 from 99.0 to 0.2%;
  removing layer 4 leaves 100.0% with markers excluded. No single head of
  layer 0 is necessary on its own (at least 84.1% recall without markers).
- Bound keys, layer last (8 learned runs): removing layer 3 drops recall to
  0.4%, removing layer 7 changes nothing.
- Global first without bound keys (the one learned run): removing either global
  layer drops recall from 38.8% to 0.3% and 4.2%, so that lookup needs two
  layers. This is the difference the paper attributes to key binding.

### Sink mass against the learning step (`scripts/sink_vs_step.py`)

Spearman correlation over the runs that learned, written to
`results/summary_sink_step.json`. Across all designs the sign flips between
subsets (-0.51 over all runs, -0.15 without BKF, p = 0.48), so the paper says
only that sink size is no consistent guide.

### Learning rate (`scripts/run_lr_sweep.sh`, `scripts/lr_sweep_summary.py`)

No prediction was written before this one. BKF and the same bound-key layer
placed last, at 1e-3 and 1e-2 (the main runs use 3e-3), seeds 0 and 1, recipe
otherwise unchanged; runs in `results/runs/lr_sweep`, summary in
`results/summary_lr_sweep.json`.

| Learning rate | BKF, learning step | Layer last | BKF, 16x | Layer last, 16x |
| --- | --- | --- | --- | --- |
| 1e-3 | 400, 400 | 1000, 900 | 99.4, 50.3 | 94.4, 62.0 |
| 3e-3 | 300, 300 | 1900, 1900 | 100.0, 97.4 | 87.5, 100.0 |
| 1e-2 | 200, 200 | 1400, 1100 | 100.0, 48.4 | 100.0, 99.9 |

Learning: BKF first in all six pairs, so the placement result does not depend
on the learning rate. Recall at 16x: with markers excluded every one of the
eight runs is at 99.8% or above, so the misses are marker outputs on the early
questions of an input (per-slot accuracy in the run files), not failed lookups.
Exact match away from 3e-3 is seed-dependent for both layouts, which the paper
states in the limitations and in Appendix A.
