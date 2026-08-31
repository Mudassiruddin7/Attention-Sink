# DeepSeek-V4.1-Flash technical report: what bears on this paper

Source: `DeepSeek_V41_Tech_Report.pdf`, "DeepSeek-V4.1-Flash: Pushing the Limits
of KV Cache Compression" (DeepSeek-AI). Sections and tables refer to the
report; numbers are as stated there.

## Model (sections 2 and 4.2.1)

- Multimodal MoE with 552B backbone parameters and 196B Engram parameters; 8B
  active per token in prefill and 16B in decode; contexts up to 1M tokens;
  trained on 45T tokens.
- 40 layers, split into a 20-layer causal encoder and a 20-layer decoder
  (Causal Encoder-Decoder, after YOCO). The decoder's global KV is projected
  from the last encoder layer's hidden state; sliding-window KV stays
  layer-local.
- Every layer has global attention and sliding-window attention (window 128),
  except the first two layers, which have sliding-window attention only.
- Global attention is Compressed Sparse Attention 2 (CSA2). An indexer scores
  compressed KV entries and each query attends to the top 512 of them together
  with its local window. Encoder layers compress two tokens into one entry;
  decoder layers keep every token. Layers share main KV, indexer keys and top-k
  indices through Full, Reindex and Reuse modes, and a hierarchical indexer
  limits later decoder indexers to 16,384 candidates chosen by the first.
- CSA2 dropped two things CSA had inside its compressor: overlapping spans and
  an absolute positional embedding. The KV latent has RoPE and non-RoPE parts,
  both cached in FP4.
- Engram (hashed n-gram memory) is kept without its short causal convolution,
  whose "performance gains do not justify the added complexity" in their
  inference stack.

## Training (section 4.2.2)

- Sparse attention is trained from scratch at 64K tokens "without any dense
  attention warmup stages", and the sequence length is extended to 1M at 34T
  tokens.

## Evaluation (section 4.3, Table 1)

- LongBench-V2 (EM, 1-shot) is the only long-context benchmark for the base
  models: V4-Flash 44.7, V4-Pro 51.5, V4.1-Flash 45.2. The 6.3-point gap to
  V4-Pro is the second largest in Table 1, after SimpleQA-Verified (12.9).
- The report has no attention-sink, position-bias or needle analysis (searched
  for sink, lost in the middle, needle, RULER and position).

## Limitations (section 6)

- "Potential selection errors in CSA2 and approximate state reconstruction in
  SWA Bounded Replay may still cause capability degradation in untested
  boundary cases", with future stress tests aimed at "sparse retrieval over
  long contexts".

## How it connects to our results

These are hypotheses about a model 300,000 times larger than ours, not tests.

1. Keys that carry local context. CSA2's global keys summarize adjacent tokens,
   and from the third layer on every layer pairs global attention with a local
   window. In our controlled models a global layer learns retrieval early when
   its keys are bound to their preceding tokens and it reads token-level
   features. DeepSeek's layout differs (two local-only layers come first) and the
   report ablates neither choice, so this suggests why compressed-span keys may
   help sparse retrieval; it does not support our claim.
2. Warm-up. DeepSeek trains sparse global attention without a dense-attention
   warm-up. Ours is a different warm-up (copy-rich data), so the parallel is
   loose: the bound-key layer first learned without it (2 of 2 seeds) where the
   usual layout did not (0 of 1).
3. What is not measured. A 1M-context model that trails its larger sibling by
   6.3 points on long context, and whose authors name sparse retrieval as an
   untested boundary, reports no position-resolved retrieval. That is the gap
   position-resolved recall with matched controls addresses, and a reason not
   to treat sink size as a proxy.
4. Local convolutions are not a free win everywhere: Engram dropped its short
   convolution. Our claim stays limited to attention keys in the tested models.
