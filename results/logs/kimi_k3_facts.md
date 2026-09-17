# Kimi K3 facts checked against the technical report

Source: Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv:2607.24653
(47 pages). Text extracted with pypdf from the arXiv PDF on 2026-09-15;
the first 40 pages are saved in `kimi_k3_first40pages.txt`.

## Confirmed (Table 1 and Section 2)

| Quantity | Value in the report |
| --- | --- |
| Layers | 93 (Kimi K2: 61) |
| Attention layer composition | 69 KDA + 24 MLA |
| Block pattern | 3 KDA layers followed by 1 Gated MLA layer |
| Total / activated parameters | 2.78T / 104.2B |
| Hidden dimension | 7,168 |
| Attention heads | 96 |
| Routed experts / active per token | 896 / 16 |
| Positional encoding | NoPE on all MLA layers; position comes from KDA decay |
| Context | pre-training at 8K, extended to 64K; extrapolates to 1M without RoPE rescaling |
| KDA | delta rule with channel-wise forget gate, lower-bounded decay, full-rank output gate |
| Gated MLA | input-dependent, channel-wise, full-rank output gate |
| AttnRes | learned pseudo-queries over the embedding and block outputs, RMSNorm on sources, partial sums within a block |

## Not stated in the first 40 pages

- MLA KV latent rank and KDA state head geometry (the cost model keeps these as declared assumptions).
- No occurrence of "attention sink", "massive activation", "needle", "RULER", "LongBench" or "lost in the middle". The only "first token" hit is time-to-first-token in the serving section.
  Pages 41 to 47 were searched afterwards (saved in `kimi_k3_pages41_47.txt`) with the same terms plus "haystack", "position bias" and "recency": zero hits. The report publishes no sink or position-resolved retrieval diagnostics anywhere in its 47 pages.

## Earlier claims in paper.tex that need care

- "2.8 trillion parameter" matches 2.78T.
- "eight times past the range where these diagnostics have been reported" compares 1M with the 128K of Qiu et al.; K3 itself trains to 64K.
- "16 of 896 experts" matches.
