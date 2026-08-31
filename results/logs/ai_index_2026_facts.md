# AI Index Report 2026: what bears on this paper

Source: `ai_index_report_2026.pdf` (Stanford HAI), chapter 2, Technical
Performance, section 2.2 Language: the note on retrieval-augmented generation
(report page 84) and the highlight "The Gap Between Long Context Windows and
Deep Understanding" (page 86). Numbers are quoted as the report states them;
the primary sources it cites are named so they can be checked before citing.

- Context windows have grown by almost 30x per year since mid-2023; models
  that accepted a few thousand tokens now accept 1 million or more (Figure
  2.2.5).
- On Fiction.liveBench (narrative comprehension) and MRCR (multi-needle
  retrieval), the input length at which leading models reach 80% accuracy grew
  roughly 250x over nine months (Burnham and Adamczewski, 2025).
- The report's reading: "bigger context windows do not translate into deeper
  understanding, as the gap between accepted and usable context length is
  wide."
- LongBench v2: human experts scored 53.7% under a 15-minute limit and the best
  model 57.7% (Bai et al., 2025); models prompted to reason step by step did
  better.
- Models handle simple lookups but struggle to find multiple matching pieces
  of information or to apply conditions across a long document (Yu et al.,
  2025).
- Longer inputs bring slower responses, higher cost "and reduced accuracy for
  information that appears later in the input". This is the opposite direction
  to the recency bias of our RoPE models; check the primary source before
  using it.
- Measuring long-context ability is hard: separating it from what a model
  already knows reorders models, and one that ranked seventh on raw scores
  ranked first on long-context ability alone (Yang et al., 2025,
  100-LongBench).
- On RAG (page 84): longer windows can hold more retrieved material, "though
  that does not guarantee better performance since models have to parse
  through the information with reliable attention across the entire window."

Use in the paper: motivation only, namely the gap between accepted and usable
context, and the report's call for evaluations that isolate long-context
ability, which position-resolved recall p(d) with matched controls provides.
No claim of ours depends on these numbers.
