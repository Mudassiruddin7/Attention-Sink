"""Push attention off position 0 in a released checkpoint and re-measure retrieval by depth.

    python -m sinkprobe.hf_intervene --model Qwen/Qwen3-0.6B-Base --biases 0 -2 -4 -inf \
        --lengths 2048 4096 8192 --depths 11 --trials 4 --out results/hf/intervene_Qwen3-0.6B-Base.json

A bias b is added to the logit of position 0 in every full-attention layer,
for every query after the first. b = -inf removes the sink entirely. Every bias
level is scored on the same prompts, so differences between levels are
paired. The biased layer output is computed exactly from the unbiased one
(sinkprobe.interventions), so the attention itself still runs on the
memory-efficient SDPA kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .hf_probe import PromptBuilder, load_sentences, score_answer
from .interventions import biased_first_key_output, first_key_attention

STATE = {"bias": 0.0, "collect": False, "layers": {}}


def register():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    from transformers.modeling_utils import AttentionInterface

    def attention(module, query, key, value, attention_mask, **kwargs):
        if query.shape[2] != key.shape[2]:
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        scaling = kwargs.get("scaling") or getattr(module, "scaling", query.shape[-1] ** -0.5)
        hq, hkv = query.shape[1], key.shape[1]
        if hkv != hq:
            key = key.repeat_interleave(hq // hkv, dim=1)
            value = value.repeat_interleave(hq // hkv, dim=1)
        kernels = ([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if query.is_cuda
                   else [SDPBackend.MATH])
        with sdpa_kernel(kernels):
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=scaling)
        bias = STATE["bias"]
        if bias != 0.0 or STATE["collect"]:
            p0 = first_key_attention(query, key, scaling)
            if bias != 0.0:
                new, p0 = biased_first_key_output(out.float(), value[:, :, :1].float(), p0, bias)
                out = new.to(out.dtype)
            if STATE["collect"]:
                t = p0.shape[-1]
                per_head = p0[:, :, 1:].double().mean(dim=(0, 2))
                ref = float((1.0 / torch.arange(2, t + 1, dtype=torch.float64)).mean())
                idx = int(getattr(module, "layer_idx", len(STATE["layers"])))
                STATE["layers"][idx] = {"a0": float(per_head.mean()), "ref": ref,
                                        "a0_per_head": per_head.tolist()}
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("sinkprobe_bias", attention)
    AttentionMaskInterface.register("sinkprobe_bias", sdpa_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--biases", nargs="+", default=["0", "-2", "-4", "off"],
                    help="logit shifts for position 0; 'off' removes it entirely (-inf)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--depths", type=int, default=11)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--sink-trials", type=int, default=1)
    ap.add_argument("--needles", type=int, default=16)
    ap.add_argument("--text", default="data/wikitext103_validation.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    register()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sinkprobe_bias").cuda().eval()
    builder = PromptBuilder(tok, load_sentences(a.text))
    biases = [float("-inf") if str(b).lower() in ("off", "-inf", "ninf") else float(b)
              for item in a.biases for b in str(item).split(",") if b.strip()]

    rows, sinks = [], []
    if os.path.exists(a.out):
        prev = json.load(open(a.out, encoding="utf-8"))
        rows, sinks = prev["rows"], prev["sinks"]
    done = {(r["bias"], r["length"], r["depth_index"], r["trial"]) for r in rows}
    t_start = time.time()
    for length in a.lengths:
        for di in range(a.depths):
            depth = di / (a.depths - 1)
            for trial in range(a.trials):
                rng = random.Random(f"{a.seed}-{length}-{di}-{trial}")
                prompt, value, nchar, c0, c1 = builder.build(length, depth, rng, a.needles)
                enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True,
                          add_special_tokens=False)
                ids = enc["input_ids"].cuda()
                offs = enc["offset_mapping"][0, :, 0].numpy()
                t_needle, t_c0, t_c1 = (int(np.searchsorted(offs, z)) for z in (nchar, c0, c1))
                for bias in biases:
                    if (bias, length, di, trial) in done:
                        continue
                    STATE.update(bias=bias, collect=trial < a.sink_trials, layers={})
                    t0 = time.time()
                    correct, logprob, first = score_answer(model, tok, ids, value)
                    rows.append({"bias": bias, "length": length, "depth_index": di,
                                 "depth_target": round(depth, 3), "trial": trial,
                                 "n_tokens": int(ids.shape[1]),
                                 "depth_actual": (t_needle - t_c0) / max(1, t_c1 - t_c0),
                                 "correct": correct, "answer_logprob": logprob,
                                 "first_token": first, "seconds": round(time.time() - t0, 2)})
                    if STATE["collect"] and STATE["layers"]:
                        lay = STATE["layers"]
                        a0 = float(np.mean([v["a0"] for v in lay.values()]))
                        ref = float(np.mean([v["ref"] for v in lay.values()]))
                        heads = [x for v in lay.values() for x in v["a0_per_head"]]
                        sinks.append({"bias": bias, "length": length, "depth_index": di,
                                      "trial": trial, "layers_used": len(lay), "sink_mass": a0,
                                      "sink_ratio": a0 / ref,
                                      "sink_rate": float(np.mean([x > 0.3 for x in heads])),
                                      "per_layer": {str(k): v["a0"] for k, v in sorted(lay.items())}})
                    STATE["collect"] = False
                    print(f"{a.model} bias {bias:>5} len {length:6d} depth {depth:.1f} trial {trial} "
                          f"ok {correct} lp {logprob:.2f}"
                          + (f" sink {sinks[-1]['sink_mass']:.3f}" if trial < a.sink_trials and sinks else ""),
                          flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            tmp = a.out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"model": a.model, "args": vars(a), "elapsed_s": round(time.time() - t_start, 1),
                           "rows": rows, "sinks": sinks}, f)
            os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
