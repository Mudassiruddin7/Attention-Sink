"""Sink diagnostics and position resolved retrieval on released checkpoints.

    python -m sinkprobe.hf_probe --model Qwen/Qwen3-0.6B-Base \
        --lengths 2048 8192 32768 --trials 8 --out results/hf/qwen3_0.6b.json

The probe registers its own attention function with transformers. It
computes the layer output with the standard SDPA kernel and, during the
prefill pass only, reads the post-rotary queries and keys to measure how much
attention lands on position 0. Queries are subsampled and scored against every
visible key with a chunked logsumexp, so the full T x T map is never held in
memory. Linear attention layers produce no distribution over positions and
are left out, and the number of layers used is recorded beside every value.
For layers with a sigmoid output gate (Qwen3.5), the gate is read from the
query projection.

Retrieval follows the multi key needle task of RULER (Hsieh et al., 2024):
four needles of the form "One of the special magic numbers for KEY is: N."
sit in natural text, and one of them is asked for.

Scoring is teacher-forced exact match: the gold answer is appended and the
trial counts as correct when every answer token is the argmax given the gold
prefix. Greedy decoding reproduces the gold answer exactly when, and only
when, this holds, so the verdict equals greedy exact match while needing one
forward pass and no key value cache. That is what lets a 4 GB card reach
30K tokens. The summed log probability of the answer is kept as a graded
score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from typing import Dict, List

import numpy as np
import torch

from .metrics import wilson_interval

KEYS = (
    "amber anchor apricot arrow basalt beacon birch bramble breeze bronze cactus canyon "
    "cedar cinder clover cobalt comet coral cotton crater cypress dune ember falcon fern "
    "fjord flint garnet glacier granite harbor hazel heron indigo iris ivory jasper juniper "
    "kestrel lagoon lantern lava lilac linen lotus magnet maple marble meadow mercury mica "
    "mint monsoon mosaic nectar nickel oasis obsidian olive onyx opal orchid otter pebble "
    "pepper pine plume prairie quartz quill raven reef ripple saffron sage sapphire shale "
    "sierra silver slate sparrow spruce summit tangent thistle timber topaz tulip tundra "
    "umber valley velvet violet walnut willow zephyr"
).split()

PREFIX = ("Some special magic numbers are hidden within the following text. Make sure to "
          "memorize it. I will quiz you about the numbers afterwards.\n")
NEEDLE = "One of the special magic numbers for {key} is: {value}."
QUESTION = ("\nWhat is the special magic number for {key} mentioned in the provided text? "
            "The special magic number for {key} mentioned in the provided text is:")


# ---------------------------------------------------------------------------
# attention hook
# ---------------------------------------------------------------------------

class Collector:
    def __init__(self, n_queries: int = 384, chunk: int = 64):
        self.active = False
        self.n_queries = n_queries
        self.chunk = chunk
        self.layers: Dict[int, Dict] = {}
        self.gates: Dict[int, float] = {}

    def reset(self):
        self.layers, self.gates = {}, {}


COLLECT = Collector()


@torch.no_grad()
def _sink_stats(query, key, scaling):
    b, hq, t, dh = query.shape
    hkv = key.shape[1]
    if t < 3:
        return None
    n = min(COLLECT.n_queries, t - 1)
    pos = torch.unique(torch.cat([
        torch.linspace(1, t - 1, n, device=query.device).round().long(),
        torch.arange(max(1, t - 16), t, device=query.device)]))
    k = key.float()
    if hkv != hq:
        k = k.repeat_interleave(hq // hkv, dim=1)
    kt = k.transpose(-1, -2)
    keys = torch.arange(t, device=query.device)
    a0_sum = torch.zeros(hq, device=query.device, dtype=torch.float64)
    ent_norm_sum = torch.zeros((), device=query.device, dtype=torch.float64)
    ref_sum = 0.0
    # Keep each (B, H, chunk, T) block near 60 MB in float32 at long contexts.
    chunk = max(4, min(COLLECT.chunk, int(1.5e7 // max(1, b * hq * t))))
    for s in range(0, len(pos), chunk):
        p = pos[s:s + chunk]
        logits = (query[:, :, p].float() @ kt) * scaling                       # (B,H,c,T)
        logits = logits.masked_fill(keys[None, None, None, :] > p[None, None, :, None],
                                    float("-inf"))
        lse = torch.logsumexp(logits, dim=-1)
        a0_sum += torch.exp(logits[..., 0] - lse).double().sum(dim=(0, 2))
        logp = logits - lse[..., None]
        prob = logp.exp()
        ent = -(prob * logp.nan_to_num(neginf=0.0)).sum(-1)                      # (B,H,c)
        ent_norm_sum += (ent / torch.log(p.double() + 1.0)[None, None, :]).double().sum()
        ref_sum += float((1.0 / (p.double() + 1.0)).sum())
    count = b * len(pos)
    per_head = (a0_sum / count).tolist()
    return {"a0_per_head": per_head, "a0": float(np.mean(per_head)),
            "ref": ref_sum / len(pos), "entropy_norm": float(ent_norm_sum / (count * hq)),
            "n_queries": int(len(pos)), "seq_len": t}


def register(transformers_module=None):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    from transformers.modeling_utils import AttentionInterface

    def sinkprobe_attention(module, query, key, value, attention_mask, **kwargs):
        scaling = kwargs.get("scaling") or getattr(module, "scaling", query.shape[-1] ** -0.5)
        if COLLECT.active and query.shape[2] > 1:
            st = _sink_stats(query, key, scaling)
            if st is not None:
                COLLECT.layers[int(getattr(module, "layer_idx", len(COLLECT.layers)))] = st
        if query.shape[2] != key.shape[2]:
            # Cached decoding is not used by the probe; keep the library path for it.
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        # One unpadded sequence, so causal masking is exact. Repeating the key and
        # value heads ourselves keeps SDPA on the memory-efficient kernel; the GQA
        # path of the library falls back to the O(T^2) math kernel on this build.
        hq, hkv = query.shape[1], key.shape[1]
        if hkv != hq:
            key = key.repeat_interleave(hq // hkv, dim=1)
            value = value.repeat_interleave(hq // hkv, dim=1)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("sinkprobe", sinkprobe_attention)
    AttentionMaskInterface.register("sinkprobe", sdpa_mask)


def add_gate_hooks(model):
    """Record the mean sigmoid output gate for attention layers that have one."""
    n = 0
    for name, mod in model.named_modules():
        q = getattr(mod, "q_proj", None)
        if q is None or not getattr(getattr(mod, "config", None), "attn_output_gate", False):
            continue
        hd = mod.head_dim
        idx = int(getattr(mod, "layer_idx", n))

        def hook(_m, _inp, out, idx=idx, hd=hd):
            if COLLECT.active and out.shape[1] > 1:
                gate = out.view(*out.shape[:2], -1, 2 * hd)[..., hd:]
                COLLECT.gates[idx] = float(torch.sigmoid(gate.float()).mean())
        q.register_forward_hook(hook)
        n += 1
    return n


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def load_sentences(path: str) -> List[str]:
    text = open(path, encoding="utf-8").read()
    text = re.sub(r" @(.)@ ", r"\1", text)
    text = re.sub(r"^\s*=.*=\s*$", " ", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[.!?]) (?=[A-Z])", text)
    return [s.strip() for s in sents if 20 <= len(s.strip()) <= 600]


class PromptBuilder:
    def __init__(self, tokenizer, sentences: List[str]):
        self.tok = tokenizer
        self.sents = sentences
        lens = tokenizer(sentences, add_special_tokens=False)["input_ids"]
        self.lens = np.array([len(x) + 1 for x in lens])
        self.cum = np.concatenate([[0], np.cumsum(self.lens)])

    def build(self, length: int, depth: float, rng: random.Random, n_needles: int = 16):
        keys = rng.sample(KEYS, n_needles)
        values = [str(rng.randint(1_000_000, 9_999_999)) for _ in keys]
        needles = [NEEDLE.format(key=k, value=v) for k, v in zip(keys, values)]
        overhead = len(self.tok(PREFIX + QUESTION.format(key=keys[0]) + " ".join(needles),
                                add_special_tokens=False)["input_ids"])
        budget = length - overhead
        start = rng.randrange(0, len(self.sents) // 2)
        end = int(np.searchsorted(self.cum, self.cum[start] + budget))
        if end >= len(self.sents):
            start = 0
            end = int(np.searchsorted(self.cum, budget))
        ctx = list(self.sents[start:end])
        m = len(ctx)
        slots = {0: int(round(depth * m))}
        for i in range(1, n_needles):
            slots[i] = rng.randrange(0, m + 1)
        for i in sorted(slots, key=lambda j: (slots[j], j), reverse=True):
            ctx.insert(slots[i], needles[i])
        context = " ".join(ctx)
        prompt = PREFIX + context + QUESTION.format(key=keys[0])
        needle_char = len(PREFIX) + context.find(needles[0])
        return prompt, values[0], needle_char, len(PREFIX), len(PREFIX) + len(context)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_answer(model, tok, prompt_ids, value: str):
    ans = tok(" " + value, add_special_tokens=False)["input_ids"]
    gold = torch.tensor(ans, device=prompt_ids.device)
    ids = torch.cat([prompt_ids, gold[None]], dim=1)
    out = model(input_ids=ids, use_cache=False, logits_to_keep=len(ans) + 1)
    logits = out.logits[0, :-1].float()                   # predicts each answer token
    correct = int((logits.argmax(-1) == gold).all())
    logprob = float(torch.log_softmax(logits, -1).gather(1, gold[:, None]).sum())
    first = tok.decode([int(logits[0].argmax())])
    return correct, logprob, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768])
    ap.add_argument("--depths", type=int, default=11)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--sink-trials", type=int, default=2)
    ap.add_argument("--text", default="data/wikitext103_validation.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.9)
    ap.add_argument("--needles", type=int, default=16,
                    help="needles in the context, one asked for and the rest distractors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    register()
    # Fail with an out-of-memory error instead of spilling into shared memory,
    # which on Windows turns a slow run into one that never finishes.
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sinkprobe").cuda().eval()
    n_gated = add_gate_hooks(model)
    builder = PromptBuilder(tok, load_sentences(a.text))
    rng = random.Random(a.seed)

    rows, sinks = [], []
    if os.path.exists(a.out):
        prev = json.load(open(a.out, encoding="utf-8"))
        rows, sinks = prev["rows"], prev["sinks"]
    done = {(r["length"], r["depth_target"], r["trial"]) for r in rows}
    t_start = time.time()
    for length in a.lengths:
        for di in range(a.depths):
            depth = di / (a.depths - 1)
            for trial in range(a.trials):
                prompt, value, nchar, c0, c1 = builder.build(length, depth, rng, a.needles)
                if (length, round(depth, 3), trial) in done:
                    continue
                enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True,
                          add_special_tokens=False)
                ids = enc["input_ids"].cuda()
                offs = enc["offset_mapping"][0, :, 0].numpy()
                t_needle = int(np.searchsorted(offs, nchar))
                t_c0, t_c1 = int(np.searchsorted(offs, c0)), int(np.searchsorted(offs, c1))
                collect = trial < a.sink_trials
                COLLECT.reset()
                COLLECT.active = collect
                t0 = time.time()
                correct, logprob, first = score_answer(model, tok, ids, value)
                row = {"length": length, "depth_target": round(depth, 3), "trial": trial,
                       "n_tokens": int(ids.shape[1]),
                       "depth_actual": (t_needle - t_c0) / max(1, t_c1 - t_c0),
                       "correct": correct, "answer_logprob": logprob,
                       "first_token": first, "seconds": round(time.time() - t0, 2),
                       "peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
                rows.append(row)
                if collect and COLLECT.layers:
                    per_layer = {str(k): v for k, v in sorted(COLLECT.layers.items())}
                    heads = [h for v in COLLECT.layers.values() for h in v["a0_per_head"]]
                    a0 = float(np.mean([v["a0"] for v in COLLECT.layers.values()]))
                    ref = float(np.mean([v["ref"] for v in COLLECT.layers.values()]))
                    sinks.append({"length": length, "depth_target": round(depth, 3),
                                  "trial": trial, "n_tokens": int(ids.shape[1]),
                                  "layers_used": len(COLLECT.layers), "sink_mass": a0,
                                  "sink_ref": ref, "sink_ratio": a0 / ref,
                                  "sink_rate": float(np.mean([h > 0.3 for h in heads])),
                                  "entropy_norm": float(np.mean([v["entropy_norm"] for v in COLLECT.layers.values()])),
                                  "gate_mean": (float(np.mean(list(COLLECT.gates.values())))
                                                if COLLECT.gates else None),
                                  "per_layer": per_layer})
                COLLECT.active = False
                print(f"{a.model} len {length:6d} depth {depth:.1f} trial {trial} "
                      f"ok {row['correct']} lp {logprob:.2f} ({row['n_tokens']} tok, "
                      f"{row['seconds']}s, {row['peak_mib']} MiB) "
                      f"{'sink %.3f' % sinks[-1]['sink_mass'] if collect and sinks else ''}",
                      flush=True)
            _save(a, rows, sinks, n_gated, t_start)
    _save(a, rows, sinks, n_gated, t_start)
    summarise(rows, sinks)


def _save(a, rows, sinks, n_gated, t_start):
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": a.model, "args": vars(a), "gated_layers": n_gated,
                   "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                   "elapsed_s": round(time.time() - t_start, 1), "rows": rows,
                   "sinks": sinks}, f)
    os.replace(tmp, a.out)


def summarise(rows, sinks):
    for length in sorted({r["length"] for r in rows}):
        rs = [r for r in rows if r["length"] == length]
        hits = sum(r["correct"] for r in rs)
        lo, hi = wilson_interval(hits, len(rs))
        q = lambda a, b: np.mean([r["correct"] for r in rs if a <= r["depth_actual"] < b] or [np.nan])
        ss = [s for s in sinks if s["length"] == length]
        print(f"len {length:6d} recall {hits / len(rs):.3f} [{lo:.2f},{hi:.2f}] "
              f"Q1 {q(0, .25):.2f} Q4 {q(.75, 1.01):.2f} "
              f"sink {np.mean([s['sink_mass'] for s in ss]) if ss else float('nan'):.3f} "
              f"ratio {np.mean([s['sink_ratio'] for s in ss]) if ss else float('nan'):.1f}")


if __name__ == "__main__":
    main()
