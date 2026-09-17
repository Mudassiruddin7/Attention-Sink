"""Does removing attention to position 0 damage ordinary language modelling?

    python scripts/hf_ppl_check.py --model Qwen/Qwen3-0.6B-Base --biases 0 off \
        --out results/interventions/ppl_Qwen3-0.6B-Base.json

Next-token loss on WikiText-103 validation text, with the logit of position 0
shifted in every full-attention layer exactly as in sinkprobe.hf_intervene.
Windows are scored after an offset, so the handful of positions that can see
little besides position 0 do not dominate the average. Windows are identical
across bias levels, so the comparison is paired.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe import hf_intervene                    # noqa: E402
from sinkprobe.hf_probe import load_sentences         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--biases", nargs="+", default=["0", "off"])
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--skip", type=int, default=64)
    ap.add_argument("--text", default="data/wikitext103_validation.txt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_intervene.register()
    torch.cuda.set_per_process_memory_fraction(0.9)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sinkprobe_bias").cuda().eval()
    ids = tok(" ".join(load_sentences(a.text)), add_special_tokens=False,
              return_tensors="pt")["input_ids"][0]
    starts = np.linspace(0, len(ids) - a.length - 1, a.windows).astype(int)
    biases = [float("-inf") if b.lower() in ("off", "-inf") else float(b) for b in a.biases]

    rows = []
    for b in biases:
        hf_intervene.STATE.update(bias=b, collect=False, layers={})
        nll = []
        for st in starts:
            x = ids[st: st + a.length].unsqueeze(0).cuda()
            with torch.no_grad():
                h = model.model(input_ids=x, use_cache=False).last_hidden_state
                total, count = 0.0, 0
                for c in range(a.skip, a.length - 1, 256):
                    e = min(a.length - 1, c + 256)
                    logits = model.lm_head(h[:, c:e]).float()
                    total += torch.nn.functional.cross_entropy(
                        logits[0], x[0, c + 1:e + 1], reduction="sum").item()
                    count += e - c
            nll.append(total / count)
        rows.append({"bias": b, "nll": float(np.mean(nll)), "ppl": float(math.exp(np.mean(nll))),
                     "nll_per_window": nll})
        print(f"{a.model} bias {b}: NLL {np.mean(nll):.4f} nats/token, perplexity {math.exp(np.mean(nll)):.2f}",
              flush=True)

    base = np.array(rows[0]["nll_per_window"])
    for r in rows[1:]:
        d = np.array(r["nll_per_window"]) - base
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
        r["nll_diff"] = [float(d.mean()), float(half)]
        print(f"  bias {r['bias']} minus bias {rows[0]['bias']}: {d.mean():+.4f} ± {half:.4f} nats/token "
              f"over {len(d)} paired windows", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"model": a.model, "args": vars(a), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
