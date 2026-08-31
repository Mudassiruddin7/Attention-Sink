"""Diagnostics for attention sinks, massive activations and position bias.

Sink quantities are computed on softmax layers only and averaged over layers,
heads and query positions 1..T-1 (position 0 can only attend to itself).

    sink_mass      mean attention on position 0, the F-Attn column of
                   Qiu et al. (2025)
    sink_ref       the same quantity for a head that attends uniformly over
                   the visible prefix, mean over t of 1/(t+1). This is about
                   ln(T)/T, not 1/T, and it shrinks with length, which is why
                   raw sink mass falls with length even when nothing changes.
    sink_ratio     sink_mass / sink_ref, the length-fair version
    sink_rate      share of (layer, head) pairs whose mean attention on
                   position 0 exceeds 0.3, the Sink-epsilon metric of
                   Gu et al. (2025)
    sink_<type>    sink mass restricted to query positions of one type
                   (noop, copy, answer), see sinkprobe.data

Activation quantities follow Sun et al. (2024).

    act_max        largest |h| entering each layer, averaged over layers
    act_ratio      that maximum over the median |h| of the same layer
    act_on_bos     share of layers whose largest |h| sits on position 0

Position bias is read from recall at the answer positions, binned by where
the queried pair sat in the haystack.

    recency_gap    recall in the last quarter minus recall in the first
    middle_dip     mean recall of the two outer quarters minus the two inner
    depth_slope    least squares slope of recall on depth, in points per unit
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .data import ANSWER, TYPE_NAMES


def wilson_interval(hits: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def trials_for_halfwidth(halfwidth: float, p: float = 0.5, z: float = 1.96) -> int:
    return int(math.ceil(z * z * p * (1 - p) / (halfwidth ** 2)))


def uniform_sink_reference(seq_len: int) -> float:
    t = np.arange(1, seq_len, dtype=np.float64)
    return float(np.mean(1.0 / (t + 1.0)))


def attention_diagnostics(stats: Dict, types: torch.Tensor, rate_eps: float = 0.3):
    layers = [l for l in stats["layers"] if "a0" in l]
    out: Dict[str, float] = {"n_softmax_layers": len(layers)}
    if not layers:
        return out
    a0 = torch.stack([l["a0"][:, :, 1:] for l in layers]).double()     # (Ls,B,H,T-1)
    ent = torch.stack([l["ent"][:, :, 1:] for l in layers]).double()
    ls, b, h, tq = a0.shape
    seq_len = tq + 1
    visible = torch.arange(2, seq_len + 1, dtype=torch.float64)           # keys seen by query t
    per_layer = a0.mean(dim=(1, 2, 3))
    per_head = a0.mean(dim=(1, 3))                                        # (Ls,H)
    out.update({
        "sink_mass": per_layer.mean().item(),
        "sink_worst_layer": per_layer.max().item(),
        "sink_ref": uniform_sink_reference(seq_len),
        "sink_rate": (per_head > rate_eps).double().mean().item(),
        "sink_per_layer": per_layer.tolist(),
        "sink_per_head": per_head.tolist(),
        "entropy": ent.mean().item(),
        "entropy_norm": (ent / visible.log()).mean().item(),
    })
    out["sink_ratio"] = out["sink_mass"] / out["sink_ref"]

    by_query = a0.mean(dim=(0, 2))                                        # (B,T-1)
    ty = types[:, 1:]
    inv_vis = (1.0 / visible).expand(b, tq)
    for code, name in TYPE_NAMES.items():
        m = ty == code
        if m.any():
            out[f"sink_{name}"] = by_query[m].mean().item()
            out[f"sink_ref_{name}"] = inv_vis[m].mean().item()
            out[f"n_query_{name}"] = int(m.sum())

    virt = [l["virt"][:, :, 1:] for l in layers if "virt" in l]
    if virt:
        out["virtual_sink"] = torch.stack(virt).double().mean().item()
    if any("row_sum" in l for l in layers):
        rs = torch.stack([l["row_sum"][:, :, 1:] for l in layers if "row_sum" in l]).double()
        out["row_mass"] = rs.mean().item()
    gates = [l for l in stats["layers"] if "gate" in l]
    if gates:
        g = torch.stack([l["gate"] for l in gates]).double()              # (Lg,B,T)
        out["gate_mean"] = g.mean().item()
        out["gate_lt_01"] = torch.stack([l["gate_lt_01"] for l in gates]).double().mean().item()
        gq = g.mean(0)
        for code, name in TYPE_NAMES.items():
            m = types == code
            if m.any():
                out[f"gate_{name}"] = gq[m].mean().item()
    return out


def activation_diagnostics(stats: Dict):
    maxes, ratios, on_bos = [], [], []
    rows = [(l["h_absmax"], l["h_median"]) for l in stats["layers"]]
    rows.append((stats["final_absmax"], stats["final_median"]))
    for absmax, med in rows:
        top = absmax.max().item()
        maxes.append(top)
        ratios.append(top / max(med, 1e-8))
        on_bos.append((absmax.argmax(dim=1) == 0).double().mean().item())
    out = {"act_max": float(np.mean(maxes)), "act_peak": float(np.max(maxes)),
           "act_ratio": float(np.mean(ratios)), "act_on_bos": float(np.mean(on_bos)),
           "act_per_layer": maxes}
    lin = [l for l in stats["layers"] if "beta" in l]
    if lin:
        out["linear_beta"] = float(torch.stack([l["beta"] for l in lin]).mean())
        out["linear_alpha"] = float(torch.stack([l["alpha"] for l in lin]).mean())
    if stats.get("depth_weights") is not None:
        out["depth_weights"] = stats["depth_weights"]
    return out


def depth_profile(correct: np.ndarray, depth: np.ndarray, n_bins: int = 10):
    correct = correct.astype(np.float64)
    out: Dict = {"recall": float(correct.mean()), "n_answers": int(len(correct))}
    hits = int(correct.sum())
    out["recall_ci"] = wilson_interval(hits, len(correct))
    bins = np.clip((depth * n_bins).astype(int), 0, n_bins - 1)
    prof = []
    for i in range(n_bins):
        m = bins == i
        n = int(m.sum())
        hb = int(correct[m].sum())
        lo, hi = wilson_interval(hb, n)
        prof.append({"bin": i, "centre": (i + 0.5) / n_bins, "n": n,
                     "acc": hb / n if n else float("nan"), "lo": lo, "hi": hi})
    out["profile"] = prof
    quart = np.clip((depth * 4).astype(int), 0, 3)
    q = [float(correct[quart == i].mean()) if (quart == i).any() else float("nan")
         for i in range(4)]
    out.update({f"acc_q{i + 1}": q[i] for i in range(4)})
    out["recency_gap"] = q[3] - q[0]
    out["middle_dip"] = (q[0] + q[3]) / 2 - (q[1] + q[2]) / 2
    if len(correct) > 2 and depth.std() > 0:
        out["depth_slope"] = float(np.polyfit(depth, correct, 1)[0])
    return out


@torch.no_grad()
def evaluate(model, task, seq_len: int, n_seqs: int, rng: np.random.Generator,
             device, batch: int | None = None, collect_seqs: int = 8) -> Dict:
    model.eval()
    batch = batch or max(4, (64 * 256) // seq_len)
    if seq_len > 2048:
        # One sequence is enough for averages over heads, layers and 4K queries,
        # and keeps the explicit attention maps within a 4 GB card.
        collect_seqs = min(collect_seqs, 1)
    if getattr(model, "explicit_attention", False) and seq_len > 512:
        batch = max(1, batch // 4)
        if seq_len > 1024:
            # Softpick and sink-logit layers keep several T x T maps per layer.
            collect_seqs = min(collect_seqs, 2)
    res = {"seq_len": seq_len}
    if collect_seqs > 0:
        # collect_seqs = 0 skips the diagnostics, whose explicit attention maps do not
        # fit past about 8K tokens on a 4 GB card; the recall sequences then differ.
        cb = task.build(min(collect_seqs, n_seqs), seq_len, rng, skew=0.0)
        _, stats = model(cb["tokens"].to(device), collect=True)
        res.update(attention_diagnostics(stats, cb["types"]))
        res.update(activation_diagnostics(stats))
        del stats

    # The query marker can outscore the retrieved value past the training length
    # (method notes, sections 13 and 20); scoring without it separates the readout
    # artefact from retrieval itself.
    reserved = list(getattr(task, "reserved_tokens", tuple)())
    correct, depth, slot_hits, correct_no_marker, marker_out = [], [], [], [], []
    ce_sum = {name: 0.0 for name in TYPE_NAMES.values()}
    ce_cnt = {name: 0 for name in TYPE_NAMES.values()}
    done = 0
    while done < n_seqs:
        bs = min(batch, n_seqs - done)
        d = task.build(bs, seq_len, rng, skew=0.0)
        x = d["tokens"].to(device)
        logits, _ = model(x)
        logits = logits.float()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             x[:, 1:].reshape(-1), reduction="none").view(bs, -1).cpu()
        ty = d["types"][:, :-1]
        for code, name in TYPE_NAMES.items():
            m = ty == code
            ce_sum[name] += float(ce[m].sum())
            ce_cnt[name] += int(m.sum())
        ap = d["answer_pos"].to(device)
        rows = logits.gather(1, ap[..., None].expand(-1, -1, logits.size(-1)))
        gold = x.gather(1, ap + 1)
        hits = (rows.argmax(-1) == gold).cpu().numpy()
        if reserved:
            free = rows.clone()
            free[..., reserved] = -float("inf")
            correct_no_marker.append((free.argmax(-1) == gold).cpu().numpy().ravel())
            marker_out.append(torch.isin(rows.argmax(-1), torch.tensor(reserved, device=rows.device))
                              .cpu().numpy().ravel())
        slot_hits.append(hits)
        correct.append(hits.ravel())
        depth.append(d["answer_depth"].numpy().ravel())
        done += bs
    res.update(depth_profile(np.concatenate(correct), np.concatenate(depth)))
    if correct_no_marker:
        # Three quantities, not one: what the model answers, what it answers when the
        # tokens that can never be an answer are excluded, and how often it emits one.
        res["recall_no_marker"] = float(np.concatenate(correct_no_marker).mean())
        res["marker_rate"] = float(np.concatenate(marker_out).mean())
    by_slot = np.concatenate(slot_hits, axis=0)
    res["acc_by_slot"] = by_slot.mean(axis=0).tolist()
    if by_slot.shape[1] > 1:
        # Every query but the first in the block. Past the training length the first
        # query can fail on recognising where the block starts rather than on retrieval.
        res["recall_after_first"] = float(by_slot[:, 1:].mean())
    for name in ce_sum:
        if ce_cnt[name]:
            res[f"ce_{name}"] = ce_sum[name] / ce_cnt[name]
    res["n_seqs"] = n_seqs
    return res
