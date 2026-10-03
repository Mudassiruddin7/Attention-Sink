"""Variable-gap retrieval: a task that no longer matches one fixed binding width.

In the earlier task the value always sat one token after its key, and the proposed
layer bound keys with a fixed causal convolution of width 4, so the task matched
the method by construction. Here the distance between a key and its value varies
within a sequence, keys can span several tokens, and decoy keys share a prefix
with a real one. A fixed window is then no longer enough, which lets us ask a
sharper question: should the binding span be fixed, or chosen from the content?

Binding variants on the keys and values of a global layer:

    none    no binding
    conv    residual depthwise causal convolution of fixed width (the old block)
    multi   residual taps at several fixed offsets (1, 2, 4, 8, 16)
    dyn     a distribution over offsets 1..W, produced per position and per head
            from the content, so the layer picks which earlier token to bind

Everything here is written so that one run is one function call, every score is
counted over independent inputs, and the answer readout never sees the markers.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, apply_rope
from .model import ModelConfig, TinyLM

OTHER, REST, ANSWER = 0, 1, 2

# Colours shared with the paper's figures, checked for colour-vision deficiency.
PALETTE = {
    "dyn_first": "#2a78d6", "dyn_last": "#e34948", "dyn_last_skip": "#4a3aa7",
    "bkf_conv4": "#eb6834", "bkf_conv16": "#1baf7a", "multi_first": "#eda100",
    "global_first_nobind": "#9a6fb0", "global_last_nobind": "#898781",
}


def _plain(ax):
    """Recessive axes: the data is the only loud thing on the plot."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0d7de")
    ax.tick_params(colors="#57606a", labelsize=8, length=0)
    ax.grid(color="#d0d7de", linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class VarGapConfig:
    n_filler: int = 256          # size of the filler vocabulary
    n_pairs: int = 16            # key-value pairs hidden in the haystack
    n_queries: int = 1           # queries per input; 1 keeps one trial per input
    gap_min: int = 1             # filler tokens between a key and its value
    gap_max: int = 4
    key_len: int = 1             # tokens per key
    n_decoys: int = 0            # keys that share the prefix of a real key
    seg_min: int = 8
    seg_max: int = 24
    p_copy: float = 0.5          # share of filler segments that copy an earlier one
    random_bos: bool = False     # position 0 is a random filler token, not a marker
    bos: int = 0
    qry: int = 1

    @property
    def filler_base(self) -> int:
        return 2

    @property
    def vocab_size(self) -> int:
        return 2 + self.n_filler

    @property
    def query_block(self) -> int:
        return 1 + self.key_len + 1          # marker, key tokens, value

    def to_dict(self) -> Dict:
        return {**asdict(self), "vocab_size": self.vocab_size}


class VarGapTask:
    """Haystack of filler segments with key-value pairs at a variable distance."""

    def __init__(self, cfg: VarGapConfig):
        self.cfg = cfg

    def _one(self, seq_len: int, rng: np.random.Generator):
        c = self.cfg
        body_len = seq_len - 1 - c.n_queries * c.query_block
        if body_len < 4 * (c.key_len + c.gap_max + 1):
            raise ValueError("sequence too short for this task configuration")

        vocab = np.arange(c.filler_base, c.vocab_size)
        n_special = c.n_pairs * (c.key_len + 1) + c.n_decoys * (c.key_len + 1)
        if n_special >= len(vocab) // 2:
            raise ValueError("filler vocabulary too small for this many pairs")
        special = rng.choice(vocab, size=n_special, replace=False)
        keys = special[: c.n_pairs * c.key_len].reshape(c.n_pairs, c.key_len)
        values = special[c.n_pairs * c.key_len: c.n_pairs * (c.key_len + 1)]
        rest = special[c.n_pairs * (c.key_len + 1):]
        allowed = np.setdiff1d(vocab, special)        # filler never repeats a key

        # Decoys share every key token but the last one, so a model that matches
        # on the first token alone answers the wrong pair.
        decoys, decoy_vals = [], []
        if c.n_decoys and c.key_len >= 2:
            src = rng.integers(0, c.n_pairs, size=c.n_decoys)
            for i in range(c.n_decoys):
                d = keys[src[i]].copy()
                d[-1] = rest[i * (c.key_len + 1)]
                decoys.append(d)
                decoy_vals.append(rest[i * (c.key_len + 1) + 1])

        gaps = rng.integers(c.gap_min, c.gap_max + 1, size=c.n_pairs)
        blocks, block_gap, is_real = [], [], []
        for p in range(c.n_pairs):
            g = int(gaps[p])
            between = allowed[rng.integers(0, len(allowed), size=g)]
            blocks.append(np.concatenate([keys[p], between, [values[p]]]))
            block_gap.append(g)
            is_real.append(True)
        for i, d in enumerate(decoys):
            g = int(rng.integers(c.gap_min, c.gap_max + 1))
            between = allowed[rng.integers(0, len(allowed), size=g)]
            blocks.append(np.concatenate([d, between, [decoy_vals[i]]]))
            block_gap.append(g)
            is_real.append(False)

        used = sum(len(b) for b in blocks)
        n_fill = body_len - used
        if n_fill < c.seg_max:
            raise ValueError("sequence too short once the pairs are placed")

        # Filler segments: fresh tokens, or a verbatim copy of an earlier segment
        # so that copying earlier text keeps paying off.
        segs, seg_pred, filled = [], [], 0
        while filled < n_fill:
            room = n_fill - filled
            if segs and rng.random() < c.p_copy:
                src = segs[int(rng.integers(len(segs)))]
                toks = src[: min(len(src), room)].copy()
                pred = np.ones(len(toks), dtype=bool)
                pred[0] = False
            else:
                n = min(int(rng.integers(c.seg_min, c.seg_max + 1)), room)
                toks = allowed[rng.integers(0, len(allowed), size=n)]
                pred = np.zeros(n, dtype=bool)
            segs.append(toks)
            seg_pred.append(pred)
            filled += len(toks)

        slot = rng.integers(0, len(segs) + 1, size=len(blocks))
        order = np.argsort(slot, kind="stable")
        body = np.empty(body_len, dtype=np.int64)
        body_pred = np.zeros(body_len, dtype=bool)
        value_pos = np.full(c.n_pairs, -1, dtype=np.int64)
        pos, j = 0, 0
        for si in range(len(segs) + 1):
            while j < len(blocks) and slot[order[j]] == si:
                bidx = order[j]
                blk = blocks[bidx]
                body[pos:pos + len(blk)] = blk
                if is_real[bidx]:
                    value_pos[bidx] = pos + len(blk) - 1     # index of the value token
                pos += len(blk)
                j += 1
            if si < len(segs):
                n = len(segs[si])
                body[pos:pos + n] = segs[si]
                body_pred[pos:pos + n] = seg_pred[si]
                pos += n

        seq = np.empty(seq_len, dtype=np.int64)
        seq[0] = allowed[rng.integers(0, len(allowed))] if c.random_bos else c.bos
        seq[1:1 + body_len] = body
        value_pos = value_pos + 1                              # shift past position 0

        types = np.full(seq_len, OTHER, dtype=np.int8)
        types[:body_len] = REST                                # predict the next body token
        asked = rng.choice(c.n_pairs, size=c.n_queries, replace=False)
        base = 1 + body_len
        ans_pos = np.empty(c.n_queries, dtype=np.int64)
        for qi, p in enumerate(asked):
            o = base + qi * c.query_block
            seq[o] = c.qry
            seq[o + 1:o + 1 + c.key_len] = keys[p]
            seq[o + 1 + c.key_len] = values[p]
            ans_pos[qi] = o + c.key_len                        # predicts the value
            types[ans_pos[qi]] = ANSWER
        depth = value_pos[asked] / max(body_len, 1)
        return (seq, types, ans_pos, gaps[asked].astype(np.int64), depth.astype(np.float32),
                value_pos, np.asarray(block_gap[:c.n_pairs], dtype=np.int64))

    def build(self, batch_size: int, seq_len: int, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        out = [self._one(seq_len, rng) for _ in range(batch_size)]
        seq, types, ans, gap, dep, vpos, vgap = (np.stack(z) for z in zip(*out))
        return {
            "tokens": torch.from_numpy(seq),
            "types": torch.from_numpy(types.astype(np.int64)),
            "answer_pos": torch.from_numpy(ans),
            "answer_gap": torch.from_numpy(gap),
            "answer_depth": torch.from_numpy(dep),
            "value_pos": torch.from_numpy(vpos),
            "value_gap": torch.from_numpy(vgap),
        }

    def reserved(self) -> Tuple[int, int]:
        return (self.cfg.bos, self.cfg.qry)

    def chance(self) -> float:
        return 1.0 / (self.cfg.n_filler - self.cfg.n_pairs * (self.cfg.key_len + 1))


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

class Binder(nn.Module):
    """Mixes earlier tokens into a key or value vector.

    conv and multi use fixed offsets. dyn reads the layer input and produces, per
    position and per head, a distribution over offsets 1..W together with a gate,
    so the span is chosen from the content instead of being set in advance.
    """

    def __init__(self, d_model: int, n_heads: int, mode: str = "conv",
                 width: int = 4, offsets: Sequence[int] = (1, 2, 4, 8, 16)):
        super().__init__()
        assert mode in ("none", "conv", "multi", "dyn")
        self.mode, self.h, self.d = mode, n_heads, d_model
        self.width = width
        self.offsets = tuple(offsets)
        if mode == "conv":
            self.conv = nn.Conv1d(d_model, d_model, width, groups=d_model,
                                  padding=width - 1, bias=False)
        elif mode == "multi":
            self.taps = nn.Parameter(torch.zeros(len(self.offsets), d_model))
        elif mode == "dyn":
            self.sel = nn.Linear(d_model, n_heads * width, bias=False)
            self.gate = nn.Linear(d_model, n_heads, bias=True)
            nn.init.normal_(self.sel.weight, std=0.02)
            nn.init.normal_(self.gate.weight, std=0.02)
            nn.init.zeros_(self.gate.bias)

    def forward(self, z: torch.Tensor, x: torch.Tensor, collect: bool = False):
        """z is the key or value stream (B, T, D); x is the layer input."""
        if self.mode == "none":
            return z, None
        b, t, d = z.shape
        if self.mode == "conv":
            return z + self.conv(z.transpose(1, 2))[..., :t].transpose(1, 2), None
        if self.mode == "multi":
            out = z
            for i, o in enumerate(self.offsets):
                out = out + self.taps[i] * F.pad(z, (0, 0, o, 0))[:, :t]
            return out, None
        w = self.width
        p = self.sel(x).view(b, t, self.h, w).softmax(-1)                   # offsets 1..W
        g = torch.sigmoid(self.gate(x)).view(b, t, self.h, 1)
        # window[..., i] holds the token at offset (W - i); flip to offsets 1..W
        window = F.pad(z, (0, 0, w, 0))[:, :t + w - 1].unfold(1, w, 1).flip(-1)
        window = window.reshape(b, t, self.h, d // self.h, w)
        mix = torch.einsum("bthdw,bthw->bthd", window, p * g)
        return z + mix.reshape(b, t, d), (p.detach() if collect else None)


class BoundAttention(nn.Module):
    """Causal softmax attention whose keys and values are bound to earlier tokens.

    embed_skip gives the layer a direct path to the token embeddings, which tests
    whether a late global layer learns slowly only because it reads the output of
    the linear layers below it.
    """

    def __init__(self, d_model: int, n_heads: int, rope: bool = False, qk_norm: bool = True,
                 gate: bool = True, logn_ref: int = 256, bind: str = "conv", width: int = 4,
                 offsets: Sequence[int] = (1, 2, 4, 8, 16), embed_skip: bool = False,
                 emb_store: Dict | None = None):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.rope, self.logn_ref = rope, logn_ref
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.dh) if qk_norm else None
        self.k_norm = RMSNorm(self.dh) if qk_norm else None
        self.gate = nn.Linear(d_model, d_model, bias=False) if gate else None
        self.bind_k = Binder(d_model, n_heads, bind, width, offsets)
        self.bind_v = Binder(d_model, n_heads, bind, width, offsets)
        self.embed_skip = embed_skip
        self.emb_store = emb_store
        if embed_skip:
            self.emb_norm = RMSNorm(d_model)
            self.emb_scale = nn.Parameter(torch.ones(1))
        for m in (self.qkv, self.out, self.gate):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, collect: bool = False):
        b, t, d = x.shape
        src = x
        if self.embed_skip and self.emb_store is not None and "e" in self.emb_store:
            src = x + self.emb_scale * self.emb_norm(self.emb_store["e"].to(x.dtype))
        q, k, v = self.qkv(src).split(d, dim=-1)
        k, off = self.bind_k(k, src, collect=collect)
        v, _ = self.bind_v(v, src, collect=False)
        q, k, v = (z.reshape(b, t, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            q, k = apply_rope(q), apply_rope(k)
        if self.logn_ref and t > self.logn_ref:
            visible = torch.arange(1, t + 1, device=x.device, dtype=torch.float32)
            lam = torch.clamp(visible.log() / math.log(self.logn_ref), min=1.0)
            q = q * lam.to(q.dtype).view(1, 1, t, 1)

        aux = {}
        if collect:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            p = torch.softmax(scores.float().masked_fill(~mask, float("-inf")), dim=-1)
            o = p.to(v.dtype) @ v
            aux["a0"] = p[..., 0].cpu()                      # attention on position 0
            if off is not None:
                aux["offsets"] = off.cpu()                   # (B, T, H, W)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(b, t, d)
        if self.gate is not None:
            o = o * torch.sigmoid(self.gate(x))
        return self.out(o), aux


DESIGNS: Dict[str, Dict] = {
    # layouts repeat over depth: S is a global softmax layer, L a gated delta rule layer
    "global_last_nobind":  dict(layout="LLLS", bind="none"),
    "global_last_conv4":   dict(layout="LLLS", bind="conv", width=4),
    "global_first_nobind": dict(layout="SLLL", bind="none"),
    "bkf_conv4":           dict(layout="SLLL", bind="conv", width=4),     # the earlier proposal
    "bkf_conv16":          dict(layout="SLLL", bind="conv", width=16),    # same idea, wider window
    "multi_first":         dict(layout="SLLL", bind="multi"),
    "dyn_first":           dict(layout="SLLL", bind="dyn", width=16),     # the proposal here
    "dyn_last":            dict(layout="LLLS", bind="dyn", width=16),
    "dyn_last_skip":       dict(layout="LLLS", bind="dyn", width=16, embed_skip=True),
    "attention_only_dyn":  dict(layout="S", bind="dyn", width=16),
}

for _i in range(8):                                    # where the single global layer sits
    DESIGNS[f"pos{_i}_dyn"] = dict(layout="L" * _i + "S" + "L" * (7 - _i), bind="dyn", width=16)


def build_model(design: str, vocab_size: int, d_model: int = 128, n_layers: int = 8,
                n_heads: int = 4, logn_ref: int = 256, rope: bool = False):
    spec = dict(DESIGNS[design])
    cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, d_ff=3 * d_model, layout=spec["layout"],
                      rope=rope, gate=True, kv_conv=0, logn_ref=logn_ref)
    model = TinyLM(cfg)
    store: Dict[str, torch.Tensor] = {}
    if spec.get("embed_skip"):
        model.embed.register_forward_hook(lambda m, i, o: store.__setitem__("e", o))
    for blk in model.blocks:
        if blk.kind == "S":
            blk.mix = BoundAttention(d_model, n_heads, rope=rope, gate=True, logn_ref=logn_ref,
                                     bind=spec["bind"], width=spec.get("width", 4),
                                     offsets=spec.get("offsets", (1, 2, 4, 8, 16)),
                                     embed_skip=spec.get("embed_skip", False), emb_store=store)
    return model, cfg


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    design: str = "dyn_first"
    seed: int = 0
    steps: int = 3000
    batch: int = 64
    lr: float = 3e-3
    warmup: int = 150
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    clip: float = 1.0
    lam: float = 1.0
    train_len: int = 256
    d_model: int = 128
    n_layers: int = 8
    n_heads: int = 4
    eval_every: int = 100
    eval_inputs: int = 512
    eval_lens: Tuple[int, ...] = (256, 1024, 4096)
    learned_at: float = 0.9
    amp: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def _autocast(device: str, enabled: bool):
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                          dtype=torch.bfloat16, enabled=enabled and device == "cuda")


def objective(logits, batch, lam: float):
    x = batch["tokens"]
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                         x[:, 1:].reshape(-1), reduction="none").view(x.size(0), -1)
    kinds = batch["types"][:, :-1]
    is_ans = (kinds == ANSWER).float()
    is_rest = (kinds == REST).float()
    ce_ans = (ce * is_ans).sum() / is_ans.sum().clamp_min(1.0)
    ce_rest = (ce * is_rest).sum() / is_rest.sum().clamp_min(1.0)
    return ce_ans + lam * ce_rest, ce_ans, ce_rest


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% interval for a proportion, which a normal interval gets wrong near 0 and 1."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


@torch.no_grad()
def exact_match(model, task: VarGapTask, length: int, n_inputs: int, rng, device: str,
                batch: int = 64, amp: bool = True) -> Dict:
    """Exact match over independent inputs. The markers are removed from the readout."""
    model.eval()
    hits, total, by_gap = 0, 0, {}
    bos, qry = task.reserved()
    while total < n_inputs:
        n = min(batch, n_inputs - total)
        b = task.build(n, length, rng)
        tok = b["tokens"].to(device)
        with _autocast(device, amp):
            logits, _ = model(tok)
        ap = b["answer_pos"].to(device)
        sel = logits.gather(1, ap[..., None].expand(-1, -1, logits.size(-1))).float()
        sel[..., bos] = -float("inf")
        sel[..., qry] = -float("inf")
        pred = sel.argmax(-1).cpu()
        gold = b["tokens"].gather(1, b["answer_pos"] + 1)
        ok = (pred == gold)
        hits += int(ok.sum())
        total += int(ok.numel())
        for g, o in zip(b["answer_gap"].reshape(-1).tolist(), ok.reshape(-1).tolist()):
            a, c = by_gap.get(g, (0, 0))
            by_gap[g] = (a + int(o), c + 1)
    lo, hi = wilson(hits, total)
    model.train()
    return {"em": hits / total, "n": total, "lo": lo, "hi": hi,
            "by_gap": {int(g): {"em": a / c, "n": c} for g, (a, c) in sorted(by_gap.items())}}


def train_one(cfg: TrainConfig, task_cfg: VarGapConfig, device: str = "cuda",
              eval_task_cfg: VarGapConfig | None = None, verbose: bool = True) -> Dict:
    """One run: train, track held-out exact match, then score at several lengths.

    The held-out set comes from the training distribution, so the learning step is
    not read off an easier mix of data.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    task = VarGapTask(task_cfg)
    eval_task = VarGapTask(eval_task_cfg or task_cfg)
    model, mcfg = build_model(cfg.design, task_cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads)
    model = model.to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": other, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95), fused=(device == "cuda"))

    train_rng = np.random.default_rng(10_000 + cfg.seed)
    curve, learned_step = [], None
    t0 = time.time()
    model.train()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            held = exact_match(model, eval_task, cfg.train_len, 256,
                               np.random.default_rng(777), device, amp=cfg.amp)
            curve.append({"step": step, "em": held["em"], "lo": held["lo"], "hi": held["hi"]})
            if learned_step is None and held["em"] >= cfg.learned_at:
                learned_step = step
            if verbose and (step % (cfg.eval_every * 5) == 0 or step == cfg.steps):
                print(f"[{cfg.design} s{cfg.seed}] step {step:5d} held-out EM {held['em']:.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        if step == cfg.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        b = {k: v.to(device, non_blocking=True) for k, v in
             task.build(cfg.batch, cfg.train_len, train_rng).items()}
        with _autocast(device, cfg.amp):
            logits, _ = model(b["tokens"])
        loss, ce_ans, ce_rest = objective(logits, b, cfg.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
    train_seconds = time.time() - t0

    lengths = {}
    for i, L in enumerate(cfg.eval_lens):
        rng = np.random.default_rng(50_000 + 97 * i + cfg.seed)
        n = cfg.eval_inputs if L <= 4096 else max(128, cfg.eval_inputs // 4)
        bs = 64 if L <= 1024 else (16 if L <= 4096 else 4)
        lengths[L] = exact_match(model, eval_task, L, n, rng, device, batch=bs, amp=cfg.amp)
        if verbose:
            r = lengths[L]
            print(f"[{cfg.design} s{cfg.seed}] len {L:6d} EM {r['em']:.3f} "
                  f"[{r['lo']:.3f}, {r['hi']:.3f}] over {r['n']} inputs", flush=True)
    result = {"config": cfg.to_dict(), "task": task_cfg.to_dict(),
              "eval_task": (eval_task_cfg or task_cfg).to_dict(),
              "model_config": mcfg.to_dict(), "params": model.n_params(),
              "learned_step": learned_step, "curve": curve,
              "lengths": {str(k): v for k, v in lengths.items()},
              "train_seconds": round(train_seconds, 1), "device": device}
    return result, model


# ---------------------------------------------------------------------------
# Natural text, one size up
# ---------------------------------------------------------------------------

@torch.no_grad()
def _text_scores(model, task, length: int, n_seq: int, batch: int, rng, device: str, amp: bool):
    """Copy accuracy on the probe (mean per input) and bits per byte on plain windows."""
    model.eval()
    per_input, bpb = [], []
    done = 0
    while done < n_seq:
        n = min(batch, n_seq - done)
        b = task.build(n, length, rng, skew=0.0)                 # any skew builds the probe
        tok = b["tokens"].to(device)
        with _autocast(device, amp):
            logits, _ = model(tok)
        ap = b["answer_pos"].to(device)
        pred = logits.gather(1, ap[..., None].expand(-1, -1, logits.size(-1))).argmax(-1)
        gold = tok.gather(1, ap + 1)
        per_input += (pred == gold).float().mean(-1).cpu().tolist()
        w = task.build(n, length, rng)                           # plain windows for bits per byte
        wt = w["tokens"].to(device)
        with _autocast(device, amp):
            wl, _ = model(wt)
        ce = F.cross_entropy(wl[:, :-1].float().reshape(-1, wl.size(-1)), wt[:, 1:].reshape(-1))
        bpb.append(ce.item() / math.log(2))
        done += n
    model.train()
    mean, lo, hi = t_interval(per_input)
    return {"copy_acc": mean, "lo": lo, "hi": hi, "n_inputs": len(per_input),
            "bits_per_byte": float(np.mean(bpb))}


def train_text(design: str, seed: int, train_path: str, eval_path: str, steps: int = 2000,
               batch: int = 16, seq_len: int = 2048, d_model: int = 512, n_layers: int = 12,
               n_heads: int = 8, lr: float = 1e-3, warmup: int = 200, device: str = "cuda",
               amp: bool = True, span: int = 32, lead: int = 8, n_queries: int = 8,
               eval_lens: Sequence[int] = (2048, 4096, 8192), eval_seqs: int = 64,
               verbose: bool = True):
    """One size up: byte-level natural text, plain next-token loss, copy probe at length.

    The copy probe repeats a span taken from the passage, so it measures retrieval
    on real text. The control condition repeats a span that is not in the context,
    which is what language statistics alone can reach.
    """
    from .text_task import TextConfig, TextTask
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = dict(train_path=train_path, eval_path=eval_path, span=span, lead=lead, n_queries=n_queries)
    task = TextTask(TextConfig(**base))
    control = TextTask(TextConfig(**base, control=True))
    model, mcfg = build_model(design, task.cfg.vocab_size, d_model, n_layers, n_heads,
                              logn_ref=seq_len)
    model = model.to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": other, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=(device == "cuda"))
    rng = np.random.default_rng(20_000 + seed)
    curve, t0 = [], time.time()
    model.train()
    for step in range(steps):
        frac = (step - warmup) / max(1, steps - warmup)
        now = lr * (step + 1) / warmup if step < warmup else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = now
        tok = task.build(batch, seq_len, rng)["tokens"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits, _ = model(tok)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               tok[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            curve.append({"step": step, "bits_per_byte": loss.item() / math.log(2),
                          "seconds": round(time.time() - t0, 1)})
            if verbose:
                print(f"[{design} s{seed}] step {step:5d} bits/byte {curve[-1]['bits_per_byte']:.3f} "
                      f"{curve[-1]['seconds']:.0f}s", flush=True)
    out = {}
    for L in eval_lens:
        bs = max(1, batch // max(1, L // seq_len))
        out[str(L)] = {
            "probe": _text_scores(model, task, L, eval_seqs, bs, np.random.default_rng(99 + L), device, amp),
            "control": _text_scores(model, control, L, max(16, eval_seqs // 2), bs,
                                    np.random.default_rng(199 + L), device, amp),
        }
        if verbose:
            p, c = out[str(L)]["probe"], out[str(L)]["control"]
            print(f"[{design} s{seed}] len {L:6d} copy {p['copy_acc']:.3f} "
                  f"[{p['lo']:.3f}, {p['hi']:.3f}] control {c['copy_acc']:.3f} "
                  f"bits/byte {p['bits_per_byte']:.3f}", flush=True)
    return {"design": design, "seed": seed, "model_config": mcfg.to_dict(),
            "params": model.n_params(), "seq_len": seq_len, "steps": steps,
            "curve": curve, "evals": out, "train_seconds": round(time.time() - t0, 1)}, model


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def sink_stats(model, task: VarGapTask, length: int, n_seq: int, rng, device: str,
               amp: bool = True) -> Dict:
    """Attention on position 0, measured at the length it is reported for.

    With random_bos set on the task, position 0 is an ordinary filler token, which
    separates a sink from attention to one fixed marker.
    """
    model.eval()
    b = task.build(n_seq, length, rng)
    with _autocast(device, amp):
        _, stats = model(b["tokens"].to(device), collect=True)
    vals = [a["a0"].float() for a in stats["layers"] if "a0" in a]
    model.train()
    if not vals:
        return {}
    a0 = torch.cat([v.reshape(-1) for v in vals])
    even = float(np.mean([1.0 / (t + 1) for t in range(1, length)]))
    return {"sink_mass": a0.mean().item(), "even_reference": even,
            "sink_ratio": a0.mean().item() / even, "length": length,
            "random_bos": task.cfg.random_bos}


@torch.no_grad()
def offset_report(model, task: VarGapTask, length: int, n_seq: int, rng, device: str,
                  amp: bool = True) -> Dict:
    """What the dynamic binder picks at value positions, against the true distance.

    A value token sits gap + 1 tokens after the last token of its key, so a binder
    that ties a value to its key should put its mass on that offset.
    """
    model.eval()
    b = task.build(n_seq, length, rng)
    with _autocast(device, amp):
        _, stats = model(b["tokens"].to(device), collect=True)
    model.train()
    first = next((a for a in stats["layers"] if "offsets" in a), None)
    if first is None:
        return {}
    p = first["offsets"].float()                       # (B, T, H, W)
    rows = []
    for i in range(p.shape[0]):
        for pos, gap in zip(b["value_pos"][i].tolist(), b["value_gap"][i].tolist()):
            if 0 <= pos < p.shape[1]:
                dist = p[i, pos].mean(0)               # mean over heads
                rows.append({"true_offset": int(gap) + 1,
                             "picked_offset": int(dist.argmax()) + 1,
                             "mass_on_true": float(dist[min(int(gap), dist.numel() - 1)])})
    if not rows:
        return {}
    true = np.array([r["true_offset"] for r in rows], dtype=float)
    pick = np.array([r["picked_offset"] for r in rows], dtype=float)
    agree = float((true == pick).mean())
    corr = float(np.corrcoef(true, pick)[0, 1]) if true.std() > 0 and pick.std() > 0 else float("nan")
    return {"n": len(rows), "agreement": agree, "correlation": corr,
            "mean_mass_on_true": float(np.mean([r["mass_on_true"] for r in rows])),
            "rows": rows[:2000]}


@torch.no_grad()
def ablate_global_layer(model, task: VarGapTask, length: int, n_inputs: int, rng, device: str,
                        which: int = 0, amp: bool = True) -> Dict:
    """Zero the output of one global layer at test time to locate the lookup."""
    idx = [i for i, blk in enumerate(model.blocks) if blk.kind == "S"]
    if which >= len(idx):
        return {}
    target = model.blocks[idx[which]]
    original = target.mix.forward

    def zeroed(x, collect=False):
        out, aux = original(x, collect=collect)
        return torch.zeros_like(out), aux

    target.mix.forward = zeroed
    try:
        res = exact_match(model, task, length, n_inputs, rng, device, amp=amp)
    finally:
        del target.mix.forward                      # back to the class method
    return {"layer": idx[which], "em": res["em"], "n": res["n"], "lo": res["lo"], "hi": res["hi"]}


# ---------------------------------------------------------------------------
# Statistics over seeds
# ---------------------------------------------------------------------------

def t_interval(values: Sequence[float], conf: float = 0.95) -> Tuple[float, float, float]:
    """Mean and Student interval; seeds are the unit, not inputs."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return (float(v[0]), float("nan"), float("nan"))
    try:
        from scipy import stats as st
        t = float(st.t.ppf(0.5 + conf / 2, v.size - 1))
    except Exception:
        t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(v.size, 2.09)
    half = t * v.std(ddof=1) / math.sqrt(v.size)
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def paired_sign_test(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Two-sided sign test on seed-matched pairs; ties are dropped."""
    wins = sum(1 for x, y in zip(a, b) if y > x)
    losses = sum(1 for x, y in zip(a, b) if y < x)
    n = wins + losses
    if n == 0:
        return {"wins": 0, "losses": 0, "p": 1.0}
    from math import comb
    k = min(wins, losses)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"wins": wins, "losses": losses, "p": p}


def run_job(spec: Dict) -> Dict:
    """Train one configuration and run every probe while the model is still in memory.

    The whole job is described by a plain dict so it can be sent to a worker process,
    and a finished job is never repeated: the JSON on disk is the cache.
    """
    import json as _json
    from pathlib import Path as _Path

    out_dir = _Path(spec.get("out_dir", "results_fast"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec['tag']}.json"
    if path.exists():
        return _json.loads(path.read_text(encoding="utf-8"))

    device = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    task_cfg = VarGapConfig(**spec["task"])
    cfg = TrainConfig(**spec["train"])
    started = time.time()
    res, model = train_one(cfg, task_cfg, device=device, verbose=spec.get("verbose", False))

    want = spec.get("probes", {}) or {}
    probes: Dict = {}
    probe_task = VarGapTask(task_cfg)
    rng = np.random.default_rng(900 + cfg.seed)
    if want.get("offsets"):
        report = offset_report(model, probe_task, cfg.train_len, 8, rng, device)
        report.pop("rows", None)
        probes["offsets"] = report
    if want.get("ablation"):
        probes["ablate_first_global"] = ablate_global_layer(model, probe_task, cfg.train_len,
                                                            256, rng, device)
    if want.get("sink"):
        fields = {k: v for k, v in task_cfg.to_dict().items() if k != "vocab_size"}
        probes["sink"] = []
        for length, random_bos in want["sink"]:
            marked = VarGapTask(VarGapConfig(**{**fields, "random_bos": bool(random_bos)}))
            probes["sink"].append(sink_stats(model, marked, int(length), 4,
                                             np.random.default_rng(5), device))
    res.update({"tag": spec["tag"], "design": cfg.design, "gap_max": task_cfg.gap_max,
                "seed": cfg.seed, "probes": probes,
                "job_seconds": round(time.time() - started, 1)})
    path.write_text(_json.dumps(res), encoding="utf-8")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def _cli():
    """One job per process: python -m sinkprobe.vargap --spec job.json

    Running each job in its own process keeps CUDA state clean and lets a notebook
    schedule several at once without touching multiprocessing.
    """
    import argparse
    import json as _json

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to a JSON job specification")
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = _json.load(fh)
    out = run_job(spec)
    print(f"done {spec['tag']} learned={out.get('learned_step')} "
          f"em256={out.get('lengths', {}).get('256', {}).get('em')}", flush=True)


def _design_of(record: Dict) -> str:
    return record.get("design") or record.get("config", {}).get("design", "?")


def _offsets_of(record: Dict):
    return record.get("offsets") or (record.get("probes") or {}).get("offsets")


def figure_gap(out_dir, stem: str = "fig_gap"):
    """Exact match against the key-value distance, from the stage A run files."""
    import json as _json
    from pathlib import Path as _Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _Path(out_dir)
    rows = [_json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("A_*.json"))]
    rows = [r for r in rows if "lengths" in r and "task" in r]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for design in sorted({r["config"]["design"] for r in rows}):
        gaps = sorted({r["task"]["gap_max"] for r in rows if r["config"]["design"] == design})
        mid, lo, hi = [], [], []
        for g in gaps:
            vals = [r["lengths"]["256"]["em"] for r in rows
                    if r["config"]["design"] == design and r["task"]["gap_max"] == g]
            mid.append(float(np.mean(vals)))
            lo.append(float(np.min(vals)))
            hi.append(float(np.max(vals)))
        colour = PALETTE.get(design, "#898781")
        ax.plot(gaps, mid, "-o", color=colour, linewidth=1.6, markersize=4.5, label=design)
        ax.fill_between(gaps, lo, hi, color=colour, alpha=0.12, linewidth=0)
    all_gaps = sorted({r["task"]["gap_max"] for r in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_gaps)
    ax.set_xticklabels(["1" if g == 1 else f"1-{g}" for g in all_gaps])
    ax.minorticks_off()
    ax.set_xlabel("distance between a key and its value (tokens)", fontsize=8, color="#57606a")
    ax.set_ylabel("exact match", fontsize=8, color="#57606a")
    ax.set_ylim(-0.02, 1.02)
    _plain(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_dir / f"{stem}.png"


def figure_offsets(out_dir, stem: str = "fig_offsets"):
    """How often the dynamic binder picked the distance the task actually used."""
    import json as _json
    from pathlib import Path as _Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _Path(out_dir)
    rows = [_json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]
    rows = [r for r in rows if (_offsets_of(r) or {}).get("agreement") is not None]
    if not rows:
        return None
    order = np.argsort([_offsets_of(r)["agreement"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 0.32 * len(rows) + 1.2))
    ax.barh(np.arange(len(rows)), [_offsets_of(rows[i])["agreement"] for i in order],
            color=[PALETTE.get(_design_of(rows[i]), "#898781") for i in order], height=0.7)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{_design_of(rows[i])} s{rows[i].get('seed', 0)}" for i in order], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of values where the picked offset is the true one", fontsize=8, color="#57606a")
    _plain(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_dir / f"{stem}.png"


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm correction, so a page of comparisons is not read as a page of findings."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


if __name__ == "__main__":
    _cli()
