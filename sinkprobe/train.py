"""Train one model on the haystack task and write every diagnostic to JSON.

    python -m sinkprobe.train --variant gate --seed 0 --out results/runs/dev/gate_s0.json

Task knobs and model switches can be overridden from the command line, for
example --task p_noop=0.75 query_skew=4 or --model n_layers=6.

The objective is the answer loss plus lam times the next token loss on every
other position. Both terms are means over their own positions, so the weight
of retrieval does not change with sequence length.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .data import ANSWER, COPY, HaystackTask, TaskConfig
from .metrics import attention_diagnostics, evaluate
from .model import ModelConfig, TinyLM


@dataclass
class RunConfig:
    variant: str = "softmax"
    seed: int = 0
    task: Dict = field(default_factory=dict)
    model: Dict = field(default_factory=dict)
    train_len: int = 256
    steps: int = 3000
    batch: int = 64
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup: int = 150
    weight_decay: float = 0.1
    clip: float = 1.0
    lam: float = 1.0
    eval_lens: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    eval_seqs: int = 512
    collect_seqs: int = 8
    log_every: int = 100
    probe_every: int = 500
    amp: bool = True
    cuda_graph: bool = False
    copy_warmup: int = 0
    save_path: str = ""
    task_kind: str = "haystack"


def autocast(device: str, enabled: bool, cache: bool = True):
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                          dtype=torch.bfloat16, enabled=enabled and device == "cuda",
                          cache_enabled=cache)


def lr_at(step: int, cfg: RunConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embed" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def objective(logits, batch, lam):
    """Masked means rather than boolean indexing, so the step has static shapes."""
    x = batch["tokens"]
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                         x[:, 1:].reshape(-1), reduction="none").view(x.size(0), -1)
    is_ans = (batch["types"][:, :-1] == ANSWER).float()
    rest = 1.0 - is_ans
    ce_ans = (ce * is_ans).sum() / is_ans.sum().clamp_min(1.0)
    ce_rest = (ce * rest).sum() / rest.sum().clamp_min(1.0)
    is_copy = (batch["types"][:, :-1] == COPY).float()
    ce_copy = (ce * is_copy).sum() / is_copy.sum().clamp_min(1.0)
    return ce_ans + lam * ce_rest, ce_ans, ce_rest, ce_copy


def graph_step(model, opt, batch, cfg, device):
    """One full optimiser step with static shapes, safe to capture in a CUDA graph."""
    with autocast(device, cfg.amp, cache=False):
        logits, _ = model(batch["tokens"])
    loss, ce_ans, ce_rest, ce_copy = objective(logits, batch, cfg.lam)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
    opt.step()
    # Detached, so no autograd graph outlives the step (required before capture).
    return tuple(z.detach() for z in (logits, loss, ce_ans, ce_rest, ce_copy, gn))


def run(cfg: RunConfig, device: str = "cuda", verbose: bool = True) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.task_kind == "text":
        from .text_task import TextConfig, TextTask
        tcfg = TextConfig(**cfg.task)
        task = TextTask(tcfg)
        warm_task = task
    else:
        tcfg = TaskConfig(**cfg.task)
        task = HaystackTask(tcfg)
        # Optional warm-up on copy-rich data, identical for every condition, so that
        # induction heads form before the condition-specific data starts.
        warm_task = HaystackTask(TaskConfig(**{**cfg.task, "p_noop": 0.0}))
    mcfg = ModelConfig.from_variant(cfg.variant, vocab_size=tcfg.vocab_size, **cfg.model)
    model = TinyLM(mcfg).to(device)
    use_graph = cfg.cuda_graph and device == "cuda"
    lr_t = torch.tensor(cfg.lr, device=device) if use_graph else cfg.lr
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=lr_t,
                            betas=(0.9, 0.95), fused=(device == "cuda"),
                            capturable=use_graph)

    train_rng = np.random.default_rng(10_000 + cfg.seed)
    probe = task.build(cfg.collect_seqs, cfg.train_len, np.random.default_rng(777), skew=0.0)
    curve, probes = [], []
    graph, static, side, g_out = None, None, None, None
    t0 = time.time()
    model.train()
    for step in range(cfg.steps + 1):
        if step % cfg.probe_every == 0 or step == cfg.steps:
            model.eval()
            with torch.no_grad(), autocast(device, cfg.amp):
                _, st = model(probe["tokens"].to(device), collect=True)
            diag = attention_diagnostics(st, probe["types"])
            keep = {k: diag[k] for k in ("sink_mass", "sink_ratio", "sink_rate",
                                         "sink_noop", "sink_copy", "sink_answer",
                                         "gate_mean", "gate_noop", "gate_copy",
                                         "gate_answer", "virtual_sink") if k in diag}
            keep["step"] = step
            probes.append(keep)
            model.train()
        if step == cfg.steps:
            break
        nb = (warm_task if step < cfg.copy_warmup else task).build(cfg.batch, cfg.train_len, train_rng)
        if use_graph:
            if graph is None and step < 3:
                # Warm up on a side stream before capture, as CUDA graphs require.
                if static is None:
                    static = {k: v.to(device) for k, v in nb.items()}
                    side = torch.cuda.Stream()
                for k in static:
                    static[k].copy_(nb[k])
                lr_t.fill_(lr_at(step, cfg))
                with torch.cuda.stream(side):
                    opt.zero_grad(set_to_none=True)
                    outs = graph_step(model, opt, static, cfg, device)
                torch.cuda.current_stream().wait_stream(side)
                logits, loss, ce_ans, ce_rest, ce_copy, gn = outs
            elif graph is None:
                for k in static:
                    static[k].copy_(nb[k])
                lr_t.fill_(lr_at(step, cfg))
                try:
                    graph = torch.cuda.CUDAGraph()
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.graph(graph):
                        g_out = graph_step(model, opt, static, cfg, device)
                    logits, loss, ce_ans, ce_rest, ce_copy, gn = g_out
                except Exception as exc:                      # fall back to eager steps
                    print(f"CUDA graph capture failed ({exc}); continuing eagerly", flush=True)
                    use_graph, graph = False, None
                    torch.cuda.synchronize()
                    opt.zero_grad(set_to_none=True)
                    logits, loss, ce_ans, ce_rest, ce_copy, gn = graph_step(model, opt, static, cfg, device)
            else:
                for k in static:
                    static[k].copy_(nb[k].pin_memory(), non_blocking=True)
                lr_t.fill_(lr_at(step, cfg))
                graph.replay()
                logits, loss, ce_ans, ce_rest, ce_copy, gn = g_out
            b = static
        else:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            b = {k: v.to(device, non_blocking=True) for k, v in nb.items()}
            with autocast(device, cfg.amp):
                logits, _ = model(b["tokens"])
            loss, ce_ans, ce_rest, ce_copy = objective(logits, b, cfg.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                ap = b["answer_pos"]
                pred = logits.gather(1, ap[..., None].expand(-1, -1, logits.size(-1))).argmax(-1)
                acc = (pred == b["tokens"].gather(1, ap + 1)).float().mean().item()
            rec = {"step": step, "loss": loss.item(), "ce_answer": ce_ans.item(),
                   "ce_rest": ce_rest.item(), "ce_copy": ce_copy.item(), "train_recall": acc,
                   "grad_norm": float(gn), "lr": lr_at(step, cfg),
                   "seconds": round(time.time() - t0, 1)}
            curve.append(rec)
            if verbose:
                print(f"[{cfg.variant} s{cfg.seed}] step {step:5d} loss {rec['loss']:.3f} "
                      f"ce_ans {rec['ce_answer']:.3f} ce_copy {rec['ce_copy']:.3f} recall {acc:.3f} "
                      f"{rec['seconds']:.0f}s", flush=True)
    train_seconds = time.time() - t0
    # Release the CUDA graph, its static buffers and the last batch before the
    # long-context evaluation, which needs that memory.
    graph = static = g_out = side = None
    logits = loss = ce_ans = ce_rest = ce_copy = gn = b = nb = None
    if device == "cuda":
        torch.cuda.empty_cache()

    evals = []
    for i, L in enumerate(cfg.eval_lens):
        rng = np.random.default_rng(50_000 + 97 * i + cfg.seed)
        with autocast(device, cfg.amp):
            res = evaluate(model, task, L, cfg.eval_seqs, rng, device,
                           collect_seqs=cfg.collect_seqs if L <= 1024 else max(2, cfg.collect_seqs // 2))
        evals.append(res)
        if verbose:
            print(f"[{cfg.variant} s{cfg.seed}] len {L:5d} recall {res['recall']:.3f} "
                  f"sink {res.get('sink_mass', float('nan')):.3f} "
                  f"ratio {res.get('sink_ratio', float('nan')):.1f} "
                  f"noop {res.get('sink_noop', float('nan')):.3f} "
                  f"copy {res.get('sink_copy', float('nan')):.3f} "
                  f"gap {res['recency_gap']:+.3f}", flush=True)
        torch.cuda.empty_cache() if device == "cuda" else None

    if cfg.save_path:
        # Weights are kept so that interventions can be run on trained models later.
        torch.save({"state_dict": model.state_dict(), "model_config": mcfg.to_dict(),
                    "task_config": tcfg.to_dict(), "run": asdict(cfg)}, cfg.save_path)
    chance, present = task.floors()
    return {
        "run": asdict(cfg), "model_config": mcfg.to_dict(), "task_config": tcfg.to_dict(),
        "params": model.n_params(), "train_seconds": round(train_seconds, 1),
        "device": device, "torch": torch.__version__,
        "floors": {"chance": chance, "any_present_value": present},
        "curve": curve, "probes": probes, "evals": evals,
    }


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        if v in ("True", "False"):
            v = v == "True"
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="softmax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--train-len", type=int, default=256)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--eval-seqs", type=int, default=512)
    ap.add_argument("--task", nargs="*", help="TaskConfig overrides, key=value")
    ap.add_argument("--model", nargs="*", help="ModelConfig overrides, key=value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--copy-warmup", type=int, default=0)
    ap.add_argument("--task-kind", choices=["haystack", "text"], default="haystack")
    ap.add_argument("--save-model", action="store_true",
                    help="also write the trained weights next to the JSON (.pt)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"exists, skipping: {a.out}")
        return
    cfg = RunConfig(variant=a.variant, seed=a.seed, steps=a.steps, batch=a.batch,
                    lr=a.lr, lam=a.lam, train_len=a.train_len, eval_lens=a.eval_lens,
                    eval_seqs=a.eval_seqs, task=parse_kv(a.task), model=parse_kv(a.model),
                    amp=not a.no_amp, cuda_graph=a.cuda_graph,
                    copy_warmup=a.copy_warmup,
                    save_path=(os.path.splitext(a.out)[0] + ".pt") if a.save_model else "",
                    task_kind=a.task_kind)
    result = run(cfg, device=a.device)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
