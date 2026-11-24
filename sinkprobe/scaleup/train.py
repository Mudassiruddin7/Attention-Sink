"""Training at scale, on one GPU, in sessions that can be cut off.

Every run checkpoints, so a 12-hour limit costs time but no work. Batches mix plain
text with task items: the model has to stay a language model while learning to
retrieve, which is the setting the reviewer asked about and also the only way a
bits-per-byte number means anything.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .arch import ArchConfig, SIZES, build
from .corpus import BOS, QRY, SEP, ByteWindows
from .tasks import (ANSWER, IGNORE, REST, LongContextTask, TaskConfig, easy_config,
                    study_config)

LR_BY_SIZE = {"xs": 3e-3, "s": 2e-3, "m": 1.2e-3, "l": 8e-4, "xl": 5e-4}


@dataclass
class TrainConfig:
    size: str = "s"
    design: str = "dyn_first"
    seed: int = 0
    seq_len: int = 1024
    batch: int = 16
    accum: int = 1
    steps: int = 4000
    lr: float = 0.0                      # 0 means pick by size
    warmup: int = 200
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    clip: float = 1.0
    task_rate: float = 0.5               # share of batches that are task items
    task_kinds: Tuple[str, ...] = ("kv_vargap", "kv_multikey", "multi_hop",
                                   "freq_sym", "span_copy")
    gap_max: int = 16
    train_queries: int = 8               # queries per training item, for a denser signal
    eval_every: int = 500
    eval_items: int = 128
    ckpt_every: int = 500
    amp: bool = True
    easy: bool = False                   # the gate trains on the easiest rung
    dense_early: bool = True             # dense readings over the first quarter
    probe_kind: str = "kv_vargap"        # task the curve tracks

    def resolved_lr(self) -> float:
        return self.lr or LR_BY_SIZE.get(self.size, 1e-3)

    def tokens_per_step(self) -> int:
        return self.batch * self.accum * self.seq_len

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["task_kinds"] = list(self.task_kinds)
        d["lr_used"] = self.resolved_lr()
        return d


def lr_at(step: int, cfg: TrainConfig) -> float:
    lr = cfg.resolved_lr()
    if step < cfg.warmup:
        return lr * (step + 1) / cfg.warmup
    frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def _autocast(device: str, enabled: bool):
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                          dtype=torch.bfloat16, enabled=enabled and device == "cuda")


def objective(logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple:
    """Next-token loss on text, and a separate mean on the answer positions."""
    x = batch["tokens"]
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                         x[:, 1:].reshape(-1), reduction="none").view(x.size(0), -1)
    kinds = batch["types"][:, :-1]
    is_ans = (kinds == ANSWER).float()
    is_rest = (kinds == REST).float()
    ce_ans = (ce * is_ans).sum() / is_ans.sum().clamp_min(1.0)
    ce_rest = (ce * is_rest).sum() / is_rest.sum().clamp_min(1.0)
    has_ans = is_ans.sum() > 0
    loss = ce_rest + (ce_ans if has_ans else 0.0 * ce_rest)
    return loss, ce_ans, ce_rest


class Mixture:
    """Draws either a plain text batch or one task batch, in a fixed rotation."""

    def __init__(self, cfg: TrainConfig, windows: ByteWindows, rng: np.random.Generator):
        self.cfg, self.windows, self.rng = cfg, windows, rng
        self.tasks: List[LongContextTask] = []
        for kind in cfg.task_kinds:
            tcfg = easy_config(kind) if cfg.easy else study_config(kind, gap_max=cfg.gap_max)
            tcfg.n_queries = cfg.train_queries if kind in ("kv_vargap", "kv_multikey") else 1
            task = LongContextTask(tcfg, windows)
            if task.required_length() <= cfg.seq_len:
                self.tasks.append(task)
        self.turn = 0

    def skipped(self) -> List[str]:
        keep = {t.cfg.kind for t in self.tasks}
        return [k for k in self.cfg.task_kinds if k not in keep]

    def next_batch(self) -> Dict[str, torch.Tensor]:
        if self.tasks and self.rng.random() < self.cfg.task_rate:
            task = self.tasks[self.turn % len(self.tasks)]
            self.turn += 1
            return task.build(self.cfg.batch, self.cfg.seq_len, self.rng)
        text = self.windows.batch(self.cfg.batch, self.cfg.seq_len, self.rng)
        text[:, 0] = BOS
        types = np.full_like(text, REST)
        types[:, 0] = IGNORE
        return {"tokens": torch.from_numpy(text), "types": torch.from_numpy(types),
                "answer_pos": torch.zeros((self.cfg.batch, 0), dtype=torch.long),
                "answer_gap": torch.zeros((self.cfg.batch, 0), dtype=torch.long),
                "answer_depth": torch.zeros((self.cfg.batch, 0), dtype=torch.float32)}


@torch.no_grad()
def quick_accuracy(model, task: LongContextTask, length: int, n_items: int, device: str,
                   batch: int = 32, amp: bool = True,
                   rng: Optional[np.random.Generator] = None,
                   reach: Optional[int] = None):
    """Exact match on held-out items, markers barred from the prediction.

    With `reach` given, the same pass also returns accuracy with the key inside the
    binder's window and with it beyond, which costs nothing extra and is what tells a
    short-range shortcut apart from a lookup while a run is still going.
    """
    model.eval()
    rng = rng or np.random.default_rng(0)
    hits = seen = 0
    k_in = n_in = k_out = n_out = 0
    while seen < n_items:
        n = min(batch, n_items - seen)
        b = task.build(n, length, rng)
        tok = b["tokens"].to(device)
        with _autocast(device, amp):
            logits, _ = model(tok)
        logits = logits.float()
        logits[..., [BOS, QRY, SEP]] = -float("inf")
        pos = b["answer_pos"].to(device)
        pred = logits.gather(1, pos[..., None].expand(-1, -1, logits.size(-1))).argmax(-1)
        want = tok.gather(1, pos + 1)
        ok = (pred == want)
        hits += int(ok.sum())
        seen += int(pred.numel())        # scored positions, not items: one item can hold several
        if reach is not None:
            inside = (b["answer_gap"].to(device) <= reach)
            k_in += int((ok & inside).sum())
            n_in += int(inside.sum())
            k_out += int((ok & ~inside).sum())
            n_out += int((~inside).sum())
    model.train()
    acc = hits / max(1, seen)
    if reach is None:
        return acc, None, None
    return (acc, (k_in / n_in) if n_in else None, (k_out / n_out) if n_out else None)


def eval_steps(total: int, n_early: int = 20, early_frac: float = 0.25,
               n_late: int = 12) -> List[int]:
    """Evaluation points: dense over the first quarter, sparse after.

    Retrieval appeared inside one evaluation interval in the earlier runs, and at the
    resolution used there that interval was a twelfth of the whole budget. Saying whether
    the change is a step or a steep slope needs readings close together where it happens,
    and costs almost nothing because evaluation is a fraction of a step.
    """
    cut = max(1, int(total * early_frac))
    early = {int(round(i * cut / n_early)) for i in range(n_early + 1)}
    late = {int(round(cut + i * (total - cut) / n_late)) for i in range(n_late + 1)}
    return sorted(s for s in (early | late | {0, total}) if 0 <= s <= total)


@torch.no_grad()
def bits_per_byte(model, windows: ByteWindows, length: int, n_windows: int, device: str,
                  batch: int = 8, amp: bool = True,
                  rng: Optional[np.random.Generator] = None) -> float:
    """Language modelling cost on held-out text, in bits per byte."""
    model.eval()
    rng = rng or np.random.default_rng(1)
    total, count = 0.0, 0
    done = 0
    while done < n_windows:
        n = min(batch, n_windows - done)
        text = windows.batch(n, length, rng)
        text[:, 0] = BOS
        tok = torch.from_numpy(text).to(device)
        with _autocast(device, amp):
            logits, _ = model(tok)
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                             tok[:, 1:].reshape(-1), reduction="sum")
        total += float(ce)
        count += tok[:, 1:].numel()
        done += n
    model.train()
    return total / max(1, count) / math.log(2)


def train(cfg: TrainConfig, train_windows: ByteWindows, val_windows: ByteWindows,
          device: str = "cuda", ckpt_dir: Optional[Path] = None, verbose: bool = True,
          store=None, tag: str = "", ckpt_sync_mb: int = 200,
          ckpt_sync_every: int = 2) -> Tuple[Dict, torch.nn.Module]:
    """Train one model, resuming from a checkpoint when one is there.

    The curve is sent to the store at every evaluation, so a session that dies part way
    still leaves the measurement that matters: when, and whether, retrieval appeared.
    A small enough checkpoint goes with it, which lets a cut-off run resume in the next
    session instead of starting over.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    acfg = ArchConfig(size=cfg.size, design=cfg.design, logn_ref=cfg.seq_len)
    model, mcfg = build(acfg)
    model = model.to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": other, "weight_decay": 0.0}],
                            lr=cfg.resolved_lr(), betas=(0.9, 0.95),
                            fused=(device == "cuda"))

    start_step, curve = 0, []
    ckpt_path = (ckpt_dir / "state.pt") if ckpt_dir else None
    if ckpt_path and not ckpt_path.exists() and store is not None and tag:
        if getattr(store, "get", None) and store.get(f"{tag}.state.pt", ckpt_path):
            print(f"pulled an unfinished {tag} back from the store", flush=True)
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step, curve = state["step"], state["curve"]
        if verbose:
            print(f"resumed {cfg.design}/{cfg.size} at step {start_step}", flush=True)

    rng = np.random.default_rng(10_000 + cfg.seed + start_step)
    mix = Mixture(cfg, train_windows, rng)
    probe_cfg = (easy_config(cfg.probe_kind) if cfg.easy
                 else study_config(cfg.probe_kind, gap_max=cfg.gap_max))
    probe = LongContextTask(probe_cfg, val_windows)
    if probe.required_length() > cfg.seq_len:          # fall back rather than fail
        probe = LongContextTask(easy_config(cfg.probe_kind), val_windows)
        if probe.required_length() > cfg.seq_len:
            probe = None
    from .arch import binder_reach
    reach = binder_reach(cfg.design)
    if reach is not None and probe is not None:
        # Only track the split while there is a split to track. A window that reaches
        # past every gap the probe draws, or short of all of them, puts every item on one
        # side, and a curve of that against an empty column says nothing. The design's
        # true reach is still recorded; the distance bands at the end of a run are where
        # a window wider than the training range shows its edge.
        if not (probe.cfg.gap_min <= reach < probe.cfg.gap_max):
            reach = None
    marks = set(eval_steps(cfg.steps)) if cfg.dense_early else set()
    if verbose and mix.skipped():
        print(f"tasks that do not fit at {cfg.seq_len} tokens: {mix.skipped()}", flush=True)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, step_times = time.time(), []

    evals_done = [0]

    def _keep_partial(at_step: int) -> None:
        """The curve so far, uploaded, so an interrupted run is not a total loss."""
        if store is None or not tag or not getattr(store, "enabled", False):
            return
        snapshot = {"tag": tag, "partial": True, "step": at_step, "steps": cfg.steps,
                    "train_config": cfg.to_dict(), "arch": acfg.to_dict(),
                    "params": model.n_params(), "curve": curve,
                    "tokens_seen": cfg.tokens_per_step() * at_step,
                    "minutes": round((time.time() - t0) / 60, 1)}
        where = (ckpt_dir or Path(".")) / f"{tag}.partial.json"
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(json.dumps(snapshot), encoding="utf-8")
        store.put(where, f"{tag}.partial.json")
        evals_done[0] += 1
        # Sent often, because the machine can be taken away without warning: a sandbox
        # reclaimed mid-run should cost the minutes since the last upload, not the run.
        if (ckpt_path and ckpt_path.exists() and evals_done[0] % ckpt_sync_every == 0
                and ckpt_path.stat().st_size <= ckpt_sync_mb * 1_000_000):
            store.put(ckpt_path, f"{tag}.state.pt")

    def _save(at_step: int) -> None:
        """Checkpoint, plus a small progress file so a caller can see how far this got."""
        if not ckpt_path:
            return
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": at_step, "curve": curve}, ckpt_path)
        (ckpt_path.parent / "progress.json").write_text(
            json.dumps({"step": at_step, "steps": cfg.steps,
                        "fraction": round(at_step / max(1, cfg.steps), 4),
                        "updated": time.time()}), encoding="utf-8")

    try:
        for step in range(start_step, cfg.steps + 1):
            if step % cfg.eval_every == 0 or step in marks or step == cfg.steps:
                acc = a_in = a_out = None
                if probe is not None:
                    acc, a_in, a_out = quick_accuracy(
                        model, probe, cfg.seq_len, cfg.eval_items, device, amp=cfg.amp,
                        rng=np.random.default_rng(777), reach=reach)
                bpb = bits_per_byte(model, val_windows, cfg.seq_len, 16, device,
                                    amp=cfg.amp, rng=np.random.default_rng(778))
                curve.append({"step": step, "kv_accuracy": acc,
                              "kv_in_reach": a_in, "kv_beyond_reach": a_out,
                              "reach": reach,
                              "bits_per_byte": round(bpb, 4),
                              "minutes": round((time.time() - t0) / 60, 1)})
                if verbose:
                    shown = "n/a" if acc is None else f"{acc:.3f}"
                    # Either side can be empty: a configuration whose gaps all fall on
                    # one side of the window has nothing to put in the other column.
                    split = ""
                    if a_in is not None or a_out is not None:
                        _i = "n/a" if a_in is None else f"{a_in:.3f}"
                        _o = "n/a" if a_out is None else f"{a_out:.3f}"
                        split = f" (gap<={reach} {_i} / beyond {_o})"
                    print(f"[{cfg.design}/{cfg.size}] step {step:6d} kv {shown}{split} "
                          f"bpb {bpb:.3f} {(time.time() - t0) / 60:.0f}min", flush=True)
                _keep_partial(step)
            if step == cfg.steps:
                break

            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            tick = time.time()
            opt.zero_grad(set_to_none=True)
            for _ in range(cfg.accum):
                b = {k: v.to(device, non_blocking=True)
                     for k, v in mix.next_batch().items()}
                with _autocast(device, cfg.amp):
                    logits, _ = model(b["tokens"])
                loss, ce_ans, ce_rest = objective(logits, b)
                (loss / cfg.accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            if device == "cuda" and step % 50 == 0:
                torch.cuda.synchronize()
            step_times.append(time.time() - tick)

            if step > start_step and step % cfg.ckpt_every == 0:
                _save(step)
    except KeyboardInterrupt:
        # a stopped cell should cost seconds, not the whole run
        _save(step)
        print(f"interrupted at step {step}; checkpoint written, re-run to continue",
              flush=True)
        raise

    median_step = float(np.median(step_times)) if step_times else 0.0
    record = {"train_config": cfg.to_dict(), "arch": acfg.to_dict(),
              "model_config": mcfg.to_dict(), "params": model.n_params(),
              "curve": curve, "device": device,
              "tokens_seen": cfg.tokens_per_step() * cfg.steps,
              "median_step_seconds": round(median_step, 4),
              "tokens_per_second": round(cfg.tokens_per_step() / median_step, 1) if median_step else None,
              "peak_memory_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                 if device == "cuda" else None),
              "wall_minutes": round((time.time() - t0) / 60, 1),
              "skipped_tasks": mix.skipped(), "binder_reach": reach,
              "eval_marks": sorted(marks) if marks else None}
    _save(cfg.steps)
    return record, model
