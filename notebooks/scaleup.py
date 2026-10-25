# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "pyarrow",
#     "torch>=2.6",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Does layer order and key binding still matter at scale?

Press Run all and leave it. Nothing waits for input. A learnability gate runs first and
the ladder is skipped if it fails, every run checkpoints so a cut-off session loses time
and not work, and the last cell zips results, logs and figures.
"""

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Layer order and key binding, at larger sizes and on more than one task

        What a reviewer asked for, and where it is answered:

        | Asked | Answered by |
        |---|---|
        | Model sizes are too small | a ladder from 7.5M to 133M parameters, same architecture, equal token budget |
        | Only key-value retrieval | six tasks: variable-gap retrieval, hard distractors, multi-hop, aggregation, long-range copy, and plain language modelling |
        | Statistical robustness | identical items for every design, Wilson intervals, paired bootstrap, McNemar tests with Holm correction, seeds at the smallest size |
        | Computational overhead | parameters, throughput and peak memory per design against the matched baseline |
        | Stronger baselines and ablations | an all-attention transformer and an all-linear model, plus removing each global layer at test time |

        Models train from scratch on byte-level real text with task items mixed in, so one
        GPU and no API is enough. Answers come from a symbol alphabet disjoint from text
        bytes: the earlier version of this task hid the answer among filler drawn from the
        same vocabulary, which made it unsolvable in principle, and every design scored at
        chance. That is what the gate below exists to catch.

        The first full sweep, 200M tokens a run, then read near chance on retrieval for
        every design: best 0.078 against a chance of 0.016, while bits per byte scaled
        cleanly from 1.40 to 1.27 and every run was still improving when it stopped. One
        seed also reached 0.998 on long-range copy where its two siblings sat at 0.02, so
        single-seed comparisons at that budget were decided by whether a run happened to
        cross an induction transition. This round therefore buys training before it buys
        breadth: six runs at the cheapest size with three times the tokens, and the curve
        inside each run reports accuracy every 50M tokens, so one run is a budget sweep.
        """
    )
    return


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch

    REPO_URL = "https://github.com/Mudassiruddin7/Attention-Sink"
    _repo = Path("Attention-Sink")
    if not Path("sinkprobe").exists():
        # a folder kept across sessions can come back without its .git, and then there
        # is nothing to fetch from: clone it again rather than failing the run
        if _repo.exists() and not (_repo / ".git").exists():
            print("Attention-Sink is here without its .git; cloning again", flush=True)
            shutil.rmtree(_repo)
        if not _repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL + ".git",
                            str(_repo)], check=True)
        else:
            try:
                subprocess.run(["git", "-C", str(_repo), "fetch", "--depth", "1",
                                "origin", "main"], check=True)
                subprocess.run(["git", "-C", str(_repo), "checkout", "--force",
                                "FETCH_HEAD"], check=True)
            except subprocess.CalledProcessError as exc:
                print(f"could not refresh the clone (exit {exc.returncode}); using the "
                      f"code already on disk", flush=True)
    root = (Path(".") if Path("sinkprobe").exists() else _repo).resolve()
    if not (root / "sinkprobe" / "scaleup" / "runner.py").exists():
        raise RuntimeError(f"no sinkprobe/scaleup found under {root}; clone "
                           f"{REPO_URL} next to this notebook")
    sys.path.insert(0, str(root))
    from sinkprobe.scaleup import arch, corpus, overhead, runner

    OUT = (Path.cwd() / "results_scale").resolve()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "specs").mkdir(exist_ok=True)

    # Budget. Equal tokens per run keeps the size comparison honest; raise it if the
    # projection below shows headroom.
    TOKENS_PER_RUN = 600_000_000
    SEQ_LEN, BATCH = 1024, 16
    STEPS = TOKENS_PER_RUN // (SEQ_LEN * BATCH)
    GATE_STEPS, GATE_THRESHOLD = 10_000, 0.5
    EVAL_ITEMS = 512
    EVAL_LENGTHS = [SEQ_LEN, 4096]
    TASKS = ["kv_vargap", "kv_multikey", "multi_hop", "freq_sym", "span_copy"]

    # The first sweep read chance everywhere and was still climbing when it stopped, so the
    # open question is how much training the hard rung needs, not which design wins at a
    # budget where none of them work. Queries per item cannot help: they are capped at the
    # number of real pairs, so the previous run was already at the ceiling. What is left is
    # the task share, worth 1.6x, and more tokens.
    TASK_RATE, TRAIN_QUERIES = 0.8, 8

    KEY_DESIGNS = ["global_last", "bkf_conv4", "dyn_first"]
    BASELINES = ["all_attention", "all_linear", "global_first"]
    # Ordered by what each run decides. The first six answer the budget question at the
    # cheapest size, and the curve inside each run reports accuracy at every 50M tokens
    # along the way, so one run gives the whole budget sweep. Later rows only earn their
    # hours if those six separate; caching means a stopped session resumes here.
    LADDER = [("s", KEY_DESIGNS, [0, 1]),
              ("m", KEY_DESIGNS, [0]),
              ("l", ["global_last", "dyn_first"], [0]),
              ("s", BASELINES, [0])]

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"device   : {gpu}")
    print(f"results  : {OUT}")
    print(f"code     : {root}")
    print(f"budget   : {STEPS:,} steps x {SEQ_LEN * BATCH:,} tokens = {TOKENS_PER_RUN / 1e6:.0f}M "
          f"tokens per run")
    return (BASELINES, BATCH, EVAL_ITEMS, EVAL_LENGTHS, GATE_STEPS, GATE_THRESHOLD,
            KEY_DESIGNS, LADDER, OUT, Path, SEQ_LEN, STEPS, TASKS, TASK_RATE, TRAIN_QUERIES,
            arch, corpus, gpu, json, np, os, overhead, root, runner, shutil, subprocess,
            sys, time, torch)


@app.cell
def _(OUT, corpus, json):
    corpus_cfg = {"name": "wikitext103", "cache": str(OUT / "corpus"), "shards": 1}
    corpus_meta = corpus.ensure_corpus(corpus.CorpusConfig(**corpus_cfg), verbose=True)
    print(json.dumps({k: v for k, v in corpus_meta.items() if k != "sha1_head"}, indent=2))
    return (corpus_cfg, corpus_meta)


@app.cell
def _(GATE_STEPS, GATE_THRESHOLD, OUT, corpus_meta, json, runner, torch):
    _gate_path = OUT / "_gate.json"
    if _gate_path.exists():
        gate_result = json.loads(_gate_path.read_text(encoding="utf-8"))
    else:
        gate_result = runner.gate("cuda" if torch.cuda.is_available() else "cpu", corpus_meta,
                                  steps=GATE_STEPS, threshold=GATE_THRESHOLD, seq_len=192,
                                  verbose=True)
        _gate_path.write_text(json.dumps(gate_result), encoding="utf-8")
    print(f"\ngate passed: {gate_result['passed']} | best accuracy "
          f"{gate_result['best_accuracy']:.3f} vs threshold {gate_result['threshold']}")
    print(gate_result["note"])
    return (gate_result,)


@app.cell
def _(BATCH, LADDER, SEQ_LEN, STEPS, corpus_meta, gate_result, mo, runner, torch):
    """Price the plan before paying for it: measured seconds per step, per configuration."""
    from sinkprobe.scaleup.corpus import ByteWindows
    from sinkprobe.scaleup.train import TrainConfig

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _windows = ByteWindows(corpus_meta["train"], seed=0)
    _plan = [(size, design) for size, designs, seeds in LADDER for design in designs
             for _ in seeds]
    _unique = sorted({(s, d) for s, d in _plan})
    calibration = {}
    if gate_result["passed"]:
        for _s, _d in _unique:
            _cfg = TrainConfig(size=_s, design=_d, seq_len=SEQ_LEN, batch=BATCH, steps=STEPS,
                               amp=(_device == "cuda"))
            calibration[f"{_s}/{_d}"] = runner.calibrate(_cfg, _windows, _device, steps=12)
    _rows = [f"| {k} | {v['seconds_per_step']:.3f} | {v['tokens_per_second']:,.0f} | "
             f"{v['peak_memory_gb']} | {v['seconds_per_step'] * STEPS / 60:.0f} |"
             for k, v in calibration.items()]
    _total = sum(v["seconds_per_step"] for v in calibration.values())
    _hours = sum(v["seconds_per_step"] * STEPS for k, v in calibration.items()
                 for _ in [1]) / 3600
    _projected = sum(calibration[f"{s}/{d}"]["seconds_per_step"] * STEPS
                     for s, d in _plan if f"{s}/{d}" in calibration) / 3600
    mo.md(("## Projected cost\n\n"
           "| size/design | s per step | tokens/s | peak GB | minutes per run |\n|---|---|---|---|---|\n"
           + "\n".join(_rows)
           + f"\n\n**{len(_plan)} runs, about {_projected:.1f} hours of training in total.** "
             "Lower TOKENS_PER_RUN in the setup cell if that does not fit the session.")
          if calibration else "## Projected cost\n\n*gate did not pass, so nothing was priced*")
    return (calibration,)


@app.cell
def _(BATCH, EVAL_ITEMS, EVAL_LENGTHS, LADDER, OUT, SEQ_LEN, STEPS, TASK_RATE, TASKS,
      TRAIN_QUERIES, corpus_cfg, gate_result, mo):
    def _spec(size, design, seed):
        return {"tag": f"{size}_{design}_s{seed}", "out_dir": str(OUT), "corpus": corpus_cfg,
                "train": {"size": size, "design": design, "seed": seed, "seq_len": SEQ_LEN,
                          "batch": BATCH, "steps": STEPS, "task_rate": TASK_RATE,
                          "task_kinds": TASKS, "gap_max": 16,
                          "train_queries": TRAIN_QUERIES,
                          "eval_every": max(500, STEPS // 12), "eval_items": 256,
                          "ckpt_every": max(500, STEPS // 12)},
                "eval": {"kinds": TASKS, "lengths": EVAL_LENGTHS, "n_items": EVAL_ITEMS,
                         "ablate_items": 256},
                "overhead": {"lengths": EVAL_LENGTHS, "batch": 4}, "ablate": True}

    jobs = ([_spec(size, design, seed) for size, designs, seeds in LADDER
             for design in designs for seed in seeds] if gate_result["passed"] else [])
    mo.md(f"**{len(jobs)} runs queued**" if jobs else
          "**No runs queued.** The gate did not pass, so the ladder is skipped: the zip "
          "below carries the gate curve, which says whether the task or the budget is at "
          "fault.")
    return (jobs,)


@app.cell
def _(OUT, jobs, json, os, root, subprocess, sys, time):
    """One run at a time: a 133M model wants the whole GPU."""
    _done, _failed, _t0 = [], [], time.time()
    for _i, _spec in enumerate(jobs, 1):
        _target = OUT / f"{_spec['tag']}.json"
        if _target.exists():
            _done.append(_spec["tag"])
            print(f"[{_i}/{len(jobs)}] {_spec['tag']} already on disk", flush=True)
            continue
        _sp = OUT / "specs" / f"{_spec['tag']}.json"
        _sp.write_text(json.dumps(_spec), encoding="utf-8")
        _log = OUT / "logs" / f"{_spec['tag']}.log"
        print(f"[{_i}/{len(jobs)}] {_spec['tag']} started "
              f"({(time.time() - _t0) / 60:.0f} min elapsed)", flush=True)
        with open(_log, "w", encoding="utf-8") as _fh:
            _proc = subprocess.run(
                [sys.executable, "-m", "sinkprobe.scaleup.runner", "--spec", str(_sp)],
                cwd=str(root), stdout=_fh, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "4"})
        if _target.exists():
            _done.append(_spec["tag"])
        else:
            _tail = _log.read_text(encoding="utf-8", errors="replace")[-800:]
            _failed.append({"tag": _spec["tag"], "returncode": _proc.returncode, "tail": _tail})
            print(f"FAILED {_spec['tag']} (exit {_proc.returncode})\n{_tail}", flush=True)
            if len(_failed) >= 2 and not _done:
                break                      # two failures and nothing finished: stop early
    if _failed:
        (OUT / "_failures.txt").write_text(
            "\n\n".join(f"{f['tag']} (exit {f['returncode']})\n{f['tail']}" for f in _failed),
            encoding="utf-8")
    run_summary = {"queued": len(jobs), "finished": len(_done), "failed": len(_failed),
                   "minutes": round((time.time() - _t0) / 60, 1)}
    run_summary
    return (run_summary,)


@app.cell
def _(OUT, json, run_summary):
    records = {}
    for _p in sorted(OUT.glob("*.json")):
        if _p.name.startswith("_"):
            continue
        _r = json.loads(_p.read_text(encoding="utf-8"))
        if "tasks" in _r:
            records[_r["tag"]] = _r
    print(f"{len(records)} runs loaded, {run_summary['minutes']} min this session")
    return (records,)


@app.cell(hide_code=True)
def _(SEQ_LEN, mo, np, records):
    """Does the effect hold as the model grows?"""
    _sizes = ["s", "m", "l"]
    _designs = sorted({r["design"] for r in records.values()})
    _rows = []
    for _d in _designs:
        _cells = []
        for _s in _sizes:
            _v = [r["tasks"]["kv_vargap"]["lengths"][str(SEQ_LEN)]["em"]
                  for r in records.values()
                  if r["design"] == _d and r["size"] == _s
                  and str(SEQ_LEN) in r["tasks"]["kv_vargap"]["lengths"]]
            _p = [r["params"] for r in records.values()
                  if r["design"] == _d and r["size"] == _s]
            _cells.append(f"{np.mean(_v):.3f} (n={len(_v)})" if _v else "-")
        _rows.append(f"| {_d} | " + " | ".join(_cells) + " |")
    mo.md("## Variable-gap retrieval against model size\n\n"
          "Exact match at the training length, equal token budget at every size.\n\n"
          "| design | s (7.5M) | m (44M) | l (133M) |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(SEQ_LEN, TASKS, mo, records):
    """One table per size: every task, every design."""
    _out = []
    for _size in ["s", "m", "l"]:
        _runs = {t: r for t, r in records.items() if r["size"] == _size and r["seed"] == 0}
        if not _runs:
            continue
        _designs = sorted({r["design"] for r in _runs.values()})
        _head = "| task | " + " | ".join(_designs) + " |"
        _sep = "|" + "|".join(["---"] * (len(_designs) + 1)) + "|"
        _lines = [_head, _sep]
        for _k in TASKS:
            _cells = []
            for _d in _designs:
                _r = next((r for r in _runs.values() if r["design"] == _d), None)
                _s = (_r or {}).get("tasks", {}).get(_k, {}).get("lengths", {}).get(str(SEQ_LEN))
                _cells.append(f"{_s['em']:.3f} [{_s['lo']:.3f},{_s['hi']:.3f}]" if _s else "-")
            _lines.append(f"| {_k} | " + " | ".join(_cells) + " |")
        _r0 = next(iter(_runs.values()))
        _bpb = {d: next((r["tasks"]["language_model"]["lengths"][str(SEQ_LEN)]["bits_per_byte"]
                         for r in _runs.values() if r["design"] == d), None) for d in _designs}
        _lines.append("| bits per byte | " + " | ".join(
            f"{_bpb[d]:.3f}" if _bpb[d] else "-" for d in _designs) + " |")
        _out.append(f"### size {_size} ({_r0['params'] / 1e6:.1f}M parameters)\n\n"
                    + "\n".join(_lines))
    mo.md("## Every task, every design\n\nExact match with Wilson intervals over "
          f"{512} independent items; bits per byte on held-out text (lower is better).\n\n"
          + ("\n\n".join(_out) if _out else "*nothing yet*"))
    return


@app.cell(hide_code=True)
def _(SEQ_LEN, TASKS, mo, records):
    """Paired tests on identical items, corrected across the task family."""
    from sinkprobe.scaleup.evaluate import family_tests

    _by_size = {}
    for _size in ["s", "m", "l"]:
        _runs = {r["design"]: r for r in records.values()
                 if r["size"] == _size and r["seed"] == 0}
        if "global_last" not in _runs or "dyn_first" not in _runs:
            continue
        _rows = family_tests({"a": _runs["global_last"], "b": _runs["dyn_first"]},
                             "a", "b", TASKS, SEQ_LEN)
        _lines = ["| task | dyn_first minus global_last | 95% CI | wins/losses | p | p (Holm) |",
                  "|---|---|---|---|---|---|"]
        for _k, _v in _rows.items():
            _lines.append(f"| {_k} | {_v['diff']:+.3f} | [{_v['lo']:+.3f}, {_v['hi']:+.3f}] "
                          f"| {_v['wins']}/{_v['losses']} | {_v['p']:.4f} "
                          f"| {_v.get('p_holm', float('nan')):.4f} |")
        _by_size[_size] = "\n".join(_lines)
    mo.md("## Chosen span against the baseline order, paired on identical items\n\n"
          "Difference in exact match on the same inputs, bootstrap interval, and McNemar's "
          "exact test with Holm correction across the five tasks.\n\n"
          + ("\n\n".join(f"### size {k}\n\n{v}" for k, v in _by_size.items())
             if _by_size else "*needs a baseline and a candidate at the same size*"))
    return


@app.cell(hide_code=True)
def _(mo, records):
    """Removing one layer at a time, with a control for layer 0."""
    _nl = chr(10)
    _lines = ["| run | layer | kind | EM without it | full model |",
              "|---|---|---|---|---|"]
    for _t, _r in sorted(records.items()):
        _seq = str(_r["train_config"]["seq_len"])
        _full = _r["tasks"]["kv_vargap"]["lengths"].get(_seq, {})
        for _a in (_r.get("ablate_layers") or _r.get("ablate_global_layers") or []):
            _lines.append(f"| {_t} | {_a['layer']} | {_a.get('kind', '?')} "
                          f"| {_a['em']:.3f} [{_a['lo']:.3f},{_a['hi']:.3f}] "
                          f"| {_full.get('em', float('nan')):.3f} |")
    _note = ("Every global layer, plus layer 0 whatever kind it is. Only a global-first "
             "design has a global layer at index 0, so without that control a collapse "
             "there cannot be told apart from removing the first layer of any model.")
    mo.md(_nl.join(["## Ablation: what each layer is carrying", "", _note, "",
                    _nl.join(_lines) if len(_lines) > 2 else "*no ablations recorded*"]))
    return


@app.cell(hide_code=True)
def _(mo, records):
    """What the designs cost."""
    _lines = ["| run | params | binder share | tokens/s (train) | peak GB | s/step |",
              "|---|---|---|---|---|---|"]
    for _t, _r in sorted(records.items()):
        _a = _r.get("arch_summary", {})
        _share = 100 * _a.get("binder_params", 0) / max(1, _r["params"])
        _lines.append(f"| {_t} | {_r['params'] / 1e6:.1f}M | {_share:.2f}% "
                      f"| {_r.get('tokens_per_second')} | {_r.get('peak_memory_gb')} "
                      f"| {_r.get('median_step_seconds')} |")
    mo.md("## Computational overhead\n\nMeasured during the run that produced the accuracy "
          "above, so the two can be read together.\n\n" + "\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo, records):
    """Long-range copy is bimodal, so it is reported as a count."""
    _nl = chr(10)
    _tally = {}
    for _r in records.values():
        _seq = str(_r["train_config"]["seq_len"])
        _s = _r["tasks"]["span_copy"]["lengths"].get(_seq)
        if _s is None:
            continue
        _k = (_r["size"], _r["design"])
        _hit, _n = _tally.get(_k, (0, 0))
        _tally[_k] = (_hit + int(_s["em"] > 0.5), _n + 1)
    _lines = ["| size | design | seeds that formed induction | seeds run |",
              "|---|---|---|---|"]
    for (_sz, _d), (_hit, _n) in sorted(_tally.items()):
        _lines.append(f"| {_sz} | {_d} | {_hit} | {_n} |")
    _note = ("Long-range copy lands near 0.02 or near 1.00 with almost nothing between, "
             "so a mean over seeds describes neither outcome. The count is the honest "
             "summary, and it doubles as the clearest signal of whether a run formed an "
             "induction head at all.")
    mo.md(_nl.join(["## Induction, counted rather than averaged", "", _note, "",
                    _nl.join(_lines)]))
    return


@app.cell
def _(OUT, SEQ_LEN, np, records):
    """Accuracy against parameter count, one line per design."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    if records:
        fig, ax = plt.subplots(figsize=(5.6, 3.1))
        for design in sorted({r["design"] for r in records.values()}):
            pts = sorted(((r["params"],
                           r["tasks"]["kv_vargap"]["lengths"][str(SEQ_LEN)]["em"])
                          for r in records.values() if r["design"] == design
                          and str(SEQ_LEN) in r["tasks"]["kv_vargap"]["lengths"]),
                         key=lambda z: z[0])
            if pts:
                ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], marker="o",
                        label=design, linewidth=1.6, markersize=4)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (millions)")
        ax.set_ylabel("exact match")
        ax.set_title("Variable-gap retrieval against model size")
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = OUT / "fig_scaling.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        figures.append(p)
    [str(p) for p in figures]
    return (figures,)


@app.cell(hide_code=True)
def _(figures, mo):
    mo.vstack([mo.image(str(p)) for p in figures] or [mo.md("*no figures yet*")])
    return


@app.cell
def _(OUT, Path, figures, mo, run_summary, shutil):
    """Zip everything except the checkpoints, which are large and only needed to resume."""
    _staging = Path("scale_bundle")
    if _staging.exists():
        shutil.rmtree(_staging)
    _staging.mkdir()
    for _p in OUT.iterdir():
        if _p.name in ("ckpt", "corpus"):
            continue
        (shutil.copytree if _p.is_dir() else shutil.copy2)(_p, _staging / _p.name)
    _zip = Path(shutil.make_archive("scaleup_results", "zip", _staging)).resolve()
    mo.vstack([
        mo.md(f"**{_zip.name}** - {_zip.stat().st_size / 1e6:.1f} MB, "
              f"{run_summary['finished']} of {run_summary['queued']} runs, "
              f"{run_summary['failed']} failures, {len(figures)} figures.\n\n`{_zip}`"),
        mo.download(data=_zip.read_bytes(), filename="scaleup_results.zip",
                    label="Download the results"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## If the session is cut off

        Run all again. Finished runs are read from disk and a part-trained run resumes from
        its checkpoint, so the ladder picks up where it stopped.

        ## What this cannot claim

        133M parameters is two orders of magnitude above the earlier paper and three below a
        frontier model. The honest claim is a trend across the sizes that one GPU can train
        from scratch, on a task suite where the answer is identifiable and the items are
        shared across designs. Say that in the paper rather than implying the result extends
        to production scale, and the reviewer's objection is answered on its own terms.
        """
    )
    return


if __name__ == "__main__":
    app.run()
