# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "pyarrow",
#     "huggingface-hub",
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
"""How far does key binding reach, and does anything else change?

Press Run all and leave it. Nothing waits for input after the token. A learnability gate
runs first, the sweep is priced before it is paid for, every run checkpoints and uploads
its curve as it goes, and the last cell zips everything.
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
        # The reach of key binding

        A hybrid stack is mostly linear-attention layers with a few global softmax layers.
        An earlier sweep found that putting the global layer first and adding a short causal
        convolution to its keys and values raised variable-distance retrieval from 0.058 to
        0.193 over six seeds each. This notebook asks what that number was made of.

        The arithmetic that motivates it. A pair is laid out as

            [key] [filler x gap] [value]

        so the value sits `gap + 1` positions after its key. A depthwise causal convolution
        of width `w`, read at the value, mixes in the `w - 1` positions before it. The key is
        therefore inside the window exactly when

            gap + 1 <= w - 1,   that is   gap <= w - 2.

        At `w = 4` that is `gap <= 2`. Re-reading the 25 finished runs at that boundary: over
        the runs that did not solve the task, the binding design scores **0.760** inside the
        window and **0.080** beyond it, against **0.056** and **0.052** for the same model with
        the global layer last. Gaps are uniform on 1..16, so the window holds an eighth of the
        items and carries about four fifths of the aggregate gain. None of the nine aggregate
        comparisons in that draft survives a Holm correction.

        So the claim to test is not "binding helps". It is **the edge moves with the width and
        with nothing else**, which is a prediction a sweep can refute.

        | Asked by a reviewer | Answered by |
        |---|---|
        | Placement and binding changed together | the full 2x2: global first or last, binding on or off |
        | Does the convolution learn a lookup or a short-range shortcut | a width ladder from 2 to 32, with the edge predicted before the run |
        | Is the mechanism claim supported | binding switched off, binding patched from another input, one head at a time, and where each head looks |
        | Is the benchmark an artefact of disjoint alphabets | the same lookup with keys and values from one pool, and with both drawn from real text |
        | Only 4x length generalisation | lengths to 16k, and distance bands past the training range |
        | Too few seeds, no correction | matched seeds, items shared across designs, paired tests, Holm over the family |

        Width 2 reaches no key at all. If it helps anyway, the account above is wrong and the
        convolution is doing something else, which is worth knowing either way.
        """
    )
    return


@app.cell
def _(mo):
    """Asked for here rather than in the secrets pane: paste once, nothing else to press."""
    token_input = mo.ui.text(
        label="Hugging Face token (write access to the dataset)",
        kind="password", full_width=True,
        placeholder="paste here, or leave empty if HF_TOKEN is already set")
    token_input
    return (token_input,)


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
    BRANCH = "binder-range"
    _repo = Path("Attention-Sink")
    if not Path("sinkprobe").exists():
        # a folder kept across sessions can come back without its .git, and then there is
        # nothing to fetch from: clone it again rather than failing the run
        if _repo.exists() and not (_repo / ".git").exists():
            print("Attention-Sink is here without its .git; cloning again", flush=True)
            shutil.rmtree(_repo)
        if not _repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                            REPO_URL + ".git", str(_repo)], check=True)
        else:
            try:
                subprocess.run(["git", "-C", str(_repo), "fetch", "--depth", "1",
                                "origin", BRANCH], check=True)
                subprocess.run(["git", "-C", str(_repo), "checkout", "--force",
                                "FETCH_HEAD"], check=True)
            except subprocess.CalledProcessError as exc:
                print(f"could not refresh the clone (exit {exc.returncode}); using the "
                      f"code already on disk", flush=True)
    root = (Path(".") if Path("sinkprobe").exists() else _repo).resolve()
    if not (root / "sinkprobe" / "scaleup" / "mechanism.py").exists():
        raise RuntimeError(
            f"no sinkprobe/scaleup/mechanism.py under {root}. This notebook needs the "
            f"{BRANCH} branch of {REPO_URL}; push it, or clone that branch next to this "
            f"notebook.")
    def use_repo():
        """Put the clone on the import path, and say where it is.

        Every cell that imports from sinkprobe takes this as an argument, so marimo knows
        it has to run the clone first. Importing on a bare sys.path side effect looked
        fine until a fresh session ordered a cell ahead of the clone and it failed with
        ModuleNotFoundError.
        """
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    use_repo()
    from sinkprobe.scaleup import arch, corpus, runner

    OUT = (Path.cwd() / "results_range").resolve()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- the budget
    # The distance profile is an item-level measurement with a few hundred trials per
    # band, so it does not need the budget that waiting for a whole-task transition does.
    # The earlier runs crossed at 200M, 250M, 300M and 500M tokens of 600M, so 300M keeps
    # most of the transitions in view at half the price. Raise it if the projection below
    # shows headroom; every run reports its own curve, so a longer run subsumes a shorter.
    SEQ_LEN, BATCH = 1024, 16
    TOKENS_PER_RUN = 300_000_000
    STEPS = TOKENS_PER_RUN // (SEQ_LEN * BATCH)
    GATE_STEPS, GATE_THRESHOLD = 10_000, 0.5
    # Stop starting new runs this far into the session, so it ends with a complete zip
    # rather than being cut off mid-run. molab documents a 12-hour lifetime, but GPU
    # sandboxes have been reported reclaimed much sooner, so this is deliberately short
    # and the run loop leans on checkpoints instead: a run uploads its state every second
    # evaluation and picks up where it stopped when the notebook is run again.
    SESSION_HOURS = 3.0
    # How many runs share the card. A run of the 7.5M rung peaks at about 13.7 GB and the
    # card has 96, and the step time says the bottleneck is kernel launches rather than
    # arithmetic, so runs overlap well. Nothing about any run's recipe changes. Lower this
    # if the projection shows peak memory climbing towards the card.
    CONCURRENCY = 5

    TASK_RATE, TRAIN_QUERIES = 0.8, 8
    EVAL_ITEMS, BAND_ITEMS = 512, 384
    # 16384 is 16x the training length. Two global layers at that length still fit, since
    # the linear layers are chunked and only the softmax rows grow.
    EVAL_LENGTHS = [SEQ_LEN, 4096, 16384]
    TASKS = ["kv_vargap", "kv_multikey", "multi_hop", "freq_sym", "span_copy"]

    # ---------------------------------------------------------------- the plan
    # Tiers, in the order they earn their hours. Set TIERS to a prefix of "123456" to run
    # fewer: tier 1 is the paper, and each tier after it closes one more of the objections
    # the earlier draft could not answer. Every block names its own size, probe, task
    # share, corpus and budget, so a tier can change one thing without disturbing another.
    TIERS = "123456"

    PLAIN = ["global_last", "bkf_conv4"]
    FACTORIAL = ["global_last", "global_first", "bkl_conv4", "bkf_conv4"]
    WIDTHS = ["bkf_conv2", "bkf_conv3", "bkf_conv6", "bkf_conv8", "bkf_conv16",
              "bkf_conv32"]
    PLACEMENT = ["global_second", "global_third", "bk2_conv4", "bk3_conv4"]
    CONTROLS = ["bkf_conv4_frozen", "bkf_conv4_noncausal", "multi_first"]
    OTHER_LINEAR = ["global_last_nodelta", "bkf_conv4_nodelta"]

    def block(tier, name, size, designs, seeds, probe="kv_vargap", task_rate=0.8,
              corpus="wikitext103", tokens=300_000_000):
        return {"tier": tier, "name": name, "size": size, "designs": designs,
                "seeds": seeds, "probe": probe, "task_rate": task_rate,
                "corpus": corpus, "tokens": tokens}

    PLAN = [
        # --- tier 1: the paper -------------------------------------------------
        # The 2x2 the earlier sweep confounded, then the width ladder that turns the
        # account into a prediction. reach = w - 2, so 0, 1, 4, 6, 14, 30.
        block(1, "A factorial", "s", FACTORIAL, [0, 1, 2, 3]),
        block(1, "B widths", "s", WIDTHS, [0, 1]),
        # --- tier 2: is the benchmark the result? ------------------------------
        block(2, "C shared pool", "s", PLAIN, [0, 1], probe="kv_shared"),
        block(2, "D real text", "s", PLAIN, [0, 1], probe="kv_intext"),
        # --- tier 3: controls the file asked for -------------------------------
        # A frozen random convolution separates the locality prior from anything learned;
        # a centred one can read the future, which a left-to-right lookup cannot use;
        # taps at 1,2,4,8,16 are a window with holes, which separates "reaches far" from
        # "reaches everything in between".
        block(3, "F controls", "s", CONTROLS, [0, 1]),
        # Where the global layer sits, as four positions rather than two, with the
        # attention budget matched: every layout holds one S in four.
        block(3, "G placement map", "s", PLACEMENT, [0, 1]),
        # Is any of this specific to the delta rule?
        block(3, "H other linear", "s", OTHER_LINEAR, [0, 1]),
        # More than one seed of the attention-only baseline, which had exactly one.
        block(3, "I attention baseline", "s", ["all_attention", "all_linear"], [0, 1, 2]),
        # --- tier 4: does it depend on the training distribution? --------------
        # 80% of batches carried task items, which is far more retrieval supervision than
        # ordinary language-model training. Three lower shares say whether the window
        # survives a realistic mixture.
        block(4, "J task share 5%", "s", PLAIN, [0], task_rate=0.05),
        block(4, "K task share 20%", "s", PLAIN, [0], task_rate=0.2),
        block(4, "L task share 50%", "s", PLAIN, [0], task_rate=0.5),
        # Three other text distributions: Python source, generated narrative, and
        # Wikipedia with the markup left in.
        block(4, "M code", "s", PLAIN, [0], corpus="code"),
        block(4, "N stories", "s", PLAIN, [0], corpus="stories"),
        block(4, "O markup", "s", PLAIN, [0], corpus="wikitext_raw"),
        # --- tier 5: scale and budget ------------------------------------------
        block(5, "P larger", "m", ["global_last", "bkf_conv4", "bkf_conv16"], [0, 1]),
        block(5, "Q largest", "l", ["global_last", "bkf_conv4"], [0]),
        # Twice the budget, to say whether a design that had not crossed by 300M simply
        # had not got there yet. This is the only thing that separates slow from unable.
        block(5, "R long budget", "s", PLAIN, [0], tokens=1_200_000_000),
    ]
    PLAN = [b for b in PLAN if str(b["tier"]) in TIERS]

    # Tier 6 trains nothing. Released hybrid models of half a billion to three billion
    # parameters are measured on the same distance axis, which is the only way the scale
    # objection can be answered here: a 3B model cannot be trained from scratch on a free
    # sandbox, but it can be read in an hour. Full-attention models of about the same size
    # are measured beside them, because a profile with no edge only means something next
    # to one that has an edge.
    RELEASED = "6" in TIERS

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    _n = sum(len(b["designs"]) * len(b["seeds"]) for b in PLAN)
    if gpu == "cpu":
        print("!" * 72)
        print("No GPU. On molab, attach one with the notebook specs button in the header,")
        print("then run all again. Training on CPU here would take weeks, so the cells")
        print("below will refuse to start a run.")
        print("!" * 72)
    print(f"device   : {gpu}")
    print(f"code     : {root}")
    print(f"results  : {OUT}")
    print(f"budget   : {STEPS:,} steps x {SEQ_LEN * BATCH:,} tokens "
          f"= {TOKENS_PER_RUN / 1e6:.0f}M per run")
    print(f"plan     : {_n} runs over {len(PLAN)} blocks")
    for _b in PLAN:
        print(f"  t{_b['tier']} {_b['name']:22s} {_b['size']}  "
              f"{len(_b['designs'])}x{len(_b['seeds'])}  {_b['probe']:10s} "
              f"task {_b['task_rate']:.2f}  {_b['corpus']:13s} "
              f"{_b['tokens'] // 1_000_000}M")
    return (BAND_ITEMS, BATCH, CONCURRENCY, EVAL_ITEMS, EVAL_LENGTHS, GATE_STEPS,
            GATE_THRESHOLD,
            OUT, PLAN, Path, RELEASED, SEQ_LEN, SESSION_HOURS, STEPS, TASKS, TASK_RATE,
            TOKENS_PER_RUN, TRAIN_QUERIES, arch, corpus, gpu, json, np, os, root,
            runner, shutil, subprocess, sys, time, torch, use_repo)


@app.cell
def _(OUT, corpus, json):
    corpus_cfg = {"name": "wikitext103", "cache": str(OUT / "corpus"), "shards": 1}
    corpus_meta = corpus.ensure_corpus(corpus.CorpusConfig(**corpus_cfg), verbose=True)
    print(json.dumps({k: v for k, v in corpus_meta.items() if k != "sha1_head"}, indent=2))
    return (corpus_cfg, corpus_meta)


@app.cell
def _(OUT, mo, os, token_input, use_repo):
    """Where finished runs go, and what earlier sessions already finished."""
    use_repo()
    from sinkprobe.scaleup.persist import make_store

    _nl = chr(10)
    _typed = (token_input.value or "").strip()
    if _typed:
        os.environ["HF_TOKEN"] = _typed          # the worker processes inherit this
    STORE = {"kind": "all", "hf_repo": "mudassiruddin7/scaleup",
             "gh_repo": "Mudassiruddin7/Attention-Sink", "prefix": "binder_range"}
    store = make_store(**STORE)
    mo.stop(not store.enabled, mo.md(_nl.join([
        "## Waiting for the token", "",
        "molab deletes whatever a run writes when the session ends, so nothing below runs "
        "until there is somewhere to keep the results. Paste a token in the field above "
        "and the sweep starts on its own.", "",
        "The token is never written to this notebook or to the repository."])))
    restored = store.sync_down(OUT)
    tidied = store.tidy()
    _who = store.whoami() if hasattr(store, "whoami") else None
    mo.md(_nl.join([
        "## Keeping the results", "",
        f"Connected as **{_who or 'an account the token belongs to'}**. "
        f"{store.describe()}.", "",
        f"Runs recovered from earlier sessions: **{len(restored)}**, "
        f"leftover files cleared: **{len(tidied)}**."]))
    return (STORE, restored, store)


@app.cell
def _(GATE_STEPS, GATE_THRESHOLD, OUT, corpus_meta, json, runner, store, torch):
    """Can a small model learn the easiest rung at all? If not, nothing else should run."""
    _gate_path = OUT / "_gate.json"
    if _gate_path.exists():
        gate_result = json.loads(_gate_path.read_text(encoding="utf-8"))
    else:
        gate_result = runner.gate("cuda" if torch.cuda.is_available() else "cpu",
                                  corpus_meta, steps=GATE_STEPS,
                                  threshold=GATE_THRESHOLD, seq_len=192, verbose=True)
        _gate_path.write_text(json.dumps(gate_result), encoding="utf-8")
        store.put(_gate_path)
    print(f"\ngate passed: {gate_result['passed']} | best accuracy "
          f"{gate_result['best_accuracy']:.3f} vs threshold {gate_result['threshold']}")
    print(gate_result["note"])
    return (gate_result,)


@app.cell
def _(BAND_ITEMS, BATCH, EVAL_ITEMS, EVAL_LENGTHS, OUT, PLAN, SEQ_LEN, STEPS, STORE,
      TASKS, TASK_RATE, TRAIN_QUERIES, corpus_cfg, gate_result, gpu, mo, restored):
    """One job specification per run. Nothing is trained here."""
    def _spec(b, design, seed):
        size, probe = b["size"], b["probe"]
        steps = b["tokens"] // (SEQ_LEN * BATCH)
        # The tag carries every axis that is not the default, so two blocks that differ
        # only in the task share or the corpus cannot overwrite each other's record.
        tag = f"{size}_{design}_s{seed}"
        if probe != "kv_vargap":
            tag += f"_{probe}"
        if b["task_rate"] != 0.8:
            tag += f"_tr{int(b['task_rate'] * 100)}"
        if b["corpus"] != "wikitext103":
            tag += f"_{b['corpus']}"
        if b["tokens"] != 300_000_000:
            tag += f"_{b['tokens'] // 1_000_000}M"
        kinds = TASKS if probe == "kv_vargap" else [probe, "kv_vargap"]
        c_cfg = {**corpus_cfg, "name": b["corpus"]}
        return {"tag": tag, "out_dir": str(OUT), "corpus": c_cfg,
                "train": {"size": size, "design": design, "seed": seed,
                          "seq_len": SEQ_LEN, "batch": BATCH, "steps": steps,
                          "task_rate": b["task_rate"], "task_kinds": kinds,
                          "gap_max": 16,
                          "train_queries": TRAIN_QUERIES, "probe_kind": probe,
                          "dense_early": True,
                          "eval_every": max(500, steps // 12), "eval_items": 256,
                          "ckpt_every": max(500, min(2000, steps // 20))},
                "eval": {"kinds": kinds, "lengths": EVAL_LENGTHS, "n_items": EVAL_ITEMS,
                         "ablate_items": 256, "band_kinds": [probe],
                         "band_length": 4096, "band_items": BAND_ITEMS,
                         "mechanism_kind": probe, "mechanism_items": 384,
                         "rows_items": 48},
                "overhead": {"lengths": [SEQ_LEN, 4096], "batch": 4},
                "ablate": True, "bands": True, "mechanism": True,
                "ckpt_sync_every": 2,
                "persist": True, "store": STORE, "ckpt_sync_mb": 600}

    _ = restored                 # ordering: decide what is missing after the restore
    _ready = gate_result["passed"] and gpu != "cpu"
    jobs = ([_spec(b, design, seed) for b in PLAN
             for design in b["designs"] for seed in b["seeds"]] if _ready else [])
    mo.md(f"**{len(jobs)} runs queued.** Finished runs are skipped on a re-run, so a "
          "reclaimed sandbox costs the minutes since the last checkpoint and nothing else."
          if jobs else
          ("**No runs queued: there is no GPU.** Attach one with the notebook specs "
           "button in the header and run all again."
           if gpu == "cpu" else
           "**No runs queued.** The gate did not pass, so the sweep is skipped: the zip "
           "below carries the gate curve, which says whether the task or the budget is "
           "at fault."))
    return (jobs,)


@app.cell
def _(BATCH, OUT, SEQ_LEN, STEPS, corpus_meta, gate_result, jobs, json, mo, runner,
      store, torch, use_repo):
    """Price the plan before paying for it: measured seconds per step, per design."""
    use_repo()
    from sinkprobe.scaleup.corpus import ByteWindows
    from sinkprobe.scaleup.train import TrainConfig

    _nl = chr(10)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _windows = ByteWindows(corpus_meta["train"], seed=0)
    _cal_path = OUT / "_calibration.json"
    calibration = (json.loads(_cal_path.read_text(encoding="utf-8"))
                   if _cal_path.exists() else {})
    _unique = sorted({(j["train"]["size"], j["train"]["design"]) for j in jobs})
    _steps_of = {j["tag"]: j["train"]["steps"] for j in jobs}
    if gate_result["passed"]:
        for _s, _d in _unique:
            if f"{_s}/{_d}" in calibration:
                continue                      # measured in an earlier session
            _cfg = TrainConfig(size=_s, design=_d, seq_len=SEQ_LEN, batch=BATCH,
                               steps=STEPS, amp=(_device == "cuda"))
            calibration[f"{_s}/{_d}"] = runner.calibrate(_cfg, _windows, _device, steps=12)
        _cal_path.write_text(json.dumps(calibration), encoding="utf-8")
        store.put(_cal_path)

    def minutes_for(size, design, steps=None):
        key = f"{size}/{design}"
        if key not in calibration:
            return None
        return calibration[key]["seconds_per_step"] * (steps or STEPS) / 60

    _rows = [f"| {_k} | {_v['seconds_per_step']:.3f} | {_v['tokens_per_second']:,.0f} "
             f"| {_v['peak_memory_gb']} | {minutes_for(*_k.split('/')):.0f} |"
             for _k, _v in sorted(calibration.items())]
    _hours = sum(minutes_for(j["train"]["size"], j["train"]["design"],
                             j["train"]["steps"]) or 0
                 for j in jobs) / 60
    mo.md(_nl.join(
        ["## Projected cost", "",
         "| size/design | s per step | tokens/s | peak GB | minutes per run |",
         "|---|---|---|---|---|"] + _rows +
        ["", f"**{len(jobs)} runs, about {_hours:.1f} hours of training in all.** "
         "Runs that cannot finish inside SESSION_HOURS are held back, so a session ends "
         "with a complete zip and re-running continues where it stopped."])
        if calibration else
        _nl.join(["## Projected cost", "", "*gate did not pass, nothing priced*"]))
    return (calibration, minutes_for)


@app.cell
def _(OUT, Path, shutil):
    def bundle():
        """Results and logs as one zip, leaving out checkpoints and the corpus."""
        staging = Path("range_bundle")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        for item in OUT.iterdir():
            if item.name in ("ckpt", "corpus"):
                continue
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, staging / item.name)
        return Path(shutil.make_archive("binder_range_results", "zip", staging)).resolve()
    return (bundle,)


@app.cell
def _(CONCURRENCY, OUT, SESSION_HOURS, bundle, jobs, json, minutes_for, os, root,
      store, subprocess, sys, time):
    """Several runs at once, each in its own process, time-boxed.

    The measured step time says this workload is launch-bound and not FLOP-bound: the
    delta-rule layer runs a chunked Python loop, so a run uses 13.7 GB of the card and
    leaves the rest of it waiting. One run at a time therefore wastes most of the GPU.
    Running several keeps the recipe of every run exactly as it was - nothing about the
    training changes, only how many are in flight - and turns an idle card into throughput.

    A separate process per run is also what keeps a leak or a bad allocation in one run
    from taking the session down with it. The pool stops starting runs once the session is
    nearly over, so it always reaches the cells below and leaves a complete zip behind.
    """
    (OUT / "specs").mkdir(parents=True, exist_ok=True)
    _started = time.time()
    run_summary = []
    _pending = []
    for _job in jobs:
        if (OUT / f"{_job['tag']}.json").exists():
            run_summary.append({"tag": _job["tag"], "state": "already done"})
        else:
            _pending.append(_job)

    def _minutes(job):
        return minutes_for(job["train"]["size"], job["train"]["design"],
                           job["train"]["steps"]) or 0

    # Biggest first, so the long runs are in flight while the short ones fill in behind
    # them rather than being the last thing left at the end of a session.
    _pending.sort(key=_minutes, reverse=True)
    _running = {}                 # proc -> (job, log handle, log path, start time)

    def _finish(proc):
        _job, _fh, _log, _t = _running.pop(proc)
        _fh.close()
        _tail = _log.read_text(encoding="utf-8", errors="ignore").splitlines()[-2:]
        _state = "done" if proc.returncode == 0 else "failed"
        print(f"  [{_state}] {_job['tag']} in {(time.time() - _t) / 60:.0f} min"
              + ("" if not _tail else "  " + _tail[-1][:90]), flush=True)
        run_summary.append({"tag": _job["tag"], "state": _state,
                            "returncode": proc.returncode,
                            "minutes": round((time.time() - _t) / 60, 1)})
        store.put(_log)

    def _launch(job):
        _spec_path = OUT / "specs" / f"{job['tag']}.json"
        _spec_path.write_text(json.dumps(job), encoding="utf-8")
        _log = OUT / "logs" / f"{job['tag']}.log"
        _fh = open(_log, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "sinkprobe.scaleup.runner", "--spec", str(_spec_path)],
            cwd=str(root), stdout=_fh, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        _running[proc] = (job, _fh, _log, time.time())
        print(f"  [start] {job['tag']}  (about {_minutes(job):.0f} min, "
              f"{len(_running)} in flight)", flush=True)

    print(f"{len(_pending)} runs to do, {CONCURRENCY} at a time", flush=True)
    while _pending or _running:
        for proc in [p for p in _running if p.poll() is not None]:
            _finish(proc)
            bundle()              # refreshed as runs land, so one always exists
        _left = SESSION_HOURS * 60 - (time.time() - _started) / 60
        while _pending and len(_running) < CONCURRENCY:
            if _minutes(_pending[0]) > _left:
                break             # and everything behind it is shorter, so try those
            _launch(_pending.pop(0))
        if _pending and len(_running) < CONCURRENCY:
            _short = [j for j in _pending if _minutes(j) <= _left]
            if _short:
                _pending.remove(_short[0])
                _launch(_short[0])
                continue
            for _job in _pending:
                run_summary.append({"tag": _job["tag"], "state": "held back",
                                    "needs_minutes": round(_minutes(_job)),
                                    "minutes_left": round(_left)})
            _pending = []
        if _running:
            time.sleep(20)
    (OUT / "_run_summary.json").write_text(json.dumps(run_summary, indent=1),
                                           encoding="utf-8")
    store.put(OUT / "_run_summary.json")
    bundle()
    print(f"\n{sum(1 for r in run_summary if r['state'] in ('done', 'already done'))}"
          f" of {len(jobs)} runs present", flush=True)
    return (run_summary,)


@app.cell
def _(OUT, RELEASED, gpu, mo, run_summary, store, use_repo):
    """Released hybrid models on the same distance axis. Inference only, no training."""
    _ = run_summary                      # ordering: after the sweep, before the tables
    released = {}
    if RELEASED and gpu != "cpu":
        use_repo()
        from sinkprobe.scaleup.released import CONTROLS, HYBRIDS, probe_all
        print("measuring released models; each is downloaded, read and freed in turn",
              flush=True)
        released = probe_all(tuple(HYBRIDS) + tuple(CONTROLS), out_dir=OUT,
                             device="cuda", store=store, verbose=True)
    mo.md(f"**{len(released)} released models measured.**" if released else
          ("*Tier 6 is off; add `6` to TIERS to measure released models.*"
           if not RELEASED else "*No GPU, so released models were not measured.*"))
    return (released,)


@app.cell(hide_code=True)
def _(mo, released):
    """Their distance profiles, against the width of their own short convolution."""
    _nl = chr(10)
    _cols = ["2", "4", "8", "16", "32", "64", "128", "256", "512"]
    _rows = []
    for _name, _r in sorted(released.items()):
        if _r.get("skipped"):
            _rows.append(f"| {_name} | - | - | " + " | ".join(["-"] * len(_cols))
                         + f" | {_r['skipped'][:40]} |")
            continue
        _p = _r.get("copy_profile") or {}
        _cells = [f"{_p[c]['accuracy']:.2f}" if c in _p else "-" for c in _cols]
        _rows.append(
            f"| {_name} | {_r.get('conv_width')} | {_r.get('predicted_reach')} | "
            + " | ".join(_cells) + f" | {_r.get('edge')} |")
    mo.md(_nl.join(
        ["## Released models, copy accuracy against how far back the cue sits", "",
         "No prompt format and no instruction following: a random block is repeated, and "
         "the model has to predict what followed each token the first time round. The "
         "convolution width is read out of each model's own configuration. In these "
         "stacks that convolution sits inside the recurrent block rather than on the keys "
         "of an attention layer, so the reach is what the same arithmetic would predict, "
         "not a claim that it applies unchanged.", "",
         "| model | conv w | reach | " + " | ".join(_cols) + " | edge |",
         "|---|---|---|" + "---|" * (len(_cols) + 1)] + _rows)
        if _rows else "*no released models measured*")
    return


@app.cell
def _(OUT, json, mo, run_summary):
    """Everything on disk, however it got there."""
    _ = run_summary
    records = {}
    for _p in sorted(OUT.glob("*.json")):
        if _p.name.startswith("_") or _p.name.endswith(".partial.json"):
            continue
        try:
            records[_p.stem] = json.loads(_p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"could not read {_p.name}")
    mo.md(f"**{len(records)} run records loaded.**")
    return (records,)


@app.cell(hide_code=True)
def _(mo, records):
    """The headline table: aggregate accuracy, then the same split at the binder's edge."""
    _nl = chr(10)
    _rows = []
    for _tag in sorted(records):
        _r = records[_tag]
        _probe = _r["train_config"].get("probe_kind", "kv_vargap")
        _t = _r.get("tasks", {}).get(_probe, {}).get("lengths", {})
        _at = _t.get(str(_r["train_config"]["seq_len"]), {})
        if not _at:
            continue
        _br = _at.get("by_reach") or {}
        _in = (_br.get("in_reach") or {}).get("em")
        _out = (_br.get("beyond_reach") or {}).get("em")
        _rows.append(
            f"| {_tag} | {_r['design']} | {_r.get('binder_reach')} | {_at['em']:.3f} "
            f"| {'-' if _in is None else f'{_in:.3f}'} "
            f"| {'-' if _out is None else f'{_out:.3f}'} "
            f"| {_at.get('chance', 0):.3f} | {_at.get('best_constant', 0):.3f} |")
    mo.md(_nl.join(
        ["## Accuracy, in aggregate and split at the binder's edge", "",
         "`reach` is the largest gap at which the binder still carries the key into its "
         "value. A design with no binder has no edge, so its whole sample is reported "
         "beyond reach.", "",
         "| run | design | reach | all gaps | gap <= reach | gap > reach | chance | "
         "best constant |", "|---|---|---|---|---|---|---|---|"] + _rows)
        if _rows else "*no finished runs yet*")
    return


@app.cell(hide_code=True)
def _(mo, records):
    """The prediction: the edge sits at w - 2 and moves only with w."""
    _nl = chr(10)
    _bands = ["1-1", "2-2", "3-3", "4-4", "5-6", "7-10", "11-16", "17-24", "25-40", "41-64"]
    _rows = []
    for _tag in sorted(records):
        _r = records[_tag]
        _probe = _r["train_config"].get("probe_kind", "kv_vargap")
        _gb = (_r.get("gap_bands") or {}).get(_probe) or {}
        if not _gb:
            continue
        _cells = []
        for _b in _bands:
            _v = _gb.get(_b) or {}
            _cells.append("-" if "em" not in _v else f"{_v['em']:.2f}")
        _rows.append(f"| {_tag} | {_r.get('binder_reach')} | " + " | ".join(_cells) + " |")
    mo.md(_nl.join(
        ["## Accuracy against the key-value distance", "",
         "Training draws the gap from 1..16, so the last three bands were never trained "
         "on. A model that learned a lookup should be flat across the row. A model whose "
         "gain comes from binding should drop just past its own `reach`.", "",
         "| run | reach | " + " | ".join(_bands) + " |",
         "|---|---|" + "---|" * len(_bands)] + _rows)
        if _rows else "*no distance profiles yet*")
    return


@app.cell(hide_code=True)
def _(mo, records):
    """Interventions: is the binding doing the work, or only sitting there?"""
    _nl = chr(10)
    _rows = []
    for _tag in sorted(records):
        _m = (records[_tag].get("mechanism") or {})
        _ko = _m.get("binder_knockout") or {}
        if not _ko.get("intact"):
            continue
        _pt = (_m.get("binder_patch") or {})

        def _cell(block, key):
            _v = (block or {}).get(key) or {}
            return "-" if "em" not in _v else f"{_v['em']:.3f}"
        _patched = _cell(_pt, "patched")
        _rows.append(
            f"| {_tag} | {_m.get('reach')} | {_ko['intact']['em']:.3f} "
            f"| {_ko['removed']['em']:.3f} | {_patched} "
            f"| {_cell(_ko.get('intact_by_reach'), 'in_reach')} "
            f"| {_cell(_ko.get('removed_by_reach'), 'in_reach')} "
            f"| {_cell(_ko.get('intact_by_reach'), 'beyond_reach')} "
            f"| {_cell(_ko.get('removed_by_reach'), 'beyond_reach')} |")
    mo.md(_nl.join(
        ["## The binding switched off, and patched from another input", "",
         "Off means every weight stays and the binder adds nothing. Patched means it adds "
         "the term computed on a different item, so the slot is filled and the content is "
         "wrong. If the binding is what carries a key into its value, the inside-window "
         "column collapses in both and the beyond column does not move.", "",
         "| run | reach | intact | binding off | patched | in: intact | in: off "
         "| beyond: intact | beyond: off |",
         "|---|---|---|---|---|---|---|---|---|"] + _rows)
        if _rows else "*no interventions yet*")
    return


@app.cell(hide_code=True)
def _(mo, records):
    """Head level: which head of which global layer the lookup needs."""
    _nl = chr(10)
    _blocks = []
    for _tag in sorted(records):
        _ha = ((records[_tag].get("mechanism") or {}).get("head_ablation") or {})
        _c = _ha.get("conditions")
        if not _c:
            continue
        _lines = [f"**{_tag}** (reach {_ha.get('reach')})", "",
                  "| removed | all gaps | gap <= reach | gap > reach |",
                  "|---|---|---|---|"]
        for _row in _c:
            _br = _row.get("by_reach") or {}
            _i = (_br.get("in_reach") or {}).get("em")
            _o = (_br.get("beyond_reach") or {}).get("em")
            _lines.append(f"| {_row['label']} | {_row['em']:.3f} | "
                          f"{'-' if _i is None else f'{_i:.3f}'} | "
                          f"{'-' if _o is None else f'{_o:.3f}'} |")
        _blocks.append(_nl.join(_lines))
    mo.md(_nl.join(["## One head at a time", "",
                    "`L4` removes the whole layer, `L4h2` only that head.", ""]
                   + [b + _nl for b in _blocks])
          if _blocks else "*no head ablations yet*")
    return


@app.cell(hide_code=True)
def _(mo, np, records):
    """Where the heads look when they answer, against the key-value distance."""
    _nl = chr(10)
    _rows = []
    for _tag in sorted(records):
        _ra = ((records[_tag].get("mechanism") or {}).get("answer_row_attention") or {})
        _heads = _ra.get("heads")
        if not _heads:
            continue
        _best = max(_heads.items(), key=lambda kv: kv[1]["value"])
        _name, _h = _best
        _bg = _h.get("value_by_gap") or {}
        _near = [v for k, v in _bg.items() if int(k) <= 2]
        _far = [v for k, v in _bg.items() if int(k) > 2]
        _rows.append(
            f"| {_tag} | {_name} | {_h['value']:.3f} | {_h['key']:.3f} "
            f"| {_h['sink']:.3f} | {_h['top_is_value']:.3f} "
            f"| {np.mean(_near) if _near else float('nan'):.3f} "
            f"| {np.mean(_far) if _far else float('nan'):.3f} |")
    mo.md(_nl.join(
        ["## Where the best-aimed head looks on the rows that must answer", "",
         "Mass on the value of the queried pair, on the queried key itself, and on "
         "position zero. `top is value` is how often the value is the single most "
         "attended position. The last two columns split the value mass at a gap of 2.", "",
         "| run | head | value | key | sink | top is value | value, gap <= 2 "
         "| value, gap > 2 |", "|---|---|---|---|---|---|---|---|"] + _rows)
        if _rows else "*no attention readings yet*")
    return


@app.cell(hide_code=True)
def _(mo, records):
    """When each half of the task appeared, read off the dense part of the curve."""
    _nl = chr(10)
    _rows = []
    for _tag in sorted(records):
        _r = records[_tag]
        _tps = _r["train_config"]["batch"] * _r["train_config"]["seq_len"]

        def _first(field, thr):
            for _p in _r.get("curve", []):
                if (_p.get(field) or 0) >= thr:
                    return _p["step"] * _tps / 1e6
            return None

        def _fmt(v):
            return "never" if v is None else f"{v:.0f}M"
        _rows.append(f"| {_tag} | {_r.get('binder_reach')} | "
                     f"{_fmt(_first('kv_in_reach', 0.5))} | "
                     f"{_fmt(_first('kv_beyond_reach', 0.5))} | "
                     f"{_fmt(_first('kv_accuracy', 0.9))} | "
                     f"{_r['tokens_seen'] / 1e6:.0f}M |")
    mo.md(_nl.join(
        ["## When each part appeared", "",
         "Tokens at which held-out accuracy first passed 0.5 inside the window and "
         "beyond it, and at which the whole task first passed 0.9. The first quarter of "
         "each run is evaluated densely, so these are not rounded to a twelfth of the "
         "budget.", "",
         "| run | reach | inside > 0.5 | beyond > 0.5 | all > 0.9 | budget |",
         "|---|---|---|---|---|---|"] + _rows)
        if _rows else "*no curves yet*")
    return


@app.cell
def _(OUT, json, np, records, records_reach):
    """Figures: the distance profile per design, and the edge against the width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    figures = []
    BANDS = ["1-1", "2-2", "3-3", "4-4", "5-6", "7-10", "11-16", "17-24", "25-40", "41-64"]
    CENTRES = [1, 2, 3, 4, 5.5, 8.5, 13.5, 20.5, 32.5, 52.5]

    def _plain(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=3)

    # one line per design, averaged over seeds
    by_design = {}
    for _tag, _r in records.items():
        _probe = _r["train_config"].get("probe_kind", "kv_vargap")
        _gb = (_r.get("gap_bands") or {}).get(_probe) or {}
        if not _gb:
            continue
        _key = (_r["design"], _probe)
        _y = [(_gb.get(_b) or {}).get("em") for _b in BANDS]
        by_design.setdefault(_key, []).append(_y)

    if by_design:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for (_design, _probe), _ys in sorted(by_design.items()):
            _arr = np.array([[np.nan if v is None else v for v in y] for y in _ys],
                            dtype=float)
            _m = np.nanmean(_arr, axis=0)
            _reach = records_reach(records, _design)
            _label = f"{_design} ({_probe})" if _probe != "kv_vargap" else _design
            line, = ax.plot(CENTRES, _m, marker="o", ms=3.5, lw=1.2, label=_label)
            if _reach is not None and _reach >= 1:
                ax.axvline(_reach + 0.5, lw=0.7, ls=":", color=line.get_color(), alpha=0.7)
        ax.axvspan(16.5, 60, color="0.92", zorder=0)
        ax.text(17.5, 0.95, "never trained on", fontsize=7, color="0.35", va="top")
        ax.set_xscale("log")
        ax.set_xlabel("key-value distance (tokens)")
        ax.set_ylabel("exact match")
        ax.set_ylim(-0.02, 1.02)
        _plain(ax)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            _p = FIG / f"distance_profile.{ext}"
            fig.savefig(_p, dpi=200)
            figures.append(_p)
        plt.close(fig)

    # the edge against the width: where accuracy falls to half its near value
    edges = []
    for _tag, _r in records.items():
        _probe = _r["train_config"].get("probe_kind", "kv_vargap")
        _gb = (_r.get("gap_bands") or {}).get(_probe) or {}
        if not _gb or _r.get("binder_reach") is None:
            continue
        _near = (_gb.get("1-1") or {}).get("em")
        if _near is None or _near < 0.2:
            continue                       # nothing to find an edge in
        _half = _near / 2
        _edge = None
        for _b, _c in zip(BANDS, CENTRES):
            _v = (_gb.get(_b) or {}).get("em")
            if _v is not None and _v < _half:
                _edge = _c
                break
        if _edge is not None:
            edges.append((_r["binder_reach"], _edge, _tag))
    if edges:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        xs = [e[0] for e in edges]
        ys = [e[1] for e in edges]
        lim = max(max(xs), max(ys)) * 1.3 + 1
        ax.plot([0.8, lim], [0.8, lim], lw=0.8, ls="--", color="0.6",
                label="edge = reach")
        ax.scatter(xs, ys, s=22, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("reach predicted from the width  (w - 2)")
        ax.set_ylabel("measured edge (tokens)")
        _plain(ax)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            _p = FIG / f"edge_vs_width.{ext}"
            fig.savefig(_p, dpi=200)
            figures.append(_p)
        plt.close(fig)

    (OUT / "_edges.json").write_text(json.dumps(
        [{"reach": r, "edge": e, "tag": t} for r, e, t in edges], indent=1),
        encoding="utf-8")
    print(f"{len(figures)} figure files, {len(edges)} runs with a measurable edge")
    return (figures,)


@app.cell
def _():
    def records_reach(records, design):
        """The reach of a design, taken from any run of it."""
        for _r in records.values():
            if _r.get("design") == design:
                return _r.get("binder_reach")
        return None
    return (records_reach,)


@app.cell
def _(bundle, figures, mo, run_summary, store):
    _nl = chr(10)
    _zip = bundle()
    store.put(_zip)
    _held = [r for r in run_summary if r["state"] == "held back"]
    _failed = [r for r in run_summary if r["state"] == "failed"]
    mo.md(_nl.join(
        ["## Download", "",
         f"[`{_zip.name}`]({_zip.name}) holds every run record, every log, "
         f"{len(figures)} figure files and the run summary. It is also in the dataset, so "
         "it survives the session.", "",
         (f"**{len(_held)} runs were held back** for lack of session time. Run this "
          "notebook again and they start from where the clock ran out."
          if _held else "Every queued run finished."),
         "",
         (f"**{len(_failed)} runs failed**: {', '.join(r['tag'] for r in _failed)}. "
          "Their logs are in the zip." if _failed else "")]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ### Reading the tables

        The one that decides the paper is the distance profile. Three outcomes, and they
        say different things:

        - **Flat across the row, including the bands past 16.** The model learned a lookup.
          Binding, if it helped, helped it get there.
        - **High up to `reach`, then down to the no-binder level.** The gain is the window.
          The aggregate number was measuring how often a pair happened to fall inside it,
          and would rise with the width without any retrieval being learned.
        - **An edge that does not move with the width.** The account in the first cell is
          wrong. Worth reporting as it stands.

        Width 2 is the one to check first. Its window never holds a key, so if it still
        beats the no-binder baseline then the convolution is helping for a reason that has
        nothing to do with binding keys to values, and the name should change.
        """
    )
    return


if __name__ == "__main__":
    app.run()
