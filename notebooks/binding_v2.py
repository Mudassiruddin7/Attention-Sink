# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
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
"""Variable-gap retrieval: the short run.

Press Run all and leave it. Nothing waits for input. Each job runs in its own process,
writes its result and its log to one absolute directory, and is read from disk if it is
already there, so a restart continues where it stopped. The last cell zips results, logs
and figures together.
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
        # Does the binding span have to be chosen?

        The earlier paper put each value one token after its key and bound keys with a
        width-4 convolution, so the task matched the method. Here the key-value distance
        varies from 1 to 16 tokens, keys span two tokens, and decoy keys share a prefix.

        | Objection | Answered by |
        |---|---|
        | The task matched the method | gap sweep: fixed widths against a chosen span |
        | One run per design | 6 seeds on the headline designs, paired sign tests, Holm |
        | "Placement decides" was asserted | layer-position sweep, embedding skip, layer removal |
        | The binder was never shown to bind | offset readout against the true distance |
        | The sink claim was too strong | sink at test length, with the first token randomised |

        Scores are exact match over independent inputs, markers removed from the readout,
        held-out data from the training distribution.

        Two things differ from the first attempt, which produced nothing. Inputs are 384
        tokens, because 12 pairs and 3 decoys at gaps up to 16 cannot fit in 256 and every
        gap-16 job died on that. Training inputs carry 8 queries each while scoring uses 1,
        so the gradient signal is 8 times denser without giving up one trial per input.
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

    _repo = Path("Attention-Sink")
    if not Path("sinkprobe").exists() and not _repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Mudassiruddin7/Attention-Sink.git", str(_repo)],
                       check=True)
    root = (Path(".") if Path("sinkprobe").exists() else _repo).resolve()
    sys.path.insert(0, str(root))
    from sinkprobe import vargap as vg

    # Budget. The first run showed four jobs finishing every two minutes on this GPU.
    STEPS, BATCH, WORKERS, TRAIN_LEN = 3000, 32, 6, 384
    GAPS = (1, 4, 16)
    MAIN = ["global_last_nobind", "bkf_conv4", "bkf_conv16", "dyn_first"]
    SEEDS_A, SEEDS_B = (0, 1), (2, 3, 4, 5)
    MECHANISM = ["pos0_dyn", "pos3_dyn", "pos7_dyn", "dyn_last_skip"]
    SINK_DESIGNS = ["global_last_nobind", "bkf_conv4", "dyn_first"]

    # Absolute, because the workers run with their working directory inside the clone.
    OUT = (Path.cwd() / "results_v2").resolve()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "specs").mkdir(exist_ok=True)
    print(f"results -> {OUT}")
    print(f"code    -> {root}")
    return (BATCH, GAPS, MAIN, MECHANISM, OUT, Path, SEEDS_A, SEEDS_B, SINK_DESIGNS,
            STEPS, TRAIN_LEN, WORKERS, json, np, os, root, shutil, subprocess, sys, time, vg)


@app.cell
def _(BATCH, GAPS, MAIN, MECHANISM, OUT, SEEDS_A, SEEDS_B, SINK_DESIGNS, STEPS, TRAIN_LEN,
      mo, vg):
    def _task(gap_max, n_queries):
        return dict(n_pairs=12, gap_min=1, gap_max=gap_max, key_len=2, n_decoys=3,
                    n_queries=n_queries)

    def _train(design, seed, gap_max):
        lens = [TRAIN_LEN, 1024, 4096] if (gap_max == 16 and seed <= 2) else [TRAIN_LEN, 1024]
        return dict(design=design, seed=seed, steps=STEPS, batch=BATCH, train_len=TRAIN_LEN,
                    eval_every=250, eval_inputs=256, eval_lens=lens)

    def _probes(design, seed, gap_max):
        if gap_max != 16 or seed != 0:
            return {}
        want = {"ablation": True}
        if "dyn" in design or design.startswith("pos"):
            want["offsets"] = True
        if design in SINK_DESIGNS:
            want["sink"] = [[TRAIN_LEN, False], [TRAIN_LEN, True], [1024, False], [1024, True]]
        return want

    def _job(stage, design, seed, gap_max):
        return {"tag": f"{stage}_{design}_gap{gap_max}_s{seed}", "out_dir": str(OUT),
                "task": _task(gap_max, 8), "eval_task": _task(gap_max, 1),
                "train": _train(design, seed, gap_max),
                "probes": _probes(design, seed, gap_max)}

    jobs = [_job("A", d, s, g) for g in GAPS for d in MAIN for s in SEEDS_A]
    jobs += [_job("B", d, s, 16) for d in ["bkf_conv4", "bkf_conv16", "dyn_first"] for s in SEEDS_B]
    jobs += [_job("B", "dyn_last", s, 16) for s in (0, 1, 2, 3, 4, 5)]
    jobs += [_job("C", d, 0, 16) for d in MECHANISM]

    # Nothing launches until every configuration fits at every length it is scored at.
    _bad = []
    for _j in jobs:
        _need = max(vg.VarGapTask(vg.VarGapConfig(**_j["task"])).required_length(),
                    vg.VarGapTask(vg.VarGapConfig(**_j["eval_task"])).required_length())
        _short = min([_j["train"]["train_len"]] + list(_j["train"]["eval_lens"]))
        if _short < _need:
            _bad.append((_j["tag"], _short, _need))
    if _bad:
        raise ValueError(f"{len(_bad)} jobs cannot fit, e.g. {_bad[:3]}")
    mo.md(f"**{len(jobs)} jobs**, {STEPS} steps each at {TRAIN_LEN} tokens, all sizes checked. "
          f"Finished jobs are read from disk, so only what is missing runs.")
    return (jobs,)


@app.cell
def _(OUT, WORKERS, jobs, json, os, root, subprocess, sys, time):
    def _finished():
        return sum(1 for j in jobs if (OUT / f"{j['tag']}.json").exists())

    _todo = [j for j in jobs if not (OUT / f"{j['tag']}.json").exists()]
    _pre = len(jobs) - len(_todo)
    print(f"{_pre} of {len(jobs)} already on disk; running {len(_todo)}", flush=True)

    _pending, _running, _failed, _closed, _t0 = list(_todo), [], [], 0, time.time()
    while _pending or _running:
        while _pending and len(_running) < WORKERS:
            _spec = _pending.pop(0)
            _sp = OUT / "specs" / f"{_spec['tag']}.json"
            _sp.write_text(json.dumps(_spec), encoding="utf-8")
            _lg = open(OUT / "logs" / f"{_spec['tag']}.log", "w", encoding="utf-8")
            _proc = subprocess.Popen(
                [sys.executable, "-m", "sinkprobe.vargap", "--spec", str(_sp)],
                cwd=str(root), stdout=_lg, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            _running.append((_spec, _proc, _lg))
        time.sleep(5)
        for _spec, _proc, _lg in list(_running):
            if _proc.poll() is None:
                continue
            _running.remove((_spec, _proc, _lg))
            _lg.close()
            _closed += 1
            _wrote = (OUT / f"{_spec['tag']}.json").exists()
            if not _wrote:
                _tail = (OUT / "logs" / f"{_spec['tag']}.log").read_text(
                    encoding="utf-8", errors="replace")[-700:]
                _failed.append({"tag": _spec["tag"], "returncode": _proc.returncode,
                                "tail": _tail})
                print(f"FAILED {_spec['tag']} (exit {_proc.returncode})\n{_tail}", flush=True)
            _done = _finished()
            _rate = (time.time() - _t0) / max(1, _done - _pre)
            _state = "ok" if _wrote else "no result"
            print(f"[{_done}/{len(jobs)}] {_spec['tag']} {_state} "
                  f"| elapsed {(time.time() - _t0) / 60:.0f} min "
                  f"| about {(len(jobs) - _done) * _rate / 60:.0f} min left", flush=True)
        # if the first processes all end without writing anything, stop instead of grinding on
        if _closed >= min(3, len(_todo)) and _finished() == _pre:
            for _s, _p, _l in _running:
                _p.kill()
                _l.close()
            raise RuntimeError(f"first {_closed} jobs wrote no results; logs in {OUT / 'logs'}\n"
                               + (_failed[0]["tail"] if _failed else ""))

    if _failed:
        (OUT / "_failures.txt").write_text(
            "\n\n".join(f"{f['tag']} (exit {f['returncode']})\n{f['tail']}" for f in _failed),
            encoding="utf-8")
    run_summary = {"jobs": len(jobs), "finished": _finished(), "failed": len(_failed),
                   "minutes": round((time.time() - _t0) / 60, 1)}
    run_summary
    return (run_summary,)


@app.cell
def _(OUT, json, run_summary):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))]
    records = [r for r in records if "lengths" in r]
    print(f"{len(records)} runs, {run_summary['minutes']} min, {run_summary['failed']} failures")
    return (records,)


@app.cell(hide_code=True)
def _(GAPS, MAIN, mo, np, records, vg):
    def _em(design, gap):
        return [r["lengths"][vg._len_key(r)]["em"] for r in records
                if vg._design_of(r) == design and r.get("gap_max") == gap
                and vg._len_key(r) in r["lengths"]]

    def _steps(design, gap):
        return [r["learned_step"] for r in records
                if vg._design_of(r) == design and r.get("gap_max") == gap]

    _rows = []
    for _d in MAIN:
        _row = {"design": _d}
        for _g in GAPS:
            _v, _all = _em(_d, _g), _steps(_d, _g)
            _s = [x for x in _all if x is not None]
            _row[f"gap 1-{_g}"] = f"{np.mean(_v):.2f} (n={len(_v)})" if _v else "-"
            _row[f"learned {_g}"] = (f"{int(np.median(_s))} ({len(_s)}/{len(_all)})" if _s
                                     else f"0/{len(_all)}")
        _rows.append(_row)
    _cols = ["design"] + [c for _g in GAPS for c in (f"gap 1-{_g}", f"learned {_g}")]
    mo.md("## Does a fixed span survive a growing gap?\n\n"
          "Exact match at the training length, and the step where held-out exact match first "
          "passed 0.9, with how many seeds got there.\n\n"
          + "\n".join(["| " + " | ".join(_cols) + " |", "|" + "|".join(["---"] * len(_cols)) + "|"]
                      + ["| " + " | ".join(str(r.get(c, "")) for c in _cols) + " |"
                         for r in _rows]))
    return


@app.cell(hide_code=True)
def _(mo, records, vg):
    _designs = ["bkf_conv4", "bkf_conv16", "dyn_first", "dyn_last", "global_last_nobind"]

    def _at(design, key):
        return [r["lengths"][key]["em"] for r in records
                if vg._design_of(r) == design and r.get("gap_max") == 16 and key in r["lengths"]]

    _short = {d: _at(d, "384") for d in _designs}
    _long = {d: _at(d, "4096") for d in _designs}
    _rows = []
    for _d in _designs:
        _m, _lo, _hi = vg.t_interval(_short[_d])
        _lm, _llo, _lhi = vg.t_interval(_long[_d])
        _rows.append({"design": _d, "seeds": len(_short[_d]),
                      "EM at 384": f"{_m:.3f} [{_lo:.3f}, {_hi:.3f}]" if _short[_d] else "-",
                      "EM at 4096": f"{_lm:.3f} [{_llo:.3f}, {_lhi:.3f}]" if _long[_d] else "-"})
    _tests = {f"dyn_first vs {_d}": vg.paired_sign_test(_short[_d], _short["dyn_first"])
              for _d in _designs
              if _d != "dyn_first" and _short[_d] and len(_short[_d]) == len(_short["dyn_first"])}
    _adj = vg.holm({_k: _v["p"] for _k, _v in _tests.items()}) if _tests else {}
    _c1 = ["design", "seeds", "EM at 384", "EM at 4096"]
    _t1 = "\n".join(["| " + " | ".join(_c1) + " |", "|" + "|".join(["---"] * len(_c1)) + "|"]
                    + ["| " + " | ".join(str(r[c]) for c in _c1) + " |" for r in _rows])
    _t2 = "\n".join(["| comparison | wins/losses | p | p (Holm) |", "|---|---|---|---|"]
                    + [f"| {_k} | {_v['wins']}/{_v['losses']} | {_v['p']:.4f} | {_adj[_k]:.4f} |"
                       for _k, _v in _tests.items()])
    mo.md("## Seeds and statistics, at gaps 1 to 16\n\n" + _t1
          + "\n\n### Paired over seeds (sign test, Holm corrected)\n\n"
          + (_t2 if _tests else "*not enough matched seeds yet*"))
    return


@app.cell(hide_code=True)
def _(mo, records, vg):
    _rows = []
    for _r in sorted(records, key=lambda r: (vg._design_of(r), r.get("seed", 0))):
        if not (_r.get("probes") or {}):
            continue
        _off = vg._offsets_of(_r) or {}
        _ab = _r["probes"].get("ablate_first_global") or {}
        _learned = _r["learned_step"] if _r["learned_step"] is not None else "never"
        _rows.append({"design": vg._design_of(_r), "seed": _r.get("seed"),
                      "learned step": _learned,
                      "EM": f"{_r['lengths'][vg._len_key(_r)]['em']:.3f}",
                      "picked = true offset": f"{_off['agreement']:.2f}" if _off else "-",
                      "offset correlation": (f"{_off.get('correlation', float('nan')):.2f}"
                                             if _off else "-"),
                      "EM without first global": (f"{_ab.get('em', float('nan')):.3f}"
                                                  if _ab else "-")})
    _c = ["design", "seed", "learned step", "EM", "picked = true offset",
          "offset correlation", "EM without first global"]
    mo.md("## Mechanism\n\nWhere the global layer sits, what the binder picks (a value sits "
          "`gap + 1` tokens after its key), and what happens when the first global layer is "
          "removed at test time.\n\n"
          + ("\n".join(["| " + " | ".join(_c) + " |", "|" + "|".join(["---"] * len(_c)) + "|"]
                       + ["| " + " | ".join(str(r[x]) for x in _c) + " |" for r in _rows])
             if _rows else "*no probes in these runs*"))
    return


@app.cell(hide_code=True)
def _(mo, records, vg):
    _rows = []
    for _r in records:
        for _s in ((_r.get("probes") or {}).get("sink") or []):
            _rows.append({"design": vg._design_of(_r), "length": _s["length"],
                          "first token": "random filler" if _s["random_bos"] else "fixed marker",
                          "sink mass": f"{_s['sink_mass']:.4f}",
                          "x even attention": f"{_s['sink_ratio']:.1f}"})
    _c = ["design", "length", "first token", "sink mass", "x even attention"]
    mo.md("## The sink, measured at the length it is reported for\n\n"
          "A number that only holds with a fixed first token is attention to that marker, not "
          "a sink.\n\n"
          + ("\n".join(["| " + " | ".join(_c) + " |", "|" + "|".join(["---"] * len(_c)) + "|"]
                       + ["| " + " | ".join(str(r[x]) for x in _c) + " |" for r in _rows])
             if _rows else "*no sink probes in these runs*"))
    return


@app.cell(hide_code=True)
def _(mo, records, vg):
    _lines = []
    for _r in sorted(records, key=lambda r: (r.get("gap_max", 0), vg._design_of(r))):
        if _r.get("seed") != 0:
            continue
        _curve = _r.get("curve") or []
        _em = " ".join(f"{p['em']:.2f}" for p in _curve)
        _ce = " ".join("-" if p.get("ce_answer") is None else f"{p['ce_answer']:.2f}"
                       for p in _curve)
        _lines.append(f"**{vg._design_of(_r)}, gaps 1-{_r.get('gap_max')}**  \n"
                      f"held-out EM `{_em}`  \nanswer loss `{_ce}`")
    mo.md("## Curves, seed 0\n\nOne number per evaluation, so a flat run can be read without the "
          "figures: an answer loss that falls while exact match stays near zero means the model "
          "is learning the haystack but not the lookup.\n\n"
          + ("\n\n".join(_lines) if _lines else "*nothing to show*"))
    return


@app.cell
def _(OUT, records, vg):
    figures = [p for p in (vg.figure_gap(OUT), vg.figure_offsets(OUT)) if p is not None]
    len(records), [str(p) for p in figures]
    return (figures,)


@app.cell(hide_code=True)
def _(figures, mo):
    mo.vstack([mo.image(str(p)) for p in figures] or [mo.md("*no figures yet*")])
    return


@app.cell
def _(OUT, Path, figures, mo, run_summary, shutil):
    _zip = Path(shutil.make_archive("vargap_results", "zip", OUT)).resolve()
    mo.vstack([
        mo.md(f"**{_zip.name}** - {_zip.stat().st_size / 1e6:.1f} MB, "
              f"{run_summary['finished']} of {run_summary['jobs']} runs, {len(figures)} figures, "
              f"{run_summary['failed']} failures. Every job's log is inside.\n\n`{_zip}`"),
        mo.download(data=_zip.read_bytes(), filename="vargap_results.zip",
                    label="Download the results"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## After this run

        Send the zip back. It carries every job's log now, so a failure can be read instead of
        guessed at. What the numbers decide:

        * exact match falling with the gap for the fixed widths but holding for the chosen span
          is the paper, and the long run (more seeds, byte-level text at 40M parameters) is
          worth the GPU hours;
        * every design surviving the gap means the binding story does not hold, and the honest
          move is to lead with the layer-order result and shrink the rest;
        * nothing learning at all is a budget problem rather than a result, and the curves above
          say which it is.

        Still outstanding for a submission: the scale-up on real text, an anonymous code link in
        the PDF, an AI use statement that matches what was used, and a verified account for
        every co-author.
        """
    )
    return


if __name__ == "__main__":
    app.run()
