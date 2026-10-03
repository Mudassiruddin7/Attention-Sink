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
"""Follow-up study: does the binding span have to be chosen, or can it be fixed?

Run on molab with an RTX Pro 6000 attached, or locally with uvx marimo edit.
Every stage writes JSON into results_vargap/ and skips work that is already there,
so a 12-hour session can be stopped and resumed.
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
        # Chosen binding spans for retrieval in hybrid models

        The previous paper proposed BKF: put the global layer first and bind each key to
        the three tokens before it with a fixed convolution. A reviewer made the fair
        objection that **the task handed the method its answer**, because the value always
        sat one token after its key, which is exactly what a width-4 convolution covers.

        This notebook runs the study that answers that objection and carries the novelty.
        The key-value distance now varies, keys span two tokens, and decoy keys share a
        prefix. A fixed window cannot cover a distance it was not built for, so the
        question becomes whether the span should be **chosen from the content**.

        | Objection raised | Stage here | What would settle it |
        |---|---|---|
        | The task matches the method | A | Fixed widths fall with the gap; a chosen span does not |
        | Thin statistics, one run per design | B | 8 to 20 seeds, paired sign tests, Holm correction |
        | "Placement decides" is asserted | C | Position sweep, embedding skip, layer removal, offset readout |
        | The sink claim is too strong | D | Sink measured at test length, with the marker randomised |
        | Too small, synthetic only | E | 40M parameters on byte-level WikiText with a copy probe |

        Scores are exact match counted over **independent inputs** (one query per input),
        the markers are removed from the readout, and the held-out set comes from the
        training distribution, so no number is read off an easier mix of data.
        """
    )
    return


@app.cell
def _():
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch

    _repo = Path("Attention-Sink")
    if not Path("sinkprobe").exists() and not _repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Mudassiruddin7/Attention-Sink.git", str(_repo)],
                       check=True)
    root = Path(".") if Path("sinkprobe").exists() else _repo
    sys.path.insert(0, str(root.resolve()))
    from sinkprobe import vargap as vg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "no GPU found"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    OUT = Path("results_vargap")
    OUT.mkdir(exist_ok=True)
    return OUT, Path, device, gpu, json, np, root, time, torch, vg


@app.cell(hide_code=True)
def _(OUT, device, gpu, mo, root, torch):
    mo.md(
        f"""
        Running on **{gpu}** (device `{device}`, torch {torch.__version__}).
        Code from `{root.resolve()}`, results in `{OUT.resolve()}`.
        """
    )
    return


@app.cell
def _(OUT, json, time):
    def run_cached(tag, fn):
        """Run fn() unless its JSON is already on disk, so a session can be resumed."""
        path = OUT / f"{tag}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        started = time.time()
        out = fn()
        out["_tag"] = tag
        out["_wall_seconds"] = round(time.time() - started, 1)
        path.write_text(json.dumps(out), encoding="utf-8")
        return out

    def table(rows, columns):
        """Small markdown table; keeps every number in one place for the paper."""
        head = "| " + " | ".join(columns) + " |"
        rule = "|" + "|".join(["---"] * len(columns)) + "|"
        body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
        return "\n".join([head, rule] + body)
    return run_cached, table


@app.cell
def _(mo):
    seeds = mo.ui.slider(1, 20, value=8, label="seeds per design (stage B)")
    steps = mo.ui.slider(500, 8000, step=500, value=3000, label="training steps")
    batch = mo.ui.slider(16, 256, step=16, value=64, label="batch size")
    run_a = mo.ui.run_button(label="Stage A - does the span matter?")
    run_b = mo.ui.run_button(label="Stage B - seeds and statistics")
    run_c = mo.ui.run_button(label="Stage C - mechanism")
    run_d = mo.ui.run_button(label="Stage D - the sink, measured honestly")
    run_e = mo.ui.run_button(label="Stage E - one size up on real text")
    mo.vstack([seeds, steps, batch,
               mo.hstack([run_a, run_b, run_c], justify="start"),
               mo.hstack([run_d, run_e], justify="start")])
    return batch, run_a, run_b, run_c, run_d, run_e, seeds, steps


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Stage A. Does the binding span have to be chosen?

        Three task settings: the value sits 1 token after its key, 1 to 4 tokens after, or
        1 to 16 tokens after. Five designs, three seeds each. A fixed width-4 convolution
        covers only the first setting; a width-16 convolution covers all three but spends
        its window on every token; the dynamic binder picks an offset per position.
        """
    )
    return


@app.cell
def _(batch, device, mo, run_a, run_cached, steps, vg):
    mo.stop(not run_a.value, mo.md("*Press **Stage A** to run.*"))

    a_designs = ["global_last_nobind", "global_first_nobind", "bkf_conv4", "bkf_conv16",
                 "multi_first", "dyn_first"]
    a_gaps = [1, 4, 16]
    a_rows = []
    for a_gmax in a_gaps:
        a_task = vg.VarGapConfig(n_pairs=16, gap_min=1, gap_max=a_gmax, key_len=2, n_decoys=4,
                                 n_queries=1)
        for a_design in a_designs:
            for a_seed in range(3):
                def a_job(d=a_design, s=a_seed, t=a_task):
                    cfg = vg.TrainConfig(design=d, seed=s, steps=steps.value, batch=batch.value,
                                         eval_lens=(256, 1024, 4096), eval_inputs=512)
                    res, _ = vg.train_one(cfg, t, device=device, verbose=False)
                    return res
                a_res = run_cached(f"A_{a_design}_gap{a_gmax}_s{a_seed}", a_job)
                a_rows.append({"design": a_design, "gap_max": a_gmax, "seed": a_seed,
                               "learned_step": a_res["learned_step"],
                               "em_256": a_res["lengths"]["256"]["em"],
                               "em_4096": a_res["lengths"]["4096"]["em"]})
    len(a_rows)
    return a_designs, a_gaps, a_rows


@app.cell(hide_code=True)
def _(a_designs, a_gaps, a_rows, mo, np, table, vg):
    a_summary = []
    for _d in a_designs:
        _row = {"design": _d}
        for _g in a_gaps:
            _vals = [r["em_256"] for r in a_rows if r["design"] == _d and r["gap_max"] == _g]
            _steps = [r["learned_step"] for r in a_rows if r["design"] == _d and r["gap_max"] == _g]
            _m, _lo, _hi = vg.t_interval(_vals)
            _done = [s for s in _steps if s is not None]
            _row[f"gap<={_g}"] = f"{_m:.2f} [{_lo:.2f}, {_hi:.2f}]"
            _row[f"step g{_g}"] = (f"{int(np.median(_done))} ({len(_done)}/{len(_steps)})"
                                   if _done else "never")
        a_summary.append(_row)
    mo.md("### Exact match at the training length, and the step where it passed 0.9\n\n"
          + table(a_summary, ["design"] + [c for g in a_gaps for c in (f"gap<={g}", f"step g{g}")]))
    return (a_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Stage B. Seeds and statistics

        The headline comparison at gaps 1 to 16, with the number of seeds set above.
        Reported as a mean over seeds with a Student interval, a paired sign test on
        seed-matched pairs, and Holm correction across the comparisons.
        """
    )
    return


@app.cell
def _(batch, device, mo, run_b, run_cached, seeds, steps, vg):
    mo.stop(not run_b.value, mo.md("*Press **Stage B** to run.*"))

    b_designs = ["bkf_conv4", "bkf_conv16", "dyn_first", "dyn_last"]
    b_task = vg.VarGapConfig(n_pairs=16, gap_min=1, gap_max=16, key_len=2, n_decoys=4, n_queries=1)
    b_runs = {d: [] for d in b_designs}
    for b_design in b_designs:
        for b_seed in range(seeds.value):
            def b_job(d=b_design, s=b_seed):
                cfg = vg.TrainConfig(design=d, seed=s, steps=steps.value, batch=batch.value,
                                     eval_lens=(256, 1024, 4096, 16384), eval_inputs=512)
                res, _ = vg.train_one(cfg, b_task, device=device, verbose=False)
                return res
            b_runs[b_design].append(run_cached(f"B_{b_design}_s{b_seed}", b_job))
    sum(len(v) for v in b_runs.values())
    return b_designs, b_runs, b_task


@app.cell(hide_code=True)
def _(b_designs, b_runs, mo, table, vg):
    b_em = {_d: [r["lengths"]["256"]["em"] for r in _runs] for _d, _runs in b_runs.items()}
    b_long = {_d: [r["lengths"]["16384"]["em"] for r in _runs] for _d, _runs in b_runs.items()}
    b_rows = []
    for _d in b_designs:
        _m, _lo, _hi = vg.t_interval(b_em[_d])
        _lm, _llo, _lhi = vg.t_interval(b_long[_d])
        _learned = [r["learned_step"] for r in b_runs[_d] if r["learned_step"] is not None]
        b_rows.append({"design": _d, "runs": len(b_em[_d]),
                       "EM at 256": f"{_m:.3f} [{_lo:.3f}, {_hi:.3f}]",
                       "EM at 16384": f"{_lm:.3f} [{_llo:.3f}, {_lhi:.3f}]",
                       "learned": f"{len(_learned)}/{len(b_runs[_d])}"})
    b_tests = {f"dyn_first vs {_d}": vg.paired_sign_test(b_em[_d], b_em["dyn_first"])
               for _d in b_designs if _d != "dyn_first"}
    b_adj = vg.holm({_k: _v["p"] for _k, _v in b_tests.items()})
    b_test_rows = [{"comparison": _k, "wins/losses": f"{_v['wins']}/{_v['losses']}",
                    "p": f"{_v['p']:.4f}", "p (Holm)": f"{b_adj[_k]:.4f}"}
                   for _k, _v in b_tests.items()]
    mo.md("### Across seeds\n\n"
          + table(b_rows, ["design", "runs", "EM at 256", "EM at 16384", "learned"])
          + "\n\n### Paired over seeds\n\n"
          + table(b_test_rows, ["comparison", "wins/losses", "p", "p (Holm)"]))
    return b_adj, b_em, b_long, b_rows, b_tests


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Stage C. Mechanism

        Three checks that the earlier paper asserted rather than tested.

        1. **Where the global layer sits.** One global layer, moved through all eight depths.
        2. **Why first helps.** A late global layer with a direct path to the embeddings. If
           the "it reads embeddings" story holds, the skip should recover early learning.
        3. **What the binder picks.** A value sits `gap + 1` tokens after the last key token,
           so a binder that ties them should put its mass on that offset. Plus removal of the
           first global layer at test time, to locate the lookup.
        """
    )
    return


@app.cell
def _(OUT, batch, device, json, mo, np, run_c, steps, vg):
    mo.stop(not run_c.value, mo.md("*Press **Stage C** to run.*"))

    c_task = vg.VarGapConfig(n_pairs=16, gap_min=1, gap_max=16, key_len=2, n_decoys=4, n_queries=1)
    c_designs = [f"pos{i}_dyn" for i in range(8)] + ["dyn_last", "dyn_last_skip", "dyn_first"]
    c_rows = []
    for c_design in c_designs:
        for c_seed in range(2):
            c_path = OUT / f"C_{c_design}_s{c_seed}.json"
            if c_path.exists():
                c_rows.append(json.loads(c_path.read_text(encoding="utf-8")))
                continue
            c_cfg = vg.TrainConfig(design=c_design, seed=c_seed, steps=steps.value,
                                   batch=batch.value, eval_lens=(256, 4096), eval_inputs=512)
            c_res, c_model = vg.train_one(c_cfg, c_task, device=device, verbose=False)
            c_probe = vg.VarGapTask(c_task)
            c_rng = np.random.default_rng(31337 + c_seed)
            c_entry = {"design": c_design, "seed": c_seed,
                       "learned_step": c_res["learned_step"],
                       "em_256": c_res["lengths"]["256"]["em"],
                       "em_4096": c_res["lengths"]["4096"]["em"],
                       "offsets": vg.offset_report(c_model, c_probe, 256, 8, c_rng, device),
                       "ablate_first_global": vg.ablate_global_layer(c_model, c_probe, 256, 256,
                                                                     c_rng, device, which=0)}
            c_entry["offsets"].pop("rows", None)
            c_path.write_text(json.dumps(c_entry), encoding="utf-8")
            c_rows.append(c_entry)
            del c_model
    len(c_rows)
    return c_designs, c_rows, c_task


@app.cell(hide_code=True)
def _(c_rows, mo, table):
    c_tab = []
    for _r in c_rows:
        _off = _r.get("offsets") or {}
        _ab = _r.get("ablate_first_global") or {}
        c_tab.append({"design": _r["design"], "seed": _r["seed"],
                      "learned step": _r["learned_step"] if _r["learned_step"] is not None else "never",
                      "EM 256": f"{_r['em_256']:.3f}", "EM 4096": f"{_r['em_4096']:.3f}",
                      "offset agrees": f"{_off.get('agreement', float('nan')):.2f}",
                      "offset corr": f"{_off.get('correlation', float('nan')):.2f}",
                      "EM without first global": f"{_ab.get('em', float('nan')):.3f}"})
    mo.md("### Mechanism\n\n" + table(c_tab, ["design", "seed", "learned step", "EM 256",
                                              "EM 4096", "offset agrees", "offset corr",
                                              "EM without first global"]))
    return (c_tab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Stage D. The sink, measured honestly

        Two changes to the earlier measurement. The sink is read at the length it is
        reported for, not only at the training length, and the first token is either the
        usual marker or an ordinary filler token. A number that only holds with a fixed
        marker is attention to that marker, not a sink.
        """
    )
    return


@app.cell
def _(OUT, batch, device, json, mo, np, run_d, steps, vg):
    mo.stop(not run_d.value, mo.md("*Press **Stage D** to run.*"))

    d_rows = []
    for d_design in ["dyn_first", "bkf_conv4", "global_last_nobind"]:
        d_path = OUT / f"D_{d_design}.json"
        if d_path.exists():
            d_rows.append(json.loads(d_path.read_text(encoding="utf-8")))
            continue
        d_task = vg.VarGapConfig(n_pairs=16, gap_min=1, gap_max=16, key_len=2, n_decoys=4,
                                 n_queries=1)
        d_cfg = vg.TrainConfig(design=d_design, seed=0, steps=steps.value, batch=batch.value,
                               eval_lens=(256, 1024), eval_inputs=256)
        d_res, d_model = vg.train_one(d_cfg, d_task, device=device, verbose=False)
        d_entry = {"design": d_design, "em_256": d_res["lengths"]["256"]["em"], "sinks": []}
        d_fields = {k: v for k, v in d_task.to_dict().items() if k != "vocab_size"}
        for d_len in (256, 1024):
            for d_random in (False, True):
                d_probe = vg.VarGapTask(vg.VarGapConfig(**{**d_fields, "random_bos": d_random}))
                d_entry["sinks"].append(vg.sink_stats(d_model, d_probe, d_len, 4,
                                                      np.random.default_rng(5), device))
        d_path.write_text(json.dumps(d_entry), encoding="utf-8")
        d_rows.append(d_entry)
        del d_model
    len(d_rows)
    return (d_rows,)


@app.cell(hide_code=True)
def _(d_rows, mo, table):
    d_tab = []
    for _r in d_rows:
        for _s in _r["sinks"]:
            d_tab.append({"design": _r["design"], "length": _s["length"],
                          "first token": "random filler" if _s["random_bos"] else "fixed marker",
                          "sink mass": f"{_s['sink_mass']:.4f}",
                          "times even attention": f"{_s['sink_ratio']:.1f}"})
    mo.md("### Attention on position 0\n\n"
          + table(d_tab, ["design", "length", "first token", "sink mass", "times even attention"]))
    return (d_tab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Stage E. One size up, on real text

        A 40M-parameter model on byte-level WikiText, trained with a plain next-token loss,
        then scored on the copy probe: a span from the passage is repeated after a
        separator, and the model has to continue it. The control repeats a span that is
        **not** in the context, which is what language statistics alone reach.

        This is the long run: roughly 1 to 3 hours per design on an RTX Pro 6000.
        """
    )
    return


@app.cell
def _(Path, device, mo, root, run_e, run_cached, subprocess_text, vg):
    mo.stop(not run_e.value, mo.md("*Press **Stage E** to run.*"))

    e_train = root / "data" / "wikitext2_train.txt"
    e_eval = root / "data" / "wikitext103_validation.txt"
    if not e_train.exists() or not e_eval.exists():
        subprocess_text(["python", str(root / "scripts" / "fetch_wikitext.py")], cwd=str(root))
    e_rows = []
    for e_design in ["global_last_nobind", "bkf_conv4", "dyn_first"]:
        def e_job(d=e_design):
            res, _ = vg.train_text(d, 0, str(e_train), str(e_eval), steps=4000, batch=8,
                                   seq_len=2048, d_model=512, n_layers=12, n_heads=8,
                                   lr=1e-3, device=device, eval_lens=(2048, 4096, 8192),
                                   eval_seqs=64, verbose=True)
            return res
        e_rows.append(run_cached(f"E_{e_design}", e_job))
    len(e_rows)
    return (e_rows,)


@app.cell
def _():
    import subprocess as _sp

    def subprocess_text(cmd, cwd=None):
        """Run a helper script and show its tail, so a failure is visible in the notebook."""
        out = _sp.run(cmd, cwd=cwd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])
        return out.stdout[-2000:]
    return (subprocess_text,)


@app.cell(hide_code=True)
def _(e_rows, mo, table):
    e_tab = []
    for _r in e_rows:
        for _len, _sc in _r["evals"].items():
            e_tab.append({"design": _r["design"], "params": f"{_r['params'] / 1e6:.1f}M",
                          "length": _len,
                          "copy accuracy": f"{_sc['probe']['copy_acc']:.3f} "
                                           f"[{_sc['probe']['lo']:.3f}, {_sc['probe']['hi']:.3f}]",
                          "control": f"{_sc['control']['copy_acc']:.3f}",
                          "bits per byte": f"{_sc['probe']['bits_per_byte']:.3f}"})
    mo.md("### Byte-level WikiText\n\n"
          + table(e_tab, ["design", "params", "length", "copy accuracy", "control", "bits per byte"]))
    return (e_tab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Figures

        Two figures the paper needs: exact match against the key-value gap, and what the
        dynamic binder picks against the true distance. Both read the JSON written above.
        """
    )
    return


@app.cell
def _(OUT, vg):
    figures = [p for p in (vg.figure_gap(OUT), vg.figure_offsets(OUT)) if p is not None]
    len(figures)
    return (figures,)


@app.cell(hide_code=True)
def _(figures, mo):
    mo.vstack([mo.image(str(p)) for p in figures] or
              [mo.md("*Run stage A or C first; the figures are drawn from their JSON.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## What to write up

        The paper this notebook supports has one story: **a global layer retrieves in one
        step only if the key is bound to its value, and the span of that binding has to
        match the data.** Keep the sink work as a short section that reports what the
        measurement supports, and state the scope in the title, as in
        "in small hybrids on a synthetic retrieval task".

        Before submitting anywhere, check the things that cost the last submission:
        every co-author's account verified, an AI use statement that matches what was
        actually used, and an anonymous code link in the submitted PDF.
        """
    )
    return


if __name__ == "__main__":
    app.run()
