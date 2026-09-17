"""Checks for the statistics behind the placement study (scripts/stats_placement.py).

    python tests/test_stats.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import stats_placement as sp   # noqa: E402


def test_rank_sum_matches_exact_mann_whitney_without_ties():
    rng = np.random.default_rng(0)
    for _ in range(25):
        x = rng.permutation(100)[:13].astype(float)
        a, b = x[:4], x[4:]
        expected = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue
        assert abs(sp.rank_sum_p(a, b) - expected) < 1e-12, (a, b)


def test_runs_that_never_learned_tie_above_every_observed_time():
    # Four runs at step 300 against five that never learned: only the split that puts
    # all four early runs in the first group is as extreme as the one observed.
    p = sp.rank_sum_p([300] * 4, [sp.BUDGET + 1] * 5)
    assert abs(p - 1 / math.comb(9, 4)) < 1e-12


def test_clopper_pearson_known_values():
    lo, hi = sp.clopper_pearson(4, 4)
    assert abs(lo - 0.025 ** 0.25) < 1e-9 and hi == 1.0
    lo, hi = sp.clopper_pearson(0, 2)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** 0.5)) < 1e-9


def test_discovery_counts_only_runs_learned_at_the_end():
    # Passing 90% on warm-up batches and losing it afterwards is not a discovery.
    record = {"run": {"variant": "hybrid_nope_first", "seed": 3, "copy_warmup": 1500, "train_len": 256},
              "task_config": {"p_noop": 0.5, "query_skew": 0.0},
              "model_config": {"layout": "SLLL", "delta": True, "logn_ref": 256},
              "evals": [{"seq_len": 256, "recall": 0.828}],
              "curve": [{"step": 1000, "train_recall": 0.72}, {"step": 1100, "train_recall": 0.95},
                        {"step": 1500, "train_recall": 0.25}]}
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "hybrid_nope_first__p0.5__g0__s3.json"), "w", encoding="utf-8") as f:
            json.dump(record, f)
        saved, sp.RUNS = sp.RUNS, tmp
        try:
            (run,) = sp.load_runs()
        finally:
            sp.RUNS = saved
    assert run["first90"] == 1100 and run["t90"] is None and not run["learned"]


def test_long_eval_picks_the_requested_length_scaling():
    name = "hybrid_nope__p0.5__g0__s1"
    ev = {"llls_nope_logn": {(name, 4096): {"recall": 0.881}},
          "long_baselines": {(name, 4096): {"recall": 0.868, "recall_after_first": 0.992}}}
    run = {"name": name, "logn_ref": 0, "evals": {}}
    assert sp.long_eval(ev, run, scaling=True)["recall"] == 0.881
    assert sp.long_eval(ev, run, scaling=False)["recall"] == 0.868
    trained_with_scaling = {"name": "hybrid_bka__p0.5__g0__s9", "logn_ref": 256,
                            "evals": {4096: {"recall": 0.9, "recall_after_first": 1.0}}}
    assert sp.long_eval({}, trained_with_scaling, scaling=True)["recall"] == 0.9
    assert sp.long_eval({}, trained_with_scaling, scaling=False) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
