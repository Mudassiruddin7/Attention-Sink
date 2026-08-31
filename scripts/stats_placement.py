"""Statistics for the layer-placement study (method notes, section 15).

    python scripts/stats_placement.py

Reads   results/runs/main/*.json, results/evals/*.json
Writes  results/summary_placement.json,
        paper/generated/table_placement_groups.tex, table_placement_tests.tex

Runs with p_noop 0.5 and no query skew. Time to discovery is the first logged
step (every 100 steps) at which training recall reaches 90%, for runs that
count as learned at the end; other runs are right-censored at the 4000-step
budget, since a run can pass 90% on warm-up batches and lose it when the data
change. Discovery times are
compared with the log-rank test and with an exact two-sided permutation test
of the rank sum (mid-ranks, so runs that never learned tie above every
observed time). Learned shares carry Clopper-Pearson 95% intervals and are
compared with Fisher's exact test. Recall past the training length is
compared among learned runs with the same permutation test.
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os

import numpy as np
from scipy import stats
from scipy.stats import CensoredData, logrank

RUNS = "results/runs/main"
EVALS = "results/evals"
BUDGET = 4000
LONG = 4096
GROUPS = {  # name: (which runs, copy-rich warm-up steps)
    "attention only": (lambda r: r["layout"] == "S", 1500),
    "LLLS hybrids": (lambda r: r["layout"] == "LLLS" and r["delta"], 1500),
    "LLLS NoPE hybrids": (lambda r: r["variant"] in ("hybrid_nope", "hybrid_attnres", "hybrid_bka"), 1500),
    "LLLS, bound keys": (lambda r: r["variant"] == "hybrid_bka", 1500),
    "SLLL, bound keys": (lambda r: r["variant"] == "hybrid_bka_first", 1500),
    "SLLL, NoPE": (lambda r: r["variant"] == "hybrid_nope_first", 1500),
    "LLLS hybrids, no warm-up": (lambda r: r["layout"] == "LLLS" and r["delta"], 0),
    "SLLL, bound keys, no warm-up": (lambda r: r["variant"] == "hybrid_bka_first", 0),
}
DISCOVERY = [("SLLL, bound keys", "LLLS hybrids"), ("SLLL, bound keys", "attention only"),
             ("attention only", "LLLS hybrids"), ("SLLL, NoPE", "LLLS hybrids"),
             ("SLLL, NoPE", "SLLL, bound keys"),
             ("SLLL, bound keys, no warm-up", "LLLS hybrids, no warm-up")]
PAIRED = ("SLLL, bound keys", "LLLS, bound keys")
LONG_RECALL = [("SLLL, bound keys", "LLLS NoPE hybrids"), ("SLLL, bound keys", "attention only"),
               ("SLLL, NoPE", "LLLS NoPE hybrids")]
# The same comparisons with the length scaling on for every run, which is fairer to
# designs trained without it (it needs no retraining).
LONG_RECALL_SCALED = [("SLLL, bound keys", "LLLS NoPE hybrids")]
SCALING = ["SLLL, bound keys", "SLLL, NoPE", "LLLS NoPE hybrids"]
# Evaluations at LONG tokens made without retraining, by whether the length scaling was on.
AS_TRAINED_TAGS = ("hybrid_bka_4096", "long_baselines")
SCALING_ON_TAGS = ("llls_nope_logn", "nope_logn")
SCALING_OFF_TAGS = ("hybrid_bka_first_nologn", "hybrid_nope_first_nologn", "hybrid_bka_nologn", "bka_nologn")


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        tc = r["task_config"]
        if tc.get("p_noop") is None or abs(tc["p_noop"] - 0.5) > 1e-9 or abs(tc.get("query_skew", 0.0)) > 1e-9:
            continue
        ev = {e["seq_len"]: e for e in r["evals"]}
        mc = r["model_config"]
        learned = ev[r["run"]["train_len"]]["recall"] >= 0.9
        first90 = next((c["step"] for c in r["curve"] if c["train_recall"] >= 0.9), None)
        runs.append({"name": os.path.basename(p)[:-5], "variant": r["run"]["variant"],
                     "seed": r["run"]["seed"], "warmup": r["run"].get("copy_warmup", 0),
                     "layout": mc.get("layout", "S"), "delta": mc.get("delta", True),
                     "logn_ref": mc.get("logn_ref", 0), "evals": ev,
                     "learned": learned, "first90": first90, "t90": first90 if learned else None})
    return runs


def load_evals():
    out = {}
    for p in glob.glob(os.path.join(EVALS, "*.json")):
        rows = json.load(open(p, encoding="utf-8"))["rows"]
        out[os.path.basename(p)[:-5]] = {(x["name"], x["seq_len"]): x for x in rows}
    return out


def long_eval(ev, r, scaling):
    """Evaluation of run r at LONG tokens with the length scaling on or off, or None."""
    if scaling == (r["logn_ref"] > 0):
        e = r["evals"].get(LONG)
        if e is not None and "recall_after_first" in e:
            return e
        for tag in AS_TRAINED_TAGS:
            x = ev.get(tag, {}).get((r["name"], LONG))
            if x is not None:
                return x
        return e
    for tag in (SCALING_ON_TAGS if scaling else SCALING_OFF_TAGS):
        x = ev.get(tag, {}).get((r["name"], LONG))
        if x is not None:
            return x
    return None


def long_values(ev, group, metric, scaling=None):
    """Metric at LONG tokens for the learned runs of a group, as trained or with scaling set."""
    vals = []
    for r in group:
        e = long_eval(ev, r, r["logn_ref"] > 0 if scaling is None else scaling)
        if r["learned"] and e is not None and metric in e:
            vals.append(100 * e[metric])
    return vals


def clopper_pearson(x, n):
    lo = stats.beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = stats.beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return float(lo), float(hi)


def rank_sum_p(a, b, max_exact=3_000_000, n_mc=200_000):
    """Two-sided permutation p for the rank sum of a, exact when feasible."""
    x = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    ranks2 = (2 * stats.rankdata(x)).astype(int).tolist()       # doubled mid-ranks are integers
    n, k = len(ranks2), len(a)
    centre = k * (n + 1)
    observed = abs(sum(ranks2[:k]) - centre)
    if math.comb(n, k) <= max_exact:
        hits = total = 0
        for idx in itertools.combinations(ranks2, k):
            total += 1
            hits += abs(sum(idx) - centre) >= observed
        return hits / total
    rng = np.random.default_rng(0)
    arr = np.array(ranks2)
    sims = np.array([abs(arr[rng.permutation(n)[:k]].sum() - centre) for _ in range(n_mc)])
    return float((1 + (sims >= observed).sum()) / (n_mc + 1))


def times(group):
    return [BUDGET + 1 if r["t90"] is None else r["t90"] for r in group]


def censored(group):
    return CensoredData(uncensored=[r["t90"] for r in group if r["t90"] is not None],
                        right=[BUDGET] * sum(r["t90"] is None for r in group))


def fmt_p(p):
    if p is None:
        return "--"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.0e}"


def write_tables(out):
    """LaTeX rows for the paper: one per group, and one per test."""
    groups = []
    for name, g in out["groups"].items():
        steps = sorted(s for s in g["steps_to_90"].values() if s is not None)
        never = sum(s is None for s in g["steps_to_90"].values())
        when = ", ".join(map(str, steps)) if steps else "--"
        if never:
            when += f"; {never} not within {BUDGET}"
        groups.append(f"{name} & {g['learned']}/{g['n']} & {100 * g['ci95'][0]:.0f}--{100 * g['ci95'][1]:.0f} "
                      f"& {when} \\\\")
    tests = []
    for d in out["discovery"]:
        xa, na, xb, nb = d["learned"]
        tests.append(f"{d['a']} vs {d['b']}: discovery & {xa}/{na} vs {xb}/{nb} learned & "
                     f"Fisher {fmt_p(d['fisher_p'])}; log-rank {fmt_p(d['logrank_p'])}; "
                     f"rank sum {fmt_p(d['rank_sum_p'])} \\\\")
    if out["paired"]:
        pr = out["paired"]
        tests.append(f"{pr['a']} vs {pr['b']}: matched seeds & faster on {pr['a_faster']}, slower on "
                     f"{pr['a_slower']} & sign test {fmt_p(pr['sign_test_p'])} \\\\")
    for d in out["long"]:
        metric = "recall" if d["metric"] == "recall" else "recall after the first slot"
        tests.append(f"{d['a']} vs {d['b']}: {metric} at {LONG}, {d['setting']} & "
                     f"{np.mean(d['a_values']):.1f} vs {np.mean(d['b_values']):.1f} & "
                     f"permutation {fmt_p(d['p'])} \\\\")
    os.makedirs("paper/generated", exist_ok=True)
    with open("paper/generated/table_placement_groups.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(groups) + "\n")
    with open("paper/generated/table_placement_tests.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tests) + "\n")


def main():
    runs, ev = load_runs(), load_evals()
    groups = {name: [r for r in runs if keep(r) and r["warmup"] == w] for name, (keep, w) in GROUPS.items()}
    out = {"groups": {}, "discovery": [], "paired": None, "long": [], "scaling": {}, "profile_range_4096": {}}

    print("Groups: learned (Clopper-Pearson 95%), steps to 90% training recall")
    for name, g in groups.items():
        if not g:
            continue
        x = sum(r["learned"] for r in g)
        lo, hi = clopper_pearson(x, len(g))
        reached = sorted(r["t90"] for r in g if r["t90"] is not None)
        never = sum(r["t90"] is None for r in g)
        out["groups"][name] = {"learned": x, "n": len(g), "ci95": [lo, hi],
                               "steps_to_90": {r["name"]: r["t90"] for r in g}}
        print(f"  {name}: {x}/{len(g)} [{100 * lo:.0f}, {100 * hi:.0f}]%; {reached}"
              + (f" + {never} not within {BUDGET}" if never else ""))

    print("Discovery")
    for a, b in DISCOVERY:
        ga, gb = groups[a], groups[b]
        if not ga or not gb:
            continue
        xa, xb = sum(r["learned"] for r in ga), sum(r["learned"] for r in gb)
        fisher = float(stats.fisher_exact([[xa, len(ga) - xa], [xb, len(gb) - xb]])[1])
        try:
            lr = float(logrank(censored(ga), censored(gb)).pvalue)
        except Exception:                       # no observed event in either group
            lr = None
        rs = rank_sum_p(times(ga), times(gb))
        out["discovery"].append({"a": a, "b": b, "learned": [xa, len(ga), xb, len(gb)],
                                 "fisher_p": fisher, "logrank_p": lr, "rank_sum_p": rs})
        print(f"  {a} vs {b}: learned {xa}/{len(ga)} vs {xb}/{len(gb)} (Fisher p = {fisher:.3f}); "
              f"log-rank p = {'n/a' if lr is None else f'{lr:.1e}'}; rank-sum p = {rs:.1e}")

    a, b = PAIRED
    by_seed = {r["seed"]: r for r in groups[b]}
    pairs = [(r["seed"], r["t90"], by_seed[r["seed"]]["t90"]) for r in groups[a] if r["seed"] in by_seed]
    if pairs:
        key = lambda t: BUDGET + 1 if t is None else t    # noqa: E731
        faster = sum(key(ta) < key(tb) for _, ta, tb in pairs)
        slower = sum(key(ta) > key(tb) for _, ta, tb in pairs)
        p = float(stats.binomtest(faster, faster + slower, 0.5).pvalue) if faster + slower else None
        out["paired"] = {"a": a, "b": b, "pairs": pairs, "a_faster": faster, "a_slower": slower, "sign_test_p": p}
        print(f"Paired by seed, {a} vs {b}: {pairs}; faster {faster}, slower {slower}, sign test p = {p}")

    for setting, scaling, pairs in (("as trained", None, LONG_RECALL),
                                    ("length scaling on for every run", True, LONG_RECALL_SCALED)):
        print(f"Recall at {LONG} tokens, learned runs, {setting}")
        for a, b in pairs:
            for metric in ("recall", "recall_after_first"):
                va = long_values(ev, groups[a], metric, scaling)
                vb = long_values(ev, groups[b], metric, scaling)
                if not va or not vb:
                    continue
                p = rank_sum_p(va, vb)
                out["long"].append({"setting": setting, "a": a, "b": b, "metric": metric,
                                    "a_values": va, "b_values": vb, "p": p})
                print(f"  {metric}: {a} {np.round(va, 1).tolist()} (mean {np.mean(va):.1f}) vs {b} "
                      f"{np.round(vb, 1).tolist()} (mean {np.mean(vb):.1f}); permutation p = {p:.3f}")

    print(f"Length scaling on vs off at {LONG} tokens, learned runs, same weights")
    for name in SCALING:
        rows = []
        for r in groups[name]:
            on, off = long_eval(ev, r, True), long_eval(ev, r, False)
            if r["learned"] and on is not None and off is not None:
                rows.append({"run": r["name"], "on": 100 * on["recall"], "off": 100 * off["recall"]})
        if rows:
            out["scaling"][name] = rows
            print(f"  {name}: " + ", ".join(f"{x['run'].replace('__p0.5__g0', '')} {x['on']:.1f} vs {x['off']:.1f}" for x in rows)
                  + f" | mean {np.mean([x['on'] for x in rows]):.1f} vs {np.mean([x['off'] for x in rows]):.1f}")

    for name in ("SLLL, bound keys", "SLLL, NoPE", "LLLS NoPE hybrids"):
        bins = [100 * b["acc"] for r in groups[name] if r["learned"]
                and (e := long_eval(ev, r, r["logn_ref"] > 0)) is not None for b in e["profile"]]
        if bins:
            out["profile_range_4096"][name] = [min(bins), max(bins)]
            print(f"Depth-bin recall at {LONG}, {name}: {min(bins):.0f} to {max(bins):.0f}")

    with open("results/summary_placement.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    write_tables(out)


if __name__ == "__main__":
    main()
