"""Predictions fixed before their runs, and their outcomes scored from the saved results.

    python scripts/table_predictions.py

Reads   results/runs/main/*.json, results/runs/text/*.json, results/evals/*.json,
        results/interventions/placement.json
Writes  paper/generated/table_predictions.tex, results/summary_predictions.json

Each prediction is worded as it was written in results/logs/method_notes.md
(sections 15 to 18) before its runs, and its outcome is computed here from the
result files, so the table cannot drift from the data. A prediction whose
results do not exist yet is reported as pending.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

RUNS = "results/runs/main"
TEXT = "results/runs/text"
EVALS = "results/evals"
CONTROL = "results/evals/text_control.json"
SINK_OUT = "results/interventions/placement.json"


def load(name, root=RUNS):
    path = os.path.join(root, f"{name}.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def eval_rows(tag):
    path = os.path.join(EVALS, f"{tag}.json")
    if not os.path.exists(path):
        return None
    return {(x["name"], x["seq_len"]): x for x in json.load(open(path, encoding="utf-8"))["rows"]}


def first_90(run):
    return next((c["step"] for c in run["curve"] if c["train_recall"] >= 0.9), None)


def at(run, length):
    return {e["seq_len"]: e for e in run["evals"]}.get(length)


def name_of(variant, seed):
    return f"{variant}__p0.5__g0__s{seed}"


def haystack_runs(variant, seeds):
    runs = [(s, load(name_of(variant, s))) for s in seeds]
    return None if any(r is None for _, r in runs) else runs


def learned_runs(variant, seeds):
    runs = haystack_runs(variant, seeds)
    return None if runs is None else [(s, r) for s, r in runs if at(r, 256)["recall"] >= 0.9]


def verdict(failed):
    return "held" if not failed else "failed for seed " + ", ".join(map(str, failed))


def step_text(t):
    return "never" if t is None else f"step {t}"


# Stages 1 and 3 ---------------------------------------------------------------

def early(variant, seeds, by=1500):
    runs = haystack_runs(variant, seeds)
    if runs is None:
        return "pending", ""
    late = [s for s, r in runs if first_90(r) is None or first_90(r) > by]
    return verdict(late), "; ".join(f"seed {s}: {step_text(first_90(r))}" for s, r in runs)


def robust(variant, seeds, floor=85.0, length=4096):
    runs = learned_runs(variant, seeds)
    if runs is None:
        return "pending", ""
    values = [(s, 100 * (at(r, length) or {}).get("recall_after_first", float("nan"))) for s, r in runs]
    return verdict([s for s, v in values if not v >= floor]), "; ".join(f"seed {s}: {v:.1f}%" for s, v in values)


def early_and_robust(variant, seeds):
    (o1, d1), (o2, d2) = early(variant, seeds), robust(variant, seeds)
    if "pending" in (o1, o2):
        return "pending", ""
    return ("held" if o1 == o2 == "held" else "failed"), f"{d1} | after first slot at 4096: {d2}"


def not_before(variant, seeds, before=1500):
    runs = haystack_runs(variant, seeds)
    if runs is None:
        return "pending", ""
    early_seeds = [s for s, r in runs if first_90(r) is not None and first_90(r) < before]
    return verdict(early_seeds), "; ".join(f"seed {s}: {step_text(first_90(r))}" for s, r in runs)


def never(variant, seeds, budget=4000):
    runs = haystack_runs(variant, seeds)
    if runs is None:
        return "pending", ""
    reached = [s for s, r in runs if first_90(r) is not None and first_90(r) <= budget]
    return verdict(reached), "; ".join(f"seed {s}: {step_text(first_90(r))}, {100 * at(r, 256)['recall']:.1f}% at 256"
                                       for s, r in runs)


# WikiText ---------------------------------------------------------------------

def text_pairs(seeds):
    pairs = [(s, load(f"text_hybrid_bka_first__s{s}", TEXT), load(f"text_hybrid_bka__s{s}", TEXT)) for s in seeds]
    return None if any(a is None or b is None for _, a, b in pairs) else pairs


def lm_cost(seeds=(0,), margin=0.05):
    pairs = text_pairs(seeds)
    if pairs is None:
        return "pending", ""
    diffs = [(s, (at(a, 512)["ce_other"] - at(b, 512)["ce_other"]) / math.log(2)) for s, a, b in pairs]
    ok = all(abs(d) < margin for _, d in diffs)
    return ("held" if ok else "failed"), "; ".join(f"seed {s}: {d:+.3f} bits per byte" for s, d in diffs)


def copy_far(seeds=(0,), length=2048):
    pairs = text_pairs(seeds)
    if pairs is None or not os.path.exists(CONTROL):
        return "pending", ""
    ctrl = {(x["name"], x["seq_len"]): x for x in json.load(open(CONTROL, encoding="utf-8"))["rows"]}
    detail, ok = [], True
    for s, a, b in pairs:
        ca, cb = ctrl.get((f"text_hybrid_bka_first__s{s}", length)), ctrl.get((f"text_hybrid_bka__s{s}", length))
        if ca is None or cb is None:
            return "pending", ""
        ga = 100 * (at(a, length)["recall"] - ca["recall"])
        gb = 100 * (at(b, length)["recall"] - cb["recall"])
        ok &= ga >= gb
        detail.append(f"seed {s}: {ga:+.1f} against {gb:+.1f} points")
    return ("held" if ok else "failed"), "; ".join(detail)


# Pending evaluations ------------------------------------------------------------

def regime_stable(variant="hybrid_bka_first", seeds=range(6), margin=2.0):
    rows, runs = eval_rows("regime_copyrich"), learned_runs(variant, seeds)
    if rows is None or runs is None:
        return "pending", ""
    detail, failed = [], []
    for s, r in runs:
        diffs = []
        for length in (256, 4096):
            copyrich = rows.get((name_of(variant, s), length))
            if copyrich is None or at(r, length) is None:
                return "pending", ""
            diffs.append(100 * (copyrich["recall"] - at(r, length)["recall"]))
        if max(abs(d) for d in diffs) >= margin:
            failed.append(s)
        detail.append(f"seed {s}: {diffs[0]:+.1f} at 256, {diffs[1]:+.1f} at 4096")
    return verdict(failed), "; ".join(detail)


def regime_recovers(variant="hybrid_nope_first", seeds=range(4)):
    rows, runs = eval_rows("regime_copyrich"), haystack_runs(variant, seeds)
    if rows is None or runs is None:
        return "pending", ""
    detail, failed = [], []
    for s, r in runs:
        copyrich = rows.get((name_of(variant, s), 256))
        if copyrich is None:
            return "pending", ""
        if not copyrich["recall"] > at(r, 256)["recall"]:
            failed.append(s)
        detail.append(f"seed {s}: {100 * at(r, 256)['recall']:.1f} to {100 * copyrich['recall']:.1f}%")
    return verdict(failed), "; ".join(detail)


def best_reader(row):
    li = max(range(len(row["layers"])),
             key=lambda j: row["layers"][j]["value_mass"][row["layers"][j]["best_head"]])
    return li, row["layers"][li]["best_head"]


def attention_holds(variant="hybrid_bka_first", seeds=range(6), floor=0.25):
    rows, runs = eval_rows("attention_standard"), learned_runs(variant, seeds)
    if rows is None or runs is None:
        return "pending", ""
    detail, failed = [], []
    for s, _ in runs:
        short, far = rows.get((name_of(variant, s), 256)), rows.get((name_of(variant, s), 4096))
        if short is None or far is None:
            return "pending", ""
        li, head = best_reader(short)
        profile = np.array(far["layers"][li]["value_mass_by_depth"][head])
        mean = float(profile.mean())
        if not (mean >= floor and profile.min() >= 0.5 * mean):
            failed.append(s)
        detail.append(f"seed {s}: layer {far['layers'][li]['layer']} head {head}, {mean:.2f} "
                      f"(lowest bin {profile.min():.2f})")
    return verdict(failed), "; ".join(detail)


def first_layer_blind(variant="hybrid_nope_first", seeds=range(4), ceiling=0.05):
    rows, runs = eval_rows("attention_standard"), haystack_runs(variant, seeds)
    if rows is None or runs is None:
        return "pending", ""
    detail, failed = [], []
    for s, _ in runs:
        highest = 0.0
        for length in (256, 4096):
            row = rows.get((name_of(variant, s), length))
            if row is None:
                return "pending", ""
            highest = max(highest, max(next(l for l in row["layers"] if l["layer"] == 0)["value_mass"]))
        if highest > ceiling:
            failed.append(s)
        detail.append(f"seed {s}: highest {highest:.3f}")
    return verdict(failed), "; ".join(detail)


def longer_holds(variant="hybrid_bka_first", seeds=range(6)):
    rows, runs = eval_rows("longer"), learned_runs(variant, seeds)
    if rows is None or runs is None:
        return "pending", ""
    detail, failed = [], []
    for s, _ in runs:
        r8, r16 = rows.get((name_of(variant, s), 8192)), rows.get((name_of(variant, s), 16384))
        if r8 is None or r16 is None:
            return "pending", ""
        a8, a16 = 100 * r8["recall_after_first"], 100 * r16["recall_after_first"]
        if not (a8 >= 90 and a16 >= 80):
            failed.append(s)
        detail.append(f"seed {s}: {a8:.1f}% at 8192, {a16:.1f}% at 16384")
    return verdict(failed), "; ".join(detail)


def pairs_interference():
    r16, r64 = eval_rows("pairs16"), eval_rows("pairs64")
    first, unbound = learned_runs("hybrid_bka_first", range(6)), haystack_runs("hybrid_nope_first", range(4))
    if r16 is None or r64 is None or first is None or unbound is None:
        return "pending", ""

    def drop(variant, s):
        a, b = r16.get((name_of(variant, s), 256)), r64.get((name_of(variant, s), 256))
        return None if a is None or b is None else 100 * (a["recall"] - b["recall"])

    d_first = [(s, drop("hybrid_bka_first", s)) for s, _ in first]
    d_unbound = [(s, drop("hybrid_nope_first", s)) for s, _ in unbound]
    if any(d is None for _, d in d_first + d_unbound):
        return "pending", ""
    gap = np.mean([d for _, d in d_unbound]) - np.mean([d for _, d in d_first])
    ok = all(d <= 5 for _, d in d_first) and gap >= 10
    return ("held" if ok else "failed"), (f"bound-key layer first: {', '.join(f'{d:+.1f}' for _, d in d_first)}; "
                                          f"without bound keys: {', '.join(f'{d:+.1f}' for _, d in d_unbound)}; "
                                          f"difference of mean falls {gap:.1f}")


def sink_unused(variant="hybrid_bka_first", seeds=range(6), margin=5.0):
    runs = learned_runs(variant, seeds)
    if not os.path.exists(SINK_OUT) or runs is None:
        return "pending", ""
    rows = {(x["name"], x["seq_len"], x["bias"]): x for x in json.load(open(SINK_OUT, encoding="utf-8"))["rows"]}
    detail, failed = [], []
    for s, _ in runs:
        changes = []
        for length in (256, 2048):
            kept, removed = rows.get((name_of(variant, s), length, 0.0)), rows.get((name_of(variant, s), length, -math.inf))
            if kept is None or removed is None:
                return "pending", ""
            changes.append(100 * (removed["recall"] - kept["recall"]))
        if max(abs(c) for c in changes) >= margin:
            failed.append(s)
        detail.append(f"seed {s}: {changes[0]:+.1f} at 256, {changes[1]:+.1f} at 2048")
    return verdict(failed), "; ".join(detail)



def paired_earlier(seeds, a="hybrid_bka_first", b="hybrid_bka"):
    ra, rb = haystack_runs(a, seeds), haystack_runs(b, seeds)
    if ra is None or rb is None:
        return "pending", ""
    budget = 10 ** 9
    detail, failed = [], []
    for (s, x), (_, y) in zip(ra, rb):
        ta = first_90(x) or budget
        tb = first_90(y) or budget
        if not ta < tb:
            failed.append(s)
        detail.append(f"seed {s}: {step_text(first_90(x))} against {step_text(first_90(y))}")
    return verdict(failed), "; ".join(detail)


def marker_free_floor(variant, seeds, floor=95.0, length=4096):
    rows, runs = eval_rows("rescore_main"), learned_runs(variant, seeds)
    if rows is None or runs is None:
        return "pending", ""
    detail, failed = [], []
    for s, _ in runs:
        row = rows.get((name_of(variant, s), length))
        if row is None or "recall_no_marker" not in row:
            return "pending", ""
        v = 100 * row["recall_no_marker"]
        if v < floor:
            failed.append(s)
        detail.append(f"seed {s}: {v:.1f}%")
    return verdict(failed), "; ".join(detail)


def rate_after_stage5(seeds=(9, 10)):
    # Both new seeds early, and the learned share beats every layer-last hybrid (Fisher, two-sided).
    outcome, detail = early("hybrid_bka_first", seeds, by=500)
    if outcome == "pending":
        return "pending", ""
    from scipy.stats import fisher_exact
    from stats_placement import GROUPS, load_runs
    runs = load_runs()

    def group(name):
        keep, warmup = GROUPS[name]
        return [r for r in runs if keep(r) and r["warmup"] == warmup]

    a, b = group("SLLL, bound keys"), group("LLLS hybrids")
    ka, kb = sum(r["learned"] for r in a), sum(r["learned"] for r in b)
    p = fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1]
    held = outcome == "held" and p < 0.05
    return ("held" if held else "failed"), (f"{detail} | learned {ka}/{len(a)} against {kb}/{len(b)}, "
                                            f"Fisher p = {p:.3f}")


PREDICTIONS = [
    ("Stage 1", "Every seed of the bound-key layer first passes 90% training recall by step 1500",
     lambda: early("hybrid_bka_first", range(4))),
    ("Stage 1", "Recall after the first query slot at 4096 tokens is at least 85% for every learned seed",
     lambda: robust("hybrid_bka_first", range(4))),
    ("Stage 3", "Seeds 4 and 5 of the bound-key layer first pass 90% by step 1500 and keep at least 85% "
                "after the first query slot at 4096 tokens",
     lambda: early_and_robust("hybrid_bka_first", (4, 5))),
    ("Stage 3", "The same layer placed last, seeds 4 and 5, does not pass 90% before step 1500",
     lambda: not_before("hybrid_bka", (4, 5))),
    ("Stage 3", "The global layer first without bound keys, seeds 2 and 3, does not reach 90% within 4000 steps",
     lambda: never("hybrid_nope_first", (2, 3))),
    ("WikiText", "Held-out loss at 512 bytes differs from the same layer placed last by less than 0.05 bits per byte",
     lm_cost),
    ("WikiText", "Copy accuracy minus the control at 2048 bytes is at least that of the same layer placed last",
     copy_far),
    ("Regime", "Every learned bound-key-layer-first model scores within 2 points of its standard-data recall on "
               "copy-rich data, at 256 and at 4096 tokens", regime_stable),
    ("Regime", "Every global-layer-first model without bound keys scores higher at 256 tokens on copy-rich data "
               "than on the standard data", regime_recovers),
    ("Attention", "In every learned bound-key-layer-first run, the head that reads the value best at 256 tokens "
                  "still puts at least 0.25 on it at 4096, with no depth bin below half of its mean", attention_holds),
    ("Attention", "In the four global-layer-first runs without bound keys, no head of layer 0 puts more than 0.05 "
                  "on the value, at 256 or 4096 tokens", first_layer_blind),
    ("Longer", "Every learned bound-key-layer-first model keeps at least 90% recall after the first query slot at "
               "8192 tokens and at least 80% at 16384", longer_holds),
    ("Pairs", "From 16 to 64 pairs at 256 tokens, every learned bound-key-layer-first model loses at most 5 points, "
              "and the models without bound keys lose at least 10 points more on average", pairs_interference),
    ("Sink", "Removing attention to position 0 changes recall of every learned bound-key-layer-first model by less "
             "than 5 points, at 256 and at 2048 tokens", sink_unused),
    ("Stage 4", "Every new bound-key-layer-first seed passes 90% training recall by step 500",
     lambda: early("hybrid_bka_first", (6, 7, 8), by=500)),
    ("Stage 4", "Each new bound-key-layer-first seed reaches 90% earlier than the layer-last seed beside it",
     lambda: paired_earlier((6, 7, 8))),
    ("Stage 4", "With the markers excluded, every new bound-key-layer-first seed keeps at least 95% recall at 4096",
     lambda: marker_free_floor("hybrid_bka_first", (6, 7, 8), floor=95.0)),
    ("Stage 5", "Both new bound-key-layer-first seeds pass 90% training recall by step 500, and the success-rate "
                "comparison against the layer-last hybrids reaches Fisher p < 0.05", rate_after_stage5),
]


def tex(s):
    return s.replace("%", "\\%")


def main():
    rows = []
    for stage, prediction, score in PREDICTIONS:
        outcome, detail = score()
        rows.append({"stage": stage, "prediction": prediction, "outcome": outcome, "detail": detail})
        print(f"[{outcome}] {stage}: {prediction}" + (f"\n    {detail}" if detail else ""))
    os.makedirs("paper/generated", exist_ok=True)
    with open("paper/generated/table_predictions.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(f"{r['stage']} & {tex(r['prediction'])} & {r['outcome']} & {tex(r['detail'])} \\\\"
                          for r in rows) + "\n")
    with open("results/summary_predictions.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
