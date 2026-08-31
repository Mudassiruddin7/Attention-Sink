"""More findings: three read from the saved runs, three from the pending evaluations.

    python scripts/more_findings.py

From the saved training runs (no GPU):

    thresholds   discovery of the bound-key layer first against the LLLS hybrids
                 when learning is counted at 50, 80, 90 or 95% training recall
    sink timing  sink mass on the probes just before and after each hybrid
                 discovers retrieval, sink mass in runs that never did, and sink
                 mass at no-op against copy queries over every learned run
    uniformity   at 4096 tokens, the lowest 95% lower bound over the ten depth
                 bins, as trained and with the length scaling on. Recall here
                 includes the first query slot, which caps the LLLS runs near 87.5%

From the evaluations of scripts/run_pending.sh, once they exist:

    longer       recall and recall after the first slot at 8192 and 16384 tokens
    pairs        recall with 16, 32 and 64 key-value pairs at 256 and 2048 tokens
    sink         recall with attention to position 0 removed, against it kept

Reads   results/runs/main/*.json, results/evals/*.json, results/interventions/placement.json
Writes  results/summary_more_findings.json and, for the parts with data,
        paper/generated/table_thresholds.tex, table_uniformity.tex, table_longer.tex,
        table_pairs.tex, table_sink_placement.tex
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.stats import CensoredData, logrank

sys.path.insert(0, os.path.dirname(__file__))

import stats_placement as sp   # noqa: E402

RUNS = "results/runs/main"
EVALS = "results/evals"
SINK_OUT = "results/interventions/placement.json"
LABELS = {"hybrid_bka_first": "Bound-key layer first", "hybrid_bka": "Bound keys, layer last",
          "hybrid_nope_first": "Layer first, no bound keys", "hybrid_nope": "Layer last, NoPE",
          "hybrid_attnres": "Layer last, NoPE, AttnRes"}
ORDER = list(LABELS)


def eval_rows(tag):
    path = os.path.join(EVALS, f"{tag}.json")
    return json.load(open(path, encoding="utf-8"))["rows"] if os.path.exists(path) else None


def write_tex(name, lines):
    os.makedirs("paper/generated", exist_ok=True)
    with open(f"paper/generated/{name}", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def thresholds(runs, raw, out):
    first = [r for r in runs if r["variant"] == "hybrid_bka_first" and r["warmup"] == 1500]
    llls = [r for r in runs if r["layout"] == "LLLS" and r["delta"] and r["warmup"] == 1500]
    print("Discovery at other thresholds (runs not learned at the end are censored at 4000 steps)")
    result, lines = {}, []
    for level in (0.5, 0.8, 0.9, 0.95):
        times = [[next((c["step"] for c in raw[r["name"]]["curve"] if c["train_recall"] >= level), None)
                  if r["learned"] else None for r in group] for group in (first, llls)]
        censored = [CensoredData(uncensored=[t for t in ts if t is not None],
                                 right=[sp.BUDGET] * sum(t is None for t in ts)) for ts in times]
        p = float(logrank(*censored).pvalue)
        seen = [sorted(t for t in ts if t is not None) for ts in times]
        result[str(level)] = {"bound_key_first": times[0], "llls": times[1], "logrank_p": p}
        print(f"  {level:.0%}: bound-key layer first {seen[0]}; LLLS {seen[1]} + {times[1].count(None)} not learned;"
              f" log-rank p = {p:.1e}")
        lines.append(f"{round(100 * level)}\\% & {len(seen[0])}/{len(first)}, step {max(seen[0], default=0)} "
                     f"at the latest & {len(seen[1])}/{len(llls)}, steps {min(seen[1], default=0)} to "
                     f"{max(seen[1], default=0)} & {sp.fmt_p(p)} \\\\")
    out["thresholds"] = result
    write_tex("table_thresholds.tex", lines)


def sink_timing(runs, raw, out):
    hybrids = [r for r in runs if r["layout"] in ("LLLS", "SLLL") and r["delta"] and r["warmup"] == 1500]
    rows, changes = [], []
    for r in hybrids:
        probes = {p["step"]: p for p in raw[r["name"]]["probes"]}
        row = {"run": r["name"], "learned": r["learned"], "discovery": r["t90"],
               "sink_final": probes[max(probes)]["sink_mass"]}
        if r["learned"] and r["t90"] is not None and r["layout"] == "LLLS":
            before = max((s for s in probes if s < r["t90"]), default=None)
            after = min((s for s in probes if s >= r["t90"]), default=None)
            if before is not None and after is not None:
                row.update(sink_before=probes[before]["sink_mass"], sink_after=probes[after]["sink_mass"])
                changes.append(probes[after]["sink_mass"] - probes[before]["sink_mass"])
        rows.append(row)
    unlearned = [x["sink_final"] for x in rows if not x["learned"]]
    learned = [x["sink_final"] for x in rows if x["learned"]]
    enrich = [(r["name"], r["evals"][256]["sink_noop"] - r["evals"][256]["sink_copy"]) for r in runs
              if r["learned"] and r["warmup"] == 1500 and "sink_noop" in r["evals"][256]
              and "sink_copy" in r["evals"][256]]
    k = sum(d > 0 for _, d in enrich)
    p = float(stats.binomtest(k, len(enrich)).pvalue)
    out["sink_timing"] = {"runs": rows, "change_at_discovery": changes,
                          "noop_enrichment": {"noop_above_copy": k, "n": len(enrich), "sign_test_p": p,
                                              "exceptions": [n for n, d in enrich if d <= 0]}}
    print("Sink mass and discovery (probes every 500 steps)")
    print(f"  LLLS runs, from the probe before discovery to the probe after: {sum(c > 0 for c in changes)} up, "
          f"{sum(c < 0 for c in changes)} down, {sum(c == 0 for c in changes)} unchanged "
          f"(changes {', '.join(f'{c:+.3f}' for c in changes)})")
    print(f"  final sink mass: learned hybrids {min(learned):.3f} to {max(learned):.3f}; "
          f"hybrids that never learned {min(unlearned):.3f} to {max(unlearned):.3f}")
    print(f"  sink at no-op above copy queries in {k} of {len(enrich)} learned runs, sign test p = {p:.1e}")


def uniformity(runs, ev, out):
    result, lines = defaultdict(list), []
    print("Uniformity at 4096 tokens: lowest 95% lower bound over the ten depth bins")
    for variant in ORDER:
        for r in [r for r in runs if r["variant"] == variant and r["warmup"] == 1500 and r["learned"]]:
            settings = [("as trained", r["logn_ref"] > 0)] + ([] if r["logn_ref"] > 0 else [("scaling on", True)])
            for label, scaling in settings:
                e = sp.long_eval(ev, r, scaling)
                if e is None or "profile" not in e:
                    continue
                lowest = 100 * min(b["lo"] for b in e["profile"])
                result[variant].append({"run": r["name"], "setting": label if r["logn_ref"] == 0 else "scaling on",
                                        "recall": 100 * e["recall"], "lowest_bound": lowest})
    for variant, items in result.items():
        for setting in ("as trained", "scaling on"):
            sel = [x for x in items if x["setting"] == setting]
            if not sel:
                continue
            low = [x["lowest_bound"] for x in sel]
            print(f"  {LABELS[variant]:28s} {setting:10s} {len(sel)} runs: lowest bin bound {min(low):.1f} to {max(low):.1f}")
            lines.append(f"{LABELS[variant]} & {setting} & {len(sel)} & {min(low):.1f}--{max(low):.1f} \\\\")
    out["uniformity"] = result
    write_tex("table_uniformity.tex", lines)


def longer(out):
    rows = (eval_rows("longer") or []) + [dict(x, variant=x["variant"] + " (scaling on)")
                                          for x in (eval_rows("longer_llls_nope_logn") or [])]
    if not rows:
        out["longer"] = "pending"
        print("Longer contexts: pending")
        return
    groups, lines = defaultdict(list), []
    for x in rows:
        groups[(x["variant"], x["seq_len"])].append(x)
    print("Longer contexts, learned models: recall / after the first slot, by seed")
    for (variant, length), xs in sorted(groups.items()):
        rec = [100 * x["recall"] for x in xs]
        after = [100 * x.get("recall_after_first", float("nan")) for x in xs]
        print(f"  {variant:34s} {length:6d}: {', '.join(f'{a:.1f}/{b:.1f}' for a, b in zip(rec, after))}")
        lines.append(f"{variant.replace('_', ' ')} & {length} & {len(xs)} & {np.mean(rec):.1f} & "
                     f"{np.nanmean(after):.1f} & {np.nanmin(after):.1f} \\\\")
    out["longer"] = {f"{v}@{n}": xs for (v, n), xs in groups.items()}
    write_tex("table_longer.tex", lines)


def pairs(out):
    tables = {n: eval_rows(f"pairs{n}") for n in (16, 32, 64)}
    if any(t is None for t in tables.values()):
        out["pairs"] = "pending"
        print("More key-value pairs: pending")
        return
    recall = defaultdict(dict)
    learned = {}
    for n, rows in tables.items():
        for x in rows:
            recall[(x["name"], x["variant"], x["seq_len"])][n] = 100 * x["recall"]
            learned[x["name"]] = x["learned"]
    by_variant, lines = defaultdict(list), []
    for (name, variant, length), r in sorted(recall.items()):
        if len(r) == 3:
            by_variant[(variant, length)].append((name, r[16], r[32], r[64]))
    print("More key-value pairs: recall with 16, 32 and 64 pairs (drop from 16 to 64)")
    for (variant, length), items in sorted(by_variant.items()):
        drops = [a - c for _, a, _, c in items]
        print(f"  {variant:20s} {length:5d}: " + "; ".join(f"{n.split('__')[-1]}{'' if learned[n] else '*'} "
                                                          f"{a:.0f}/{b:.0f}/{c:.0f}" for n, a, b, c in items)
              + f" | mean drop {np.mean(drops):.1f}")
        lines.append(f"{LABELS.get(variant, variant)} & {length} & {len(items)} & "
                     f"{np.mean([a for _, a, _, _ in items]):.1f} & {np.mean([b for _, _, b, _ in items]):.1f} & "
                     f"{np.mean([c for _, _, _, c in items]):.1f} & {np.mean(drops):.1f} \\\\")
    print("  (* not learned at the training length)")
    out["pairs"] = {f"{v}@{n}": items for (v, n), items in by_variant.items()}
    write_tex("table_pairs.tex", lines)


def sink_removal(out):
    if not os.path.exists(SINK_OUT):
        out["sink_removal"] = "pending"
        print("Sink removal in the placement models: pending")
        return
    rows = json.load(open(SINK_OUT, encoding="utf-8"))["rows"]
    kept = {(x["name"], x["seq_len"]): x for x in rows if x["bias"] == 0}
    removed = {(x["name"], x["seq_len"]): x for x in rows if x["bias"] == float("-inf")}
    by_variant, lines = defaultdict(list), []
    for key, x in removed.items():
        if key in kept:
            by_variant[(x["variant"], key[1])].append((key[0], 100 * (x["recall"] - kept[key]["recall"]),
                                                       kept[key].get("sink_mass", float("nan"))))
    print("Sink removal: recall change with attention to position 0 removed, by learned model")
    for (variant, length), items in sorted(by_variant.items()):
        changes = [c for _, c, _ in items]
        print(f"  {variant:20s} {length:5d}: {', '.join(f'{c:+.1f}' for c in changes)} "
              f"(sink kept {np.nanmean([s for _, _, s in items]):.3f})")
        lines.append(f"{LABELS.get(variant, variant)} & {length} & {len(items)} & {np.mean(changes):+.1f} & "
                     f"{min(changes):+.1f} to {max(changes):+.1f} \\\\")
    out["sink_removal"] = {f"{v}@{n}": items for (v, n), items in by_variant.items()}
    write_tex("table_sink_placement.tex", lines)


def main():
    runs, ev = sp.load_runs(), sp.load_evals()
    raw = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8"))
           for p in glob.glob(os.path.join(RUNS, "*.json"))}
    out = {}
    thresholds(runs, raw, out)
    sink_timing(runs, raw, out)
    uniformity(runs, ev, out)
    longer(out)
    pairs(out)
    sink_removal(out)
    with open("results/summary_more_findings.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
