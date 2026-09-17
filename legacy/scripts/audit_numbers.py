"""Check that numbers quoted in the paper prose match the raw results.

The tables are generated, so they cannot drift. The prose is written by
hand and can. This script recomputes the handful of figures that the
prose quotes and reports any that no longer match the result files.

    python scripts/audit_numbers.py ../paper_built.tex
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys

TOL = 0.06  # a quoted percentage may be rounded


def ms(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../paper_built.tex"
    text = open(path, encoding="utf-8").read()

    pilot = json.load(open("results/pilot.json", encoding="utf-8"))
    ctrl = json.load(open("results/control_noaux.json", encoding="utf-8"))
    cost = json.load(open("results/costmodel.json", encoding="utf-8"))

    g = {}
    for r in pilot["runs"]:
        g.setdefault(r["variant"], []).append(r)

    def at(variant, idx, key, runs=None):
        rs = runs or g[variant]
        return ms([r["evals"][idx][key] for r in rs])[0]

    checks = []

    def expect(label, value, fmt="{:.3f}"):
        s = fmt.format(value)
        checks.append((label, s, s in text))

    # Sink mass and worst layer at the training length.
    expect("baseline sink mass", 100 * at("dense", 0, "sink_mass"), "{:.1f}")
    expect("baseline worst layer", at("dense", 0, "sink_mass_max_layer"))
    expect("gated sink mass", at("dense_gated", 0, "sink_mass"))
    expect("hybrid sink mass", at("hybrid", 0, "sink_mass"))

    # Control.
    cruns = ctrl["runs"]
    expect("control sink mass", 100 * at(None, 0, "sink_mass", cruns), "{:.1f}")
    expect("control worst layer", at(None, 0, "sink_mass_max_layer", cruns))
    expect("control recall", 100 * at(None, 0, "accuracy", cruns), "{:.1f}")
    expect("baseline recall", 100 * at("dense", 0, "accuracy"), "{:.1f}")

    # Per seed sink masses quoted in the prose.
    for r in g["dense"]:
        expect(f"baseline seed {r['seed']} sink",
               r["evals"][0]["sink_mass"])
    for r in cruns:
        expect(f"control seed {r['seed']} sink",
               r["evals"][0]["sink_mass"])

    # Activations.
    expect("baseline activation", at("dense", 0, "max_activation"), "{:.1f}")
    expect("gated activation", at("dense_gated", 0, "max_activation"), "{:.1f}")
    expect("hybrid activation", at("hybrid", 0, "max_activation"), "{:.1f}")
    expect("hybrid_ar activation", at("hybrid_ar", 0, "max_activation"),
           "{:.1f}")

    # Gate score.
    expect("gate score", at("dense_gated", 0, "gate_score"))

    # Recall at the training length for the hybrid rows.
    expect("hybrid recall", 100 * at("hybrid", 0, "accuracy"), "{:.1f}")
    expect("hybrid_ar recall", 100 * at("hybrid_ar", 0, "accuracy"), "{:.1f}")

    # Cost model.
    frac = 100 * cost["constant_layer_fraction"]
    expect("constant layer fraction", frac, "{:.1f}")

    bad = [c for c in checks if not c[2]]
    for label, value, ok in checks:
        print(f"  {'ok ' if ok else 'MISSING'}  {label:<28} {value}")
    print()
    if bad:
        print(f"{len(bad)} quoted value(s) not found in the paper. Either the "
              f"prose is stale or the value is phrased differently.")
        return 1
    print(f"all {len(checks)} quoted values found in the paper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
