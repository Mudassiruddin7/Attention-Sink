"""Merge per variant result files into one results/pilot.json.

run_pilot.py can be pointed at a single variant, which lets the four
variants train side by side instead of one after another. This script
stitches the pieces back together so that the rest of the pipeline sees
the same file it would have seen from a single run.

    python scripts/merge_parts.py results/part_*.json
"""

from __future__ import annotations

import glob
import json
import os
import sys


def main():
    patterns = sys.argv[1:] or ["results/part_*.json"]
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        print("no part files matched")
        return 1

    merged = None
    seen = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            part = json.load(f)
        if merged is None:
            merged = {k: v for k, v in part.items() if k != "runs"}
            merged["runs"] = []
            merged["merged_from"] = []
        merged["merged_from"].append(os.path.basename(path))
        for run in part["runs"]:
            key = (run["variant"], run["seed"])
            if key in seen:
                print(f"  skipping duplicate {key} from {path}")
                continue
            seen.add(key)
            merged["runs"].append(run)
        print(f"  {os.path.basename(path)}: {len(part['runs'])} run(s)")

    # Keep the variant order stable for downstream consumers.
    order = {"dense": 0, "dense_gated": 1, "hybrid": 2, "hybrid_ar": 3}
    merged["runs"].sort(key=lambda r: (order.get(r["variant"], 9), r["seed"]))

    with open("results/pilot.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    variants = sorted({r["variant"] for r in merged["runs"]})
    print(f"\nwrote results/pilot.json with {len(merged['runs'])} runs "
          f"across {len(variants)} variants: {', '.join(variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
