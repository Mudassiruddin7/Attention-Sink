"""Compute the cache growth table for the Kimi K3 layer mix.

Two of the numbers this needs are not published in the K3 report, namely
the latent rank of the Gated MLA layers and the head dimensions of the
Kimi Delta Attention state. They are declared as assumptions in
sinkprobe/costmodel.py and printed with the results so that anyone can
substitute the real values later.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinkprobe.costmodel import HybridSpec, gib, table   # noqa: E402

LENGTHS = [4_096, 32_768, 131_072, 262_144, 524_288, 1_048_576]


def main():
    spec = HybridSpec()
    rows = table(spec, LENGTHS)
    payload = {"assumptions": spec.summary(),
               "constant_layer_fraction": spec.constant_layer_fraction(),
               "linear_state_gib": gib(spec.linear_state_bytes()),
               "rows": rows}
    os.makedirs("results", exist_ok=True)
    with open("results/costmodel.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"layers with a cache that does not grow with context: "
          f"{spec.n_linear_layers}/{spec.n_layers} "
          f"= {100 * spec.constant_layer_fraction():.1f} percent")
    print(f"fixed recurrent state across all linear layers: "
          f"{gib(spec.linear_state_bytes()):.3f} GiB")
    print()
    head = f"{'tokens':>10} {'global GiB':>11} {'state GiB':>10} " \
           f"{'hybrid GiB':>11} {'dense GiB':>10} {'ratio':>7}"
    print(head)
    for r in rows:
        print(f"{r['tokens']:>10,} {r['global_cache_gib']:>11.2f} "
              f"{r['linear_state_gib']:>10.2f} {r['hybrid_total_gib']:>11.2f} "
              f"{r['dense_cache_gib']:>10.2f} {r['reduction_x']:>6.1f}x")
    print("\nwrote results/costmodel.json")


if __name__ == "__main__":
    main()
