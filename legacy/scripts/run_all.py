"""Run the whole pipeline in order and stop at the first failure."""

import subprocess
import sys

STEPS = [
    ["scripts/run_pilot.py"],
    ["scripts/run_costmodel.py"],
    ["scripts/make_figures.py"],
    ["scripts/make_results_tex.py"],
]


def main():
    extra = sys.argv[1:]
    for i, step in enumerate(STEPS):
        args = [sys.executable] + step + (extra if i == 0 else [])
        print("=" * 60)
        print(" ".join(args))
        print("=" * 60, flush=True)
        code = subprocess.call(args)
        if code != 0:
            print(f"step failed with code {code}")
            return code
    print("all steps complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
