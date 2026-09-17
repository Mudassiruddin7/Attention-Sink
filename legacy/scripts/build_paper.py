"""Substitute generated tables and plots into the paper source.

The paper source carries markers of the form %%TABLE_MAIN%% and
%%PLOT_SINK%%. This script replaces each one with the fragment that
scripts/make_results_tex.py produced from the raw results, and replaces
each %%PROSE_...%% marker with the matching block from paper/prose.tex.

    python scripts/build_paper.py ../paper.tex ../paper_built.tex

Any marker left unfilled is reported rather than silently dropped, so the
build fails loudly if a result is missing.
"""

from __future__ import annotations

import json
import os
import re
import sys

MARKER = re.compile(r"%%([A-Z_0-9]+)%%")


def load_prose(path="paper/prose.tex"):
    """Prose blocks separated by lines of the form %%% NAME."""
    if not os.path.exists(path):
        return {}
    blocks, name, buf = {}, None, []
    for line in open(path, encoding="utf-8"):
        if line.startswith("%%% "):
            if name:
                blocks[name] = "".join(buf).strip()
            name, buf = line[4:].strip(), []
        else:
            buf.append(line)
    if name:
        blocks[name] = "".join(buf).strip()
    return blocks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="../paper.tex")
    ap.add_argument("dst", nargs="?", default="../paper_built.tex")
    ap.add_argument("--fragments", default="paper/fragments.json")
    ap.add_argument("--prose", default="paper/prose.tex")
    args = ap.parse_args()
    src, dst = args.src, args.dst

    with open(args.fragments, encoding="utf-8") as f:
        frag = json.load(f)
    frag.update(load_prose(args.prose))

    text = open(src, encoding="utf-8").read()
    missing, filled = [], []

    def sub(m):
        key = m.group(1)
        if key in frag:
            filled.append(key)
            return frag[key]
        missing.append(key)
        return m.group(0)

    out = MARKER.sub(sub, text)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"filled {len(filled)} markers into {dst}")
    for k in filled:
        print("  ok      ", k)
    if missing:
        print(f"\n{len(missing)} markers had no content and were left in place:")
        for k in sorted(set(missing)):
            print("  missing ", k)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
