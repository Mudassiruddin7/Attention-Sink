"""Sanity check a LaTeX source without compiling it.

Useful when you are writing the paper on a machine with no TeX
installation. It catches the mistakes that cost the most time later,
namely unbalanced environments, table rows with the wrong number of
cells, references with no label, citations with no bibliography entry,
and template markers that were never filled.

    python scripts/check_tex.py ../paper.tex

Exits non zero if anything looks wrong.
"""

from __future__ import annotations

import re
import sys

PROBLEMS = []


def note(msg):
    PROBLEMS.append(msg)
    print("  PROBLEM  " + msg)


def grab_braced(text, i):
    """Return (content, index after closing brace) for the group at text[i]."""
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    return "", i


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../paper.tex"
    s = open(path, encoding="utf-8").read()
    print(f"checking {path}, {len(s.splitlines())} lines")

    if s.count("{") != s.count("}"):
        note(f"brace count differs by {s.count('{') - s.count('}')}")

    envs = {}
    for m in re.finditer(r"\\(begin|end)\{([a-zA-Z*]+)\}", s):
        name = m.group(2)
        envs.setdefault(name, [0, 0])
        envs[name][0 if m.group(1) == "begin" else 1] += 1
    for k, (b, e) in sorted(envs.items()):
        if b != e:
            note(f"environment {k} opened {b} times and closed {e}")
    print(f"  environments: {', '.join(f'{k}x{v[0]}' for k, v in sorted(envs.items()))}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", s))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
    for r in sorted(refs - labels):
        note(f"reference to {r} has no label")
    for l in sorted(labels - refs):
        print(f"  note     label {l} is never referenced")

    cites = set()
    for c in re.findall(r"\\cite\{([^}]+)\}", s):
        cites.update(k.strip() for k in c.split(","))
    bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
    for c in sorted(cites - bibs):
        note(f"citation {c} has no bibliography entry")
    for b in sorted(bibs - cites):
        print(f"  note     bibliography entry {b} is never cited")

    markers = sorted(set(re.findall(r"%%([A-Z_0-9]+)%%", s)))
    if markers:
        note(f"{len(markers)} template marker(s) never filled: "
             f"{', '.join(markers)}")

    for m in re.finditer(r"\\begin\{tabular\}", s):
        spec, after = grab_braced(s, m.end())
        end = s.find("\\end{tabular}", after)
        body = s[after:end]
        # Strip the parts of a column spec that are not columns, namely
        # @{...} separators and >{...} or <{...} cell decorators, then
        # collapse p{width} to a single token before counting.
        clean = re.sub(r"[@><]\{[^{}]*\}", "", spec)
        clean = re.sub(r"p\{[^{}]*\}", "p", clean)
        ncol = sum(1 for ch in clean if ch in "lcrp")
        line_no = s[:m.start()].count("\n") + 1
        for row in body.split("\\\\"):
            row = re.sub(r"\\(top|mid|bottom)rule", "", row)
            row = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", row)
            row = re.sub(r"\\addlinespace", "", row)
            row = row.strip()
            if not row or row.startswith("%"):
                continue
            cells = row.count("&") + 1
            if cells != ncol:
                note(f"table near line {line_no} wants {ncol} cells but a row "
                     f"has {cells}: {row[:60]!r}")

    # House style for this paper.
    body = re.sub(r"(?m)^\s*%.*$", "", s)
    if "---" in body:
        note("an em dash appears outside a comment")

    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} problem(s) found")
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
