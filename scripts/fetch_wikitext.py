"""Download the two WikiText files that the text probes read.

    python scripts/fetch_wikitext.py

Writes data/wikitext2_train.txt (WikiText-2 raw, train split) and
data/wikitext103_validation.txt (WikiText-103 raw, validation split), one
non-empty line per row. The files are not in the repository; WikiText is
distributed by its authors under CC BY-SA 3.0.

The runs in the paper used files with these MD5 sums (the lines were written
in Windows text mode, so every line ends in CRLF):
    wikitext2_train.txt          fe13cbb023d86ecc1b37651e51392d9d
    wikitext103_validation.txt   3f9dcb27e8a17844c256fff3a077c477
Pass --crlf to write the same line endings on any platform.
"""

from __future__ import annotations

import argparse
import hashlib
import os

from datasets import load_dataset

FILES = {
    "data/wikitext2_train.txt": ("wikitext-2-raw-v1", "train"),
    "data/wikitext103_validation.txt": ("wikitext-103-raw-v1", "validation"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crlf", action="store_true", help="end every line with CRLF")
    a = ap.parse_args()
    eol = b"\r\n" if a.crlf else b"\n"
    os.makedirs("data", exist_ok=True)
    for path, (config, split) in FILES.items():
        rows = load_dataset("Salesforce/wikitext", config, split=split)["text"]
        lines = [r.rstrip("\n") for r in rows if r.strip()]
        data = eol.join(line.encode("utf-8") for line in lines) + eol
        with open(path, "wb") as f:
            f.write(data)
        print(f"{path}: {len(lines)} lines, md5 {hashlib.md5(data).hexdigest()}")


if __name__ == "__main__":
    main()
