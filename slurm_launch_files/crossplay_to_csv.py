#!/usr/bin/env python3
"""Convert a crossplay.py --out matrix .txt into a CSV (same basename .csv).
crossplay.py already writes the formatted NxN table with ROW(ego)/COL(adv)
margins; this just reshapes the grid for offline tools. Usage:
  crossplay_to_csv.py <matrix.txt> [out.csv]
"""
import sys, re, csv

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else re.sub(r"\.txt$", "", src) + ".csv"
lines = [l.rstrip("\n") for l in open(src)]

hdr_i = next((i for i, l in enumerate(lines) if l.lstrip().startswith("ego\\adv")), None)
if hdr_i is None:
    sys.exit("no 'ego\\adv' header found -- not a crossplay matrix file")

cols = lines[hdr_i].split()[1:]            # includes trailing ROW(ego)
rows = []
for l in lines[hdr_i + 1:]:
    toks = l.split()
    if not toks or toks[0] == "COL(adv)" or not re.match(r"[A-Za-z]", toks[0]):
        continue
    rows.append(toks)
    if toks[0].startswith("COL"):
        break

with open(dst, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["ego\\adv"] + cols)
    for r in rows:
        wr.writerow(r)
print("wrote", dst, f"({len(rows)} rows x {len(cols)} cols incl margins)")
