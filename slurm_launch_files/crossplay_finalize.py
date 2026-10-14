#!/usr/bin/env python3
"""Assemble a row-sharded crossplay run into the full matrix + CSV.
Reads <ws>/participants.txt (label:mt:path per line) and <ws>/row_<i>.txt
(lines "<j>\\t<label>\\t<wr>") and writes <prefix>.txt + <prefix>.csv.
usage: crossplay_finalize.py <workspace_dir> [out_prefix]
"""
import sys, os, csv

ws = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ws, "matrix")
labels = [ln.split(":", 1)[0] for ln in open(os.path.join(ws, "participants.txt")) if ln.strip()]
n = len(labels)
M = [[float("nan")] * n for _ in range(n)]
missing = []
for i in range(n):
    rf = os.path.join(ws, f"row_{i}.txt")
    if not os.path.exists(rf):
        missing.append(i); continue
    for ln in open(rf):
        if ln.startswith("#") or not ln.strip():
            continue
        j, _lab, wr = ln.rstrip("\n").split("\t")
        try:
            M[i][int(j)] = float(wr)
        except ValueError:
            M[i][int(j)] = float("nan")

def fmt(x):
    return "nan" if x != x else f"{x:.3f}"
def avg(xs):
    v = [x for x in xs if x == x]
    return sum(v) / len(v) if v else float("nan")

w = max(9, max(len(l) for l in labels) + 2)
hdr = "ego\\adv"
out = [f"{hdr:<{w}}" + "".join(f"{l:>{w}}" for l in labels) + f"{'ROW(ego)':>{w}}"]
for i in range(n):
    out.append(f"{labels[i]:<{w}}" + "".join(f"{fmt(x):>{w}}" for x in M[i])
               + f"{fmt(avg(M[i])):>{w}}")
cavg = [avg([M[i][j] for i in range(n)]) for j in range(n)]
out.append(f"{'COL(adv)':<{w}}" + "".join(f"{fmt(x):>{w}}" for x in cavg))
out.append("row avg = ego strength (higher=stronger); col avg = adv weakness (higher=weaker)")
txt = "\n".join(out)

with open(prefix + ".txt", "w") as f:
    f.write(txt + "\n")
with open(prefix + ".csv", "w", newline="") as f:
    cw = csv.writer(f); cw.writerow([hdr] + labels)
    for i in range(n):
        cw.writerow([labels[i]] + ["" if M[i][j] != M[i][j] else f"{M[i][j]:.3f}" for j in range(n)])

print(txt)
print(f"\nwrote {prefix}.txt and {prefix}.csv")
if missing:
    print(f"MISSING rows (no row_<i>.txt): {missing}")
