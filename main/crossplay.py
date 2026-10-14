"""N x N crossplay win-rate matrix.

Shells out to duel.py (the canonical, correct seat-wiring dueler -- ego on LEFT,
adv on RIGHT) for every pairing, so correctness lives in ONE place. Each
participant contributes both an ego (e.g. Vega) and an adv (e.g. Guile) from the
SAME checkpoint; the diagonal is therefore self-play (Vega vs its own Guile).

Row avg  = ego/protagonist strength (higher = stronger).
Col avg  = adv weakness            (higher = weaker adversary).

Usage:
  python main/crossplay.py \
    --participant spar:spar:/abs/spar_ckpt.task \
    --participant ippo:ippo:/abs/ippo_ckpt.task \
    --participant v2:spar:/abs/v2_ckpt.task \
    --rounds 8 --ego_char Vega --adv_char Guile \
    --decision_timing joint --dwell_frames 4 --actionable_statuses 512,514,520 \
    --out logs/crossplay.txt

Each --participant is  label:model_type:ckpt_path  (repeat the flag per run).
model_type is duel.py's ego/adv model type (spar | ippo | league | ...).
"""
import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

DUEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "duel.py")


def run_duel(ego_mt, adv_mt, ego_ck, adv_ck, a):
    cmd = [sys.executable, DUEL,
           "--ego_model_type", ego_mt, "--adv_model_type", adv_mt,
           "--num_rounds", str(a.rounds),
           "--ego_char", a.ego_char, "--adv_char", a.adv_char,
           "--ego_model_file", ego_ck, "--adv_model_file", adv_ck,
           "--ego_side", "left", "--device", a.device, "--seed", str(a.seed),
           "--deterministic", a.deterministic, "--transform_action", a.transform_action,
           "--obs_type", a.obs_type,
           "--decision_timing", a.decision_timing, "--dwell_frames", str(a.dwell_frames),
           "--actionable_statuses", a.actionable_statuses]
    # NOTE: duel.py has no --max_skip_frames (its env defaults to 90, matching
    # training); passing it made duel.py exit unrecognized-arg -> every cell nan.
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        return float("nan")
    m = re.findall(r"ego_win_rate=([0-9.]+)", r.stdout)
    return float(m[-1]) if m else float("nan")


def _avg(xs):
    v = [x for x in xs if x == x]
    return sum(v) / len(v) if v else float("nan")


def _fmt(x):
    return " nan " if x != x else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description="N x N crossplay matrix via duel.py")
    ap.add_argument("--participant", action="append", required=True, metavar="label:model_type:ckpt",
                    help="repeat per run; provides an ego AND an adv from the same checkpoint")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel duels (each shells out one duel.py; set ~= cpus-per-task)")
    ap.add_argument("--row", type=int, default=-1,
                    help="if >=0, compute only this ego row and write it to --out (job-array sharding)")
    ap.add_argument("--ego_char", default="Vega")
    ap.add_argument("--adv_char", default="Guile")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", default="False", choices=["True", "False"])
    ap.add_argument("--transform_action", default="True", choices=["True", "False"])
    ap.add_argument("--obs_type", default="image")
    ap.add_argument("--decision_timing", default="joint", choices=["off", "ego", "joint"])
    ap.add_argument("--dwell_frames", type=int, default=4)
    ap.add_argument("--actionable_statuses", default="512,514,520")
    ap.add_argument("--max_skip_frames", type=int, default=90)
    ap.add_argument("--timeout", type=int, default=600, help="per-duel timeout (s)")
    ap.add_argument("--out", default="", help="also write the matrix to this file")
    a = ap.parse_args()

    parts = []
    for p in a.participant:
        bits = p.split(":")
        if len(bits) != 3:
            sys.exit(f"--participant must be label:model_type:ckpt_path, got: {p!r}")
        label, mt, ck = bits
        if not os.path.exists(ck):
            sys.exit(f"checkpoint not found for {label!r}: {ck}")
        parts.append(dict(label=label, mt=mt, ck=ck))

    n = len(parts)
    print(f"crossplay {n}x{n}: rows={a.ego_char}(ego) cols={a.adv_char}(adv)  "
          f"{a.rounds} rounds  dt={a.decision_timing}  (diagonal = self-play)", flush=True)
    M = [[float("nan")] * n for _ in range(n)]

    def _cell(i, j):
        return i, j, run_duel(parts[i]["mt"], parts[j]["mt"],
                              parts[i]["ck"], parts[j]["ck"], a)

    # Row-sharding: if --row I is given, compute ONLY ego-row I (vs all advs) and
    # write it as "<j>\t<label>\t<wr>" lines. Used by the job-array launcher so each
    # task holds just one duel's memory at a time (duels are ~18GB each at n=500).
    if a.row >= 0:
        i = a.row
        if not (0 <= i < n):
            sys.exit(f"--row {i} out of range for {n} participants")
        print(f"row {i} ego={parts[i]['label']} vs {n} advs, {a.rounds} rounds", flush=True)
        rowvals = {}
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(_cell, i, j) for j in range(n)]
            for fut in as_completed(futs):
                _, j, wr = fut.result(); rowvals[j] = wr
                print(f"  [{len(rowvals)}/{n}] {parts[i]['label']} vs {parts[j]['label']}: {_fmt(wr)}", flush=True)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
            with open(a.out, "w") as f:
                f.write(f"# row {i} ego={parts[i]['label']} n={n}\n")
                for j in range(n):
                    f.write(f"{j}\t{parts[j]['label']}\t{_fmt(rowvals[j])}\n")
            print(f"saved row {i}: {a.out}")
        return

    cells = [(i, j) for i in range(n) for j in range(n)]
    # Each duel is a blocking subprocess (duel.py); threads just wait on them,
    # so a thread pool of a.jobs gives a.jobs concurrent duels.
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = [ex.submit(_cell, i, j) for (i, j) in cells]
        done = 0
        for fut in as_completed(futs):
            i, j, wr = fut.result()
            M[i][j] = wr
            done += 1
            print(f"  [{done}/{len(cells)}] {parts[i]['label']}-{a.ego_char} vs "
                  f"{parts[j]['label']}-{a.adv_char}: {_fmt(wr)}", flush=True)

    labels = [p["label"] for p in parts]
    w = max(9, max(len(l) for l in labels) + 2)
    _hdr = 'ego\\adv'
    out = [f"{_hdr:<{w}}" + "".join(f"{l:>{w}}" for l in labels) + f"{'ROW(ego)':>{w}}"]
    for i in range(n):
        out.append(f"{labels[i]:<{w}}" + "".join(f"{_fmt(x):>{w}}" for x in M[i])
                   + f"{_fmt(_avg(M[i])):>{w}}")
    cavg = [_avg([M[i][j] for i in range(n)]) for j in range(n)]
    out.append(f"{'COL(adv)':<{w}}" + "".join(f"{_fmt(x):>{w}}" for x in cavg))
    out.append("row avg = ego strength (higher=stronger); col avg = adv weakness (higher=weaker)")
    txt = "\n".join(out)
    print("\n" + txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt + "\n")
        print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
