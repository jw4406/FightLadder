#!/usr/bin/env python3
"""Merge a base run and its continuation(s) into ONE logical run for the
exploitability-gap plot, so aggregate_local_eval_data.py renders them as a
single very-long run on a CUMULATIVE timestep axis.

Motivation: a continuation warm-started via ippo.py set_parameters() restarts
num_timesteps at 0, so its br_rewards / selfplay_rewards filenames carry
continuation-RELATIVE timesteps (1.92M, 3.84M, ...). Left alone they overlap
the base run's early points on the x-axis. This tool copies each run's reward
files into one combined subfolder, rewriting ONLY the timestep field to
timestep + offset, so the base run's points (offset 0) and the continuation's
points (offset = base final step) form a continuous, non-overlapping series.

It reuses aggregate_local_eval_data.FILENAME_RE, so any file the aggregator can
parse is offset correctly and the output is guaranteed parseable; anything it
cannot parse is skipped with a warning rather than silently mangled. A name
collision in the output (two source files mapping to the same offset filename --
i.e. offsets that overlap) is a hard error, never a silent overwrite.

Example (view ent05 followed by its ent03 continuation as one ~159M run):
  python merge_continuation_curve.py \
      --run spar_Ve_Bl_ent05:0 \
      --run spar_Ve_Bl_ent03:78720000 \
      --out spar_Ve_Bl_ent05_cont --aggregate
"""
import argparse
import os
import shutil
import subprocess
import sys

from aggregate_local_eval_data import FILENAME_RE


def parse_run_spec(spec):
    """'subdir:offset' -> (subdir, int offset). rsplit so subdir may contain ':'."""
    if ":" not in spec:
        sys.exit(f"--run spec must be 'SUBDIR:OFFSET', got {spec!r}")
    sub, off = spec.rsplit(":", 1)
    if not sub:
        sys.exit(f"--run spec has empty subdir: {spec!r}")
    try:
        off = int(off)
    except ValueError:
        sys.exit(f"--run offset must be an integer, got {off!r} in {spec!r}")
    if off < 0:
        sys.exit(f"--run offset must be >= 0, got {off} in {spec!r}")
    return sub, off


def offset_filename(name, offset):
    """Return *name* with only its timestep field shifted by *offset*, or None
    if the aggregator's canonical regex does not parse it."""
    m = FILENAME_RE.match(name)
    if not m:
        return None
    new_ts = int(m.group("timestep")) + offset
    return name[: m.start("timestep")] + str(new_ts) + name[m.end("timestep") :]


def merge_one_family(src_base, runs, out_sub):
    """Copy+offset every parseable .txt from each run subdir into out_sub.

    Returns (copied, skipped, out_dir). Exits loudly on an output-name
    collision so overlapping offsets can never silently clobber a point.
    """
    out_dir = os.path.join(src_base, out_sub)
    os.makedirs(out_dir, exist_ok=True)
    copied = skipped = 0
    seen = {}  # new_name -> "sub/entry" of the file that claimed it
    for sub, offset in runs:
        d = os.path.join(src_base, sub)
        if not os.path.isdir(d):
            print(f"  [warn] missing run dir, skipped: {d}")
            continue
        for entry in sorted(os.listdir(d)):
            if not entry.endswith(".txt"):
                continue
            new = offset_filename(entry, offset)
            if new is None:
                print(f"  [warn] unparseable, skipped: {sub}/{entry}")
                skipped += 1
                continue
            if new in seen:
                sys.exit(
                    f"COLLISION in '{out_sub}': {sub}/{entry} and {seen[new]} "
                    f"both map to {new} -- offsets overlap. Aborting so no "
                    f"point is silently overwritten."
                )
            seen[new] = f"{sub}/{entry}"
            shutil.copy2(os.path.join(d, entry), os.path.join(out_dir, new))
            copied += 1
    return copied, skipped, out_dir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run", action="append", required=True, metavar="SUBDIR:OFFSET",
        help="A run subfolder and its cumulative-step offset. Repeatable. "
             "e.g. --run spar_Ve_Bl_ent05:0 --run spar_Ve_Bl_ent03:78720000",
    )
    ap.add_argument("--out", required=True, help="Name of the combined subfolder to write.")
    ap.add_argument("--br_base", default=None,
                    help="br_rewards base dir (default: main/br_rewards next to this script).")
    ap.add_argument("--selfplay_base", default=None,
                    help="selfplay_rewards base dir (default: sibling of --br_base).")
    ap.add_argument("--aggregate", action="store_true",
                    help="After merging, run aggregate_local_eval_data.py on the combined subfolder.")
    ap.add_argument("--output_dir", default=None, help="Plot output dir passed to --aggregate.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    br_base = args.br_base or os.path.join(here, "br_rewards")
    sp_base = args.selfplay_base or br_base.replace("br_rewards", "selfplay_rewards")

    runs = [parse_run_spec(s) for s in args.run]
    print(f"Merging {len(runs)} run(s) into '{args.out}':")
    for sub, off in runs:
        print(f"  {sub:32s} +{off} (+{off/1e6:.2f}M)")

    print(f"[br_rewards] base={br_base}")
    bc, bs, br_out = merge_one_family(br_base, runs, args.out)
    print(f"  copied {bc}, skipped {bs}  -> {br_out}")

    print(f"[selfplay]   base={sp_base}")
    if os.path.isdir(sp_base):
        sc, ss, _ = merge_one_family(sp_base, runs, args.out)
        print(f"  copied {sc}, skipped {ss}")
    else:
        print(f"  [warn] selfplay base not found: {sp_base} (overlay will be skipped)")

    if bc == 0:
        sys.exit("ERROR: no br_rewards files merged -- nothing to plot.")

    if args.aggregate:
        cmd = [
            sys.executable, os.path.join(here, "aggregate_local_eval_data.py"),
            "--br_rewards_dir", br_base,
            "--selfplay_rewards_dir", sp_base,
            "--training_process", args.out,
        ]
        if args.output_dir:
            cmd += ["--output_dir", args.output_dir]
        print("Running aggregate:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
