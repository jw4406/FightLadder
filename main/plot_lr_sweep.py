#!/usr/bin/env python3
"""Cross-config comparison for an LR-ratio sweep.

Scans every $WORKDIR/lr_sweep/<tag>/FightLadder/main/br_rewards tree, builds one
exploitability curve per config, ranks the configs by a (swappable) scalar, and
overlays the top-K curves on a single axis.

Curve construction reuses aggregate_local_eval_data.py's parser and its
canonical-vs-periodic selection (so this stays consistent with the per-config
plots), then collapses each ego training step to the mean exploiter value across
all matchups / directions / replicates -> one (step, value) series per config.

Ranking is a pluggable function of that curve (see RANK_FUNCS). The prototype,
"final_gap", scores by the value at the largest ego step (lower = less
exploitable = better). Add alternatives to RANK_FUNCS without touching main().

Usage:
    python main/plot_lr_sweep.py --workdir /scratch/gpfs/FISAC/jw4406/
    python main/plot_lr_sweep.py --workdir ... --top_k 0        # overlay ALL configs
    python main/plot_lr_sweep.py --workdir ... --rank_by final_gap
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the tested parsing + selection so curves match the per-config plots.
from aggregate_local_eval_data import (
    _parse_records,
    _select_canonical_or_latest_periodic,
    _discover_training_processes,
)


def build_curve(config_br_rewards_dir):
    """Return sorted [(ego_step, value)] for one config: the mean exploiter
    value across all matchups / directions / replicates at each ego step."""
    records = []
    for _label, sub in _discover_training_processes(config_br_rewards_dir):
        records += _parse_records(sub)
    if not records:
        return []
    records = _select_canonical_or_latest_periodic(records)
    by_step = {}
    for r in records:
        by_step.setdefault(r["timestep"], []).append(r["value"])
    return [(step, sum(v) / len(v)) for step, v in sorted(by_step.items())]


# ---------------------------------------------------------------------------
# Ranking scalars: curve [(step, value), ...] -> float, LOWER = better.
# Swap by adding a function here and referencing it via --rank_by.
# ---------------------------------------------------------------------------
def rank_final_gap(curve):
    """Prototype: the exploitability value at the largest ego step (the final gap)."""
    return curve[-1][1] if curve else float("inf")


RANK_FUNCS = {
    "final_gap": rank_final_gap,
    # e.g. "late_mean": lambda c: mean of last 20%; "best": lambda c: min(v ...);
    # "auc": trapezoid area. Each takes a curve and returns a float (lower better).
}


def discover_configs(workdir):
    """(tag, br_rewards_dir) for every lr_sweep/<tag> tree that has a br_rewards dir."""
    root = os.path.join(workdir.rstrip("/"), "lr_sweep")
    pattern = os.path.join(root, "*", "FightLadder", "main", "br_rewards")
    out = []
    for br in sorted(glob.glob(pattern)):
        if os.path.isdir(br):
            tag = os.path.relpath(br, root).split(os.sep)[0]
            out.append((tag, br))
    return out


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True,
                   help="Scratch WORKDIR; globs $WORKDIR/lr_sweep/*. Run where it is mounted.")
    p.add_argument("--top_k", type=int, default=15,
                   help="Overlay the top-K ranked configs. <=0 overlays ALL (set high to see all).")
    p.add_argument("--rank_by", choices=sorted(RANK_FUNCS), default="final_gap",
                   help="Ranking scalar (lower = better).")
    p.add_argument("--out_dir", default=os.path.join(here, "lr_sweep_compare"),
                   help="Where the ranking CSV + overlay figure are written.")
    return p.parse_args()


def main():
    args = parse_args()
    rank_fn = RANK_FUNCS[args.rank_by]

    configs = discover_configs(args.workdir)
    if not configs:
        sys.exit(f"No br_rewards under {args.workdir.rstrip('/')}/lr_sweep/*/FightLadder/"
                 "main/br_rewards (is $WORKDIR mounted here?).")

    rows = []
    for tag, br in configs:
        curve = build_curve(br)
        if not curve:
            print(f"  [skip] {tag}: no reward records yet")
            continue
        rows.append({"tag": tag, "curve": curve, "score": rank_fn(curve),
                     "final_step": curve[-1][0], "n_points": len(curve)})
    if not rows:
        sys.exit("Found config trees but none had reward data yet.")

    rows.sort(key=lambda r: r["score"])  # lower = better
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "lr_sweep_ranking.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "tag", f"score[{args.rank_by}]", "final_step", "n_points"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["tag"], f"{r['score']:.6g}", r["final_step"], r["n_points"]])

    print(f"\nRanking by {args.rank_by} (lower = better) — {len(rows)} configs:")
    for i, r in enumerate(rows, 1):
        print(f"  {i:3d}. {r['tag']:18s} score={r['score']:.4g}  "
              f"(final_step={r['final_step']}, pts={r['n_points']})")
    print(f"\nCSV: {csv_path}")

    k = len(rows) if args.top_k <= 0 else min(args.top_k, len(rows))
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = matplotlib.colormaps.get_cmap("viridis")
    for idx, r in enumerate(rows[:k]):
        xs = [s for s, _ in r["curve"]]
        ys = [v for _, v in r["curve"]]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2,
                color=cmap(idx / max(1, k - 1)), label=r["tag"])
    ax.set_xlabel("ego training step")
    ax.set_ylabel("exploiter value (exploitability gap)")
    ax.set_title(f"LR-ratio sweep — top {k} of {len(rows)} by {args.rank_by} "
                 f"(lower = better; color = rank)")
    ax.grid(True, alpha=0.25)
    if k <= 25:  # a legend of hundreds is unreadable; only show it when small
        ax.legend(fontsize=8, ncol=2)
    fig_path = os.path.join(args.out_dir, f"lr_sweep_overlay_top{k}.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay ({k} curves): {fig_path}")


if __name__ == "__main__":
    main()
