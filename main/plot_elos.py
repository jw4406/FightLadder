import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_elo_history(elo_data_dir):
    path = os.path.join(elo_data_dir, "elo_history.jsonl")
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _canonical_matchup(matchup_name):
    parts = matchup_name.replace("Vs", "_vs_").split("_vs_")
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        return f"{a}_vs_{b}"
    return matchup_name


def _build_series(records):
    """Build per-matchup time series of ego and adversary ratings."""
    ego_series = {"timesteps": [], "ratings": []}
    matchup_series = defaultdict(lambda: {"timesteps": [], "ego_ratings": [], "adv_ratings": []})

    for rec in sorted(records, key=lambda r: r["timestep"]):
        ts = rec["timestep"]
        ego_rating = rec["rating_ego"]
        ego_series["timesteps"].append(ts)
        ego_series["ratings"].append(ego_rating)

        for matchup_name, mdata in rec.get("matchups", {}).items():
            canonical = _canonical_matchup(matchup_name)
            series = matchup_series[canonical]
            series["timesteps"].append(ts)
            series["ego_ratings"].append(ego_rating)
            series["adv_ratings"].append(mdata["rating_adv"])

    return ego_series, dict(matchup_series)


def _plot_master(ego_series, matchup_series, out_dir):
    fig, ax = plt.subplots(figsize=(14, 8))

    matchup_keys = sorted(matchup_series.keys())
    cmap = matplotlib.colormaps.get_cmap("tab20")
    denom = max(1, len(matchup_keys) - 1)
    color_by_matchup = {k: cmap(i / denom) for i, k in enumerate(matchup_keys)}

    ax.plot(
        ego_series["timesteps"],
        ego_series["ratings"],
        color="black",
        linewidth=2.0,
        alpha=0.9,
        label="Ego",
        zorder=10,
    )

    for matchup_key in matchup_keys:
        series = matchup_series[matchup_key]
        display = matchup_key.replace("_vs_", " vs ")
        color = color_by_matchup[matchup_key]

        ax.plot(
            series["timesteps"],
            series["adv_ratings"],
            color=color,
            linewidth=1.2,
            alpha=0.8,
            label=f"{display} (adv)",
        )

    ax.set_title("Elo Ratings: All Matchups")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Elo Rating")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2, handletextpad=0.5, columnspacing=1.6)

    output_path = os.path.join(out_dir, "elo_master.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_per_matchup(ego_series, matchup_series, out_dir):
    paths = []
    for matchup_key in sorted(matchup_series.keys()):
        series = matchup_series[matchup_key]
        display = matchup_key.replace("_vs_", " vs ")

        fig, ax = plt.subplots(figsize=(10, 6))

        parts = matchup_key.split("_vs_")
        ego_name = parts[0] if len(parts) == 2 else "Ego"
        adv_name = parts[1] if len(parts) == 2 else "Adversary"

        ax.plot(
            series["timesteps"],
            series["ego_ratings"],
            color="#1f77b4",
            linewidth=1.4,
            marker="o",
            markersize=3,
            alpha=0.9,
            label=f"{ego_name} (ego)",
        )
        ax.plot(
            series["timesteps"],
            series["adv_ratings"],
            color="#ff7f0e",
            linewidth=1.4,
            marker="^",
            markersize=3,
            alpha=0.9,
            label=f"{adv_name} (adv)",
        )

        ax.set_title(f"Elo Ratings: {display}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Elo Rating")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        out_name = f"elo_{matchup_key}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Plot Elo ratings from training history.")
    parser.add_argument(
        "--elo_data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "elo_data"),
        help="Folder containing elo_history.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "elo_plots"),
        help="Folder for saved plots.",
    )
    args = parser.parse_args()

    records = _load_elo_history(args.elo_data_dir)
    if not records:
        print(f"No Elo data found in: {args.elo_data_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    ego_series, matchup_series = _build_series(records)

    print("=== Elo data summary ===")
    print(f"Data folder:    {args.elo_data_dir}")
    print(f"Records:        {len(records)}")
    print(f"Matchups:       {sorted(matchup_series.keys())}")
    print(f"Timestep range: {records[0]['timestep']} - {records[-1]['timestep']}")
    print(f"Ego rating:     {records[0]['rating_ego']:.0f} -> {records[-1]['rating_ego']:.0f}")

    master_path = _plot_master(ego_series, matchup_series, args.output_dir)
    matchup_paths = _plot_per_matchup(ego_series, matchup_series, args.output_dir)

    print("\nSaved plots:")
    print(f"  - {master_path}")
    for p in matchup_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
