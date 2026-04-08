import argparse
import os
import re
from collections import defaultdict

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILENAME_RE = re.compile(
    r"^(?P<timestep>\d+)_main_(?P<main_side>left|right)_(?P<main_char>[A-Za-z0-9]+)"
    r"_exploiter_(?P<exploiter_side>left|right)_(?P<exploiter_char>[A-Za-z0-9]+)_\.txt$"
)


def _safe_float_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return float(content)


def _matchup_from_record(rec):
    """
    Recover the left/right state matchup from a parsed filename record.
    """
    main_side = rec["main_side"]
    exploiter_side = rec["exploiter_side"]
    main_char = rec["main_char"]
    exploiter_char = rec["exploiter_char"]

    if main_side == "left" and exploiter_side == "right":
        left_char, right_char = main_char, exploiter_char
    elif main_side == "right" and exploiter_side == "left":
        left_char, right_char = exploiter_char, main_char
    else:
        raise ValueError(
            "Unexpected side assignment in file "
            f"{rec['filename']}: main_side={main_side}, exploiter_side={exploiter_side}"
        )
    return left_char, right_char


def _state_string(left_char, right_char):
    return (
        f"two_player/{left_char}_left/"
        f"Champion.Level1.{left_char}Vs{right_char}.2Player.state"
    )


def _canonical_matchup(left_char, right_char):
    a, b = sorted([left_char, right_char])
    return f"{a}_vs_{b}"


def _parse_records(br_rewards_dir):
    records = []
    for entry in sorted(os.listdir(br_rewards_dir)):
        if not entry.endswith(".txt"):
            continue
        match = FILENAME_RE.match(entry)
        if match is None:
            # Ignore non-conforming files to support mixed historical outputs.
            continue

        rec = match.groupdict()
        rec["filename"] = entry
        rec["timestep"] = int(rec["timestep"])
        rec["path"] = os.path.join(br_rewards_dir, entry)
        rec["value"] = _safe_float_from_file(rec["path"])
        left_char, right_char = _matchup_from_record(rec)
        rec["left_char"] = left_char
        rec["right_char"] = right_char
        rec["state"] = _state_string(left_char, right_char)
        rec["matchup_key"] = _canonical_matchup(left_char, right_char)
        rec["direction"] = "main_left" if rec["main_side"] == "left" else "main_right"
        records.append(rec)
    return records


def _build_pairs(records):
    """
    Group by timestep + canonical matchup.
    Each group can contain up to two directional members (main_left/main_right).
    """
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["timestep"], rec["matchup_key"])].append(rec)
    return grouped


def _infer_sets(records):
    states = sorted({rec["state"] for rec in records})
    left_set = sorted({rec["left_char"] for rec in records})
    right_set = sorted({rec["right_char"] for rec in records})
    matchups = sorted({rec["matchup_key"] for rec in records})
    timesteps = sorted({rec["timestep"] for rec in records})
    return states, left_set, right_set, matchups, timesteps


def _plot_master(records, pairs_by_timestep_matchup, out_dir):
    fig, ax = plt.subplots(figsize=(14, 8))

    matchup_keys = sorted({rec["matchup_key"] for rec in records})
    cmap = matplotlib.colormaps.get_cmap("tab20")
    denom = max(1, len(matchup_keys) - 1)
    color_by_matchup = {k: cmap(i / denom) for i, k in enumerate(matchup_keys)}
    marker_by_direction = {"main_left": "o", "main_right": "^"}

    # Scatter all points
    for rec in records:
        ax.scatter(
            rec["timestep"],
            rec["value"],
            color=color_by_matchup[rec["matchup_key"]],
            marker=marker_by_direction[rec["direction"]],
            s=55,
            alpha=0.9,
        )

    # Connect paired points at same timestep/matchup (if both exist)
    for (_, _), group in pairs_by_timestep_matchup.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda x: x["direction"])
        xs = [g["timestep"] for g in group_sorted]
        ys = [g["value"] for g in group_sorted]
        ax.plot(xs, ys, color=color_by_matchup[group_sorted[0]["matchup_key"]], alpha=0.35, linewidth=1.0)

    ax.set_title("Local BR Eval: All Matchups (paired directions)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)

    # Build legend entries so each row is:
    #   "<matchup> (main_left)"    "<matchup> (main_right)"
    # With ncol=2, matplotlib fills by columns, so we pass all left labels first
    # and then all right labels to keep per-matchup rows aligned.
    left_handles = []
    right_handles = []
    for matchup_key in matchup_keys:
        display_matchup = matchup_key.replace("_vs_", " vs ")
        color = color_by_matchup[matchup_key]
        left_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker_by_direction["main_left"],
                color=color,
                linestyle="None",
                markersize=7,
                label=f"{display_matchup} (main_left)",
            )
        )
        right_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker_by_direction["main_right"],
                color=color,
                linestyle="None",
                markersize=7,
                label=f"{display_matchup} (main_right)",
            )
        )

    legend_handles = left_handles + right_handles
    legend_labels = [h.get_label() for h in legend_handles]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        fontsize=8,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.6,
    )

    output_path = os.path.join(out_dir, "master_local_eval_pairs.png")
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_per_matchup(records, out_dir):
    paths = []
    records_by_matchup = defaultdict(list)
    for rec in records:
        records_by_matchup[rec["matchup_key"]].append(rec)

    marker_by_direction = {"main_left": "o", "main_right": "^"}
    color_by_direction = {"main_left": "#1f77b4", "main_right": "#ff7f0e"}

    for matchup_key, m_records in sorted(records_by_matchup.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        m_records = sorted(m_records, key=lambda x: (x["timestep"], x["direction"]))

        for direction in ("main_left", "main_right"):
            d_recs = [r for r in m_records if r["direction"] == direction]
            if not d_recs:
                continue
            xs = [r["timestep"] for r in d_recs]
            ys = [r["value"] for r in d_recs]
            ax.plot(
                xs,
                ys,
                marker=marker_by_direction[direction],
                color=color_by_direction[direction],
                linewidth=1.2,
                markersize=5,
                alpha=0.9,
                label=direction,
            )

        ax.set_title(f"Local BR Eval: {matchup_key}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        out_name = f"matchup_{matchup_key}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--br_rewards_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "br_rewards"),
        help="Folder containing local br reward txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_eval_plots"),
        help="Folder for saved plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = _parse_records(args.br_rewards_dir)
    if not records:
        print(f"No parseable reward files found in: {args.br_rewards_dir}")
        return

    pairs = _build_pairs(records)
    states, left_set, right_set, matchups, timesteps = _infer_sets(records)

    print("=== Inferred local eval data summary ===")
    print(f"Rewards folder: {args.br_rewards_dir}")
    print(f"Parsed file count: {len(records)}")
    print(f"Timestep count: {len(timesteps)}")
    print(f"Matchup count: {len(matchups)}")
    print(f"left_set (ego_list candidate): {left_set}")
    print(f"right_set (adv_list candidate): {right_set}")
    print("Inferred state_list:")
    for s in states:
        print(f"  - {s}")

    complete_pairs = 0
    incomplete_pairs = 0
    for key, group in pairs.items():
        if len(group) >= 2:
            complete_pairs += 1
        else:
            incomplete_pairs += 1

    print("Pairing summary (timestep + matchup groups):")
    print(f"  complete pairs: {complete_pairs}")
    print(f"  incomplete pairs (kept): {incomplete_pairs}")

    master_path = _plot_master(records, pairs, args.output_dir)
    matchup_plot_paths = _plot_per_matchup(records, args.output_dir)

    print("Saved plots:")
    print(f"  - {master_path}")
    for p in matchup_plot_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
