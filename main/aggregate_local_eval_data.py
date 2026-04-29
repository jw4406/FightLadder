import argparse
import os
import re
from collections import defaultdict

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional <style>_ prefix (league|ippo|spar|2timescale|...) and optional
# <exp_type>_br<idx> suffix are captured but allowed to be absent so legacy
# reward files still match. exp_type ∈ {continue, dedicated} distinguishes
# continue-vs-from-scratch BR runs; br_idx is the per-job replicate index.
FILENAME_RE = re.compile(
    r"^(?:(?P<style>[A-Za-z0-9]+)_)?"
    r"(?P<timestep>\d+)_main_(?P<main_side>left|right)_(?P<main_char>[A-Za-z0-9]+)"
    r"_exploiter_(?P<exploiter_side>left|right)_(?P<exploiter_char>[A-Za-z0-9]+)"
    r"(?:_(?P<exp_type>continue|dedicated)_br(?P<br_idx>\d+))?"
    r"_\.txt$"
)


def _safe_float_from_file(path):
    """
    Read a single float from *path*. Returns None when the file is empty
    or its contents don't parse as a float, so the caller can skip the
    record instead of crashing the whole aggregation.

    Empty files happen when local_br_eval started but didn't reach the
    write step (interrupted run, crashed BR worker, race with another
    process). One bad file shouldn't kill aggregation across many runs.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return None
    try:
        return float(content)
    except ValueError:
        return None


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
        value = _safe_float_from_file(rec["path"])
        if value is None:
            print(f"  [warn] skipping unreadable reward file: {entry}")
            continue
        rec["value"] = value
        left_char, right_char = _matchup_from_record(rec)
        rec["left_char"] = left_char
        rec["right_char"] = right_char
        rec["state"] = _state_string(left_char, right_char)
        rec["matchup_key"] = _canonical_matchup(left_char, right_char)
        rec["direction"] = "main_left" if rec["main_side"] == "left" else "main_right"
        # Style may be None (legacy unprefixed files); normalize to "" so
        # downstream code can treat it uniformly.
        rec["style"] = rec.get("style") or ""
        # Exploiter-type and replicate index are also optional (legacy files
        # don't carry them). Normalize: exp_type "" → treated as "continue"
        # in plotting since legacy data was de-facto continue-mode; br_idx
        # stays None when absent so downstream can detect "no replicate
        # info" if needed.
        rec["exp_type"] = rec.get("exp_type") or ""
        br_idx_str = rec.get("br_idx")
        rec["br_idx"] = int(br_idx_str) if br_idx_str is not None else None
        # Selfplay value populated later by _attach_selfplay_values once
        # the matching sibling folder is known.
        rec["selfplay_value"] = None
        records.append(rec)
    return records


def _aggregate_replicates(records):
    """
    Collapse replicates to one record per
        (timestep, matchup_key, direction, exp_type)
    via arithmetic mean.

    Multiple br_idx files at the same (timestep, matchup, side, exp_type) are
    independent samples of the same exploitability measurement (different
    BR seeds, same target). Averaging them gives a single per-bucket value
    that's stable enough to plot directly without the per-replicate scatter.

    Selfplay values are also averaged across replicates within the bucket.
    The returned list has one record per bucket; br_idx is set to None and
    a new "replicate_count" field exposes how many samples contributed (so
    plot legends or summary lines can surface it if useful).
    """
    grouped = defaultdict(list)
    for rec in records:
        key = (
            rec["timestep"],
            rec["matchup_key"],
            rec["direction"],
            rec["exp_type"] or "continue",
        )
        grouped[key].append(rec)

    aggregated = []
    for _, group in grouped.items():
        template = dict(group[0])
        template["value"] = sum(r["value"] for r in group) / len(group)
        sp_vals = [r.get("selfplay_value") for r in group if r.get("selfplay_value") is not None]
        template["selfplay_value"] = (
            sum(sp_vals) / len(sp_vals) if sp_vals else None
        )
        template["replicate_count"] = len(group)
        template["br_idx"] = None
        # Force exp_type to the canonical bucket value (legacy "" -> "continue").
        template["exp_type"] = group[0]["exp_type"] or "continue"
        aggregated.append(template)
    return aggregated


def _compute_selfplay_means(records):
    """
    Average selfplay_value across the two direction-samples of the same
    (timestep, matchup_key). Returns dict keyed by (timestep, matchup_key),
    value = mean of available selfplay values (1 or 2 samples).

    The two files
       <step>_main_left_<L>_exploiter_right_<R>_.txt
       <step>_main_right_<R>_exploiter_left_<L>_.txt
    are two samples of the same self-play matchup viewed from each side;
    averaging gives a single per-matchup value that's stable across the
    two BR runs that produced them.
    """
    grouped = defaultdict(list)
    for rec in records:
        sp = rec.get("selfplay_value")
        if sp is None:
            continue
        grouped[(rec["timestep"], rec["matchup_key"])].append(sp)
    return {k: (sum(vs) / len(vs)) for k, vs in grouped.items()}


def _attach_selfplay_values(records, selfplay_dir):
    """
    For each record, look up the matching same-named file under
    *selfplay_dir* and attach its float value to rec["selfplay_value"].

    Records without a matching selfplay file (or with an unreadable one)
    keep selfplay_value=None so the plotters can skip the overlay
    cleanly. Returns the count of records that successfully picked up a
    selfplay value.

    The pairing key is the filename: local_br_eval.py writes both BR and
    selfplay outputs under the *same* basename in their respective dirs,
    so we don't need to re-parse the filename to align them.
    """
    if not selfplay_dir or not os.path.isdir(selfplay_dir):
        return 0
    paired = 0
    for rec in records:
        sp_path = os.path.join(selfplay_dir, rec["filename"])
        if not os.path.isfile(sp_path):
            continue
        sp_value = _safe_float_from_file(sp_path)
        if sp_value is None:
            continue
        rec["selfplay_value"] = sp_value
        paired += 1
    return paired


def _dominant_style(records):
    """
    Pick the most common training_style across a record set. Returns ""
    when no record has a style. Logs a warning when styles are mixed
    (shouldn't happen because each subfolder is one training process, but
    legacy directories can have heterogeneous content).
    """
    counts = defaultdict(int)
    for r in records:
        counts[r["style"]] += 1
    if not counts:
        return ""
    # Sort by count desc, then style asc for determinism.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and any(s for s, _ in ranked):
        # Mixed styles in one directory — surface a warning so the user
        # knows the plot filename only reflects the dominant one.
        breakdown = ", ".join(f"{s or '<none>'}={c}" for s, c in ranked)
        print(f"  [warn] mixed training_styles in directory ({breakdown})")
    return ranked[0][0]


def _discover_training_processes(br_rewards_dir):
    """
    Return [(label, abs_path), ...] — one entry per training process.

    New segregated layout: each immediate subfolder is one training process;
    label = subfolder name.

    Legacy unsegregated layout: top-level .txt files exist directly under
    br_rewards_dir; surfaced as a single ("", br_rewards_dir) bucket so old
    data still plots without restructuring on disk.

    Mixed layouts (both subfolders and top-level .txt files) are supported:
    legacy bucket and per-subfolder buckets coexist.
    """
    if not os.path.isdir(br_rewards_dir):
        return []
    entries = sorted(os.listdir(br_rewards_dir))
    runs = []
    for e in entries:
        full = os.path.join(br_rewards_dir, e)
        if os.path.isdir(full):
            runs.append((e, full))
    has_top_level_txt = any(
        e.endswith(".txt") and os.path.isfile(os.path.join(br_rewards_dir, e))
        for e in entries
    )
    if has_top_level_txt:
        runs.append(("", br_rewards_dir))
    return runs


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

    # Scatter BR points. Continue-mode = filled, dedicated-mode = hollow,
    # so the master plot can show both without expanding the color palette.
    # Legacy records (exp_type == "") render as continue (filled) since
    # pre-suffix runs were de-facto continue-mode.
    for rec in records:
        is_dedicated = (rec.get("exp_type") or "continue") == "dedicated"
        color = color_by_matchup[rec["matchup_key"]]
        ax.scatter(
            rec["timestep"], rec["value"],
            facecolors=("none" if is_dedicated else color),
            edgecolors=color,
            marker=marker_by_direction[rec["direction"]],
            s=55, alpha=0.9, linewidths=1.2,
        )

    # Connect paired BR points at same timestep/matchup (existing solid line).
    for (_, _), group in pairs_by_timestep_matchup.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda x: x["direction"])
        color = color_by_matchup[group_sorted[0]["matchup_key"]]
        xs = [g["timestep"] for g in group_sorted]
        ys = [g["value"] for g in group_sorted]
        ax.plot(xs, ys, color=color, alpha=0.35, linewidth=1.0)

    # Selfplay overlay: ONE averaged point per (timestep, matchup) — the two
    # main_left/main_right files for a matchup are two samples of the same
    # self-play game viewed from each side, so we average them. Square
    # hollow marker keeps it visually distinct from BR (circle/triangle).
    selfplay_means = _compute_selfplay_means(records)
    for (timestep, matchup_key), mean_value in selfplay_means.items():
        ax.scatter(
            timestep, mean_value,
            facecolors="none", edgecolors=color_by_matchup[matchup_key],
            marker="s", s=55, alpha=0.9, linewidths=1.2,
        )

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

    # Metric/exploiter-type legend group. Always include the continue vs
    # dedicated entries when any record carries an explicit exp_type, so
    # the filled/hollow convention is documented inline. Selfplay entry
    # only appears if any selfplay data was paired.
    has_explicit_exp_type = any(rec.get("exp_type") for rec in records)
    if has_explicit_exp_type:
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="black", linestyle="None",
                markersize=7, label="continue (filled)",
            )
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="black", linestyle="None",
                markersize=7, markerfacecolor="none", markeredgewidth=1.2,
                label="dedicated (hollow)",
            )
        )
    if any(rec.get("selfplay_value") is not None for rec in records):
        legend_handles.append(
            Line2D(
                [0], [0], marker="s", color="black", linestyle="None",
                markersize=7, markerfacecolor="none", markeredgewidth=1.2,
                label="selfplay (avg of directions, open square)",
            )
        )
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

    # Style prefix in plot filename so the artifact is self-describing
    # (e.g., league_master_local_eval_pairs.png). Empty style preserves
    # the legacy filename.
    style = _dominant_style(records)
    name_prefix = f"{style}_" if style else ""
    output_path = os.path.join(out_dir, f"{name_prefix}master_local_eval_pairs.png")
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
    # Continue and dedicated share the per-direction color but differ by
    # line style and marker fill — gives a consistent "left = blue, right
    # = orange" reading across the two exploiter types without needing a
    # 4-color palette.
    linestyle_by_exp = {"continue": "-", "dedicated": "--"}

    style = _dominant_style(records)
    name_prefix = f"{style}_" if style else ""

    for matchup_key, m_records in sorted(records_by_matchup.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        m_records = sorted(m_records, key=lambda x: (x["timestep"], x["direction"]))

        # Four BR series: (continue, dedicated) × (main_left, main_right).
        # Records arriving here have already been replicate-averaged in
        # _process_run, so each (timestep, direction, exp_type) bucket has
        # exactly one value. We plot a single line per series, marker
        # filled (continue) or hollow (dedicated). Legacy records without
        # an explicit exp_type are bucketed as "continue" by the
        # aggregator.
        for exp_type in ("continue", "dedicated"):
            for direction in ("main_left", "main_right"):
                d_recs = sorted(
                    (r for r in m_records
                     if r["direction"] == direction
                     and (r["exp_type"] or "continue") == exp_type),
                    key=lambda r: r["timestep"],
                )
                if not d_recs:
                    continue
                xs = [r["timestep"] for r in d_recs]
                ys = [r["value"] for r in d_recs]
                color = color_by_direction[direction]
                marker = marker_by_direction[direction]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyle_by_exp[exp_type],
                    linewidth=1.4,
                    marker=marker,
                    markersize=5,
                    markerfacecolor=("none" if exp_type == "dedicated" else color),
                    markeredgecolor=color,
                    alpha=0.95,
                    label=f"{exp_type} {direction}",
                )

        # Selfplay overlay: ONE averaged line per matchup. The two
        # direction-records at the same timestep are two samples of the
        # same selfplay matchup, so we average them. Drawn dashed in a
        # neutral color so it doesn't compete with the per-direction BR
        # colors. Skipped silently when no selfplay data exists.
        sp_means = _compute_selfplay_means(m_records)
        if sp_means:
            sp_sorted = sorted(sp_means.items(), key=lambda kv: kv[0][0])
            sp_xs = [k[0] for k, _ in sp_sorted]
            sp_ys = [v for _, v in sp_sorted]
            ax.plot(
                sp_xs,
                sp_ys,
                marker="s",
                color="#555555",
                linewidth=1.2,
                markersize=5,
                alpha=0.85,
                linestyle="--",
                markerfacecolor="none",
                label="selfplay (avg of directions)",
            )

        ax.set_title(f"Local BR Eval: {matchup_key}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        out_name = f"{name_prefix}matchup_{matchup_key}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    return paths


def _process_run(input_dir, output_dir, label, selfplay_dir=None):
    """
    Run the full parse + plot pipeline for a single training process.

    *input_dir* is the per-run rewards folder (subfolder of br_rewards/, or
    br_rewards/ itself in the legacy unsegregated case). *output_dir* is
    the matching plots destination. *label* is the run identifier used in
    log lines (subfolder name, or "<legacy>" for the top-level bucket).
    *selfplay_dir* (optional) is the matching selfplay rewards folder; if
    provided, each record gets a `selfplay_value` populated for overlay
    plotting.

    Returns a dict with summary stats; the caller is responsible for
    aggregate logging across multiple runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = _parse_records(input_dir)
    if not records:
        print(f"[{label}] No parseable reward files in: {input_dir}")
        return {"label": label, "count": 0}

    selfplay_paired = _attach_selfplay_values(records, selfplay_dir)
    raw_count = len(records)

    # Collapse replicates: multiple (br_idx) files for the same (timestep,
    # matchup, direction, exp_type) get averaged into one record. Plotters
    # downstream then see one point per (timestep, matchup, direction,
    # exp_type) instead of N separate replicates.
    records = _aggregate_replicates(records)

    pairs = _build_pairs(records)
    states, left_set, right_set, matchups, timesteps = _infer_sets(records)

    print(f"=== Run: {label} ===")
    print(f"  Rewards folder: {input_dir}")
    print(f"  Output folder:  {output_dir}")
    if selfplay_dir:
        print(f"  Selfplay folder: {selfplay_dir} (paired={selfplay_paired}/{raw_count})")
    print(f"  Parsed file count: {raw_count} (after replicate-mean aggregation: {len(records)})")
    print(f"  Timestep count: {len(timesteps)}")
    print(f"  Matchup count: {len(matchups)}")
    print(f"  left_set (ego_list candidate): {left_set}")
    print(f"  right_set (adv_list candidate): {right_set}")
    print("  Inferred state_list:")
    for s in states:
        print(f"    - {s}")

    complete_pairs = 0
    incomplete_pairs = 0
    for _, group in pairs.items():
        if len(group) >= 2:
            complete_pairs += 1
        else:
            incomplete_pairs += 1
    print("  Pairing summary (timestep + matchup groups):")
    print(f"    complete pairs: {complete_pairs}")
    print(f"    incomplete pairs (kept): {incomplete_pairs}")

    master_path = _plot_master(records, pairs, output_dir)
    matchup_plot_paths = _plot_per_matchup(records, output_dir)

    print("  Saved plots:")
    print(f"    - {master_path}")
    for p in matchup_plot_paths:
        print(f"    - {p}")

    return {
        "label": label,
        "count": len(records),
        "matchups": len(matchups),
        "complete_pairs": complete_pairs,
        "incomplete_pairs": incomplete_pairs,
        "selfplay_paired": selfplay_paired,
        "master_path": master_path,
        "matchup_plot_paths": matchup_plot_paths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--br_rewards_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "br_rewards"),
        help="Folder containing local br reward txt files (segregated by "
             "training-process subfolder, or legacy flat layout).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_eval_plots"),
        help="Folder for saved plots. Per-run plots land in subfolders "
             "matching the br_rewards/ subfolder layout.",
    )
    parser.add_argument(
        "--training_process",
        type=str,
        default="",
        help="Optional filter: process only this single subfolder name "
             "(useful for incremental re-plot). Empty = process all.",
    )
    parser.add_argument(
        "--selfplay_rewards_dir",
        type=str,
        default="",
        help="Folder containing local selfplay reward txt files. Subfolder "
             "layout mirrors --br_rewards_dir. Defaults to the sibling of "
             "--br_rewards_dir with 'br_rewards' replaced by "
             "'selfplay_rewards'.",
    )
    args = parser.parse_args()

    # Default selfplay dir = sibling of br_rewards_dir. local_br_eval.py
    # writes to "selfplay_rewards" in the same parent as "br_rewards", so
    # this string substitution is the natural pairing — and only fires
    # when the user hasn't overridden via CLI.
    if not args.selfplay_rewards_dir:
        if "br_rewards" in args.br_rewards_dir:
            args.selfplay_rewards_dir = args.br_rewards_dir.replace(
                "br_rewards", "selfplay_rewards"
            )
        else:
            args.selfplay_rewards_dir = ""  # unable to derive; skip overlay

    os.makedirs(args.output_dir, exist_ok=True)
    runs = _discover_training_processes(args.br_rewards_dir)
    if not runs:
        print(f"No training-process data found under: {args.br_rewards_dir}")
        return

    if args.training_process:
        runs = [(name, path) for name, path in runs if name == args.training_process]
        if not runs:
            print(
                f"--training_process={args.training_process!r} did not match "
                f"any subfolder under {args.br_rewards_dir}"
            )
            return

    print(f"Discovered {len(runs)} training-process bucket(s) under {args.br_rewards_dir}")
    if args.selfplay_rewards_dir:
        print(f"Selfplay overlay source: {args.selfplay_rewards_dir}")
    summaries = []
    for name, path in runs:
        # Empty subfolder name == legacy bucket: write plots straight into
        # the top-level output_dir to mirror old behavior. Selfplay folder
        # mirrors the same layout (subfolder under selfplay_rewards/, or
        # selfplay_rewards/ itself for the legacy bucket).
        if name:
            sub_out = os.path.join(args.output_dir, name)
            label = name
            sp_dir = (
                os.path.join(args.selfplay_rewards_dir, name)
                if args.selfplay_rewards_dir else None
            )
        else:
            sub_out = args.output_dir
            label = "<legacy>"
            sp_dir = args.selfplay_rewards_dir or None
        summaries.append(_process_run(path, sub_out, label, selfplay_dir=sp_dir))

    print("=== Aggregate summary ===")
    for s in summaries:
        line = f"  {s['label']}: parsed={s['count']}"
        if s["count"]:
            line += f", matchups={s.get('matchups', 0)}"
            line += f", selfplay_paired={s.get('selfplay_paired', 0)}"
        print(line)


if __name__ == "__main__":
    main()
