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
# exp_type ∈ {lbr, lbrgreedy, lbrshuffle} are written by local_best_response.py:
# a Local Best Response lower bound and its two controls. These are NOT trained
# best responses -- they cost ~100x less and are expected to be looser.
FILENAME_RE = re.compile(
    r"^(?:(?P<style>[A-Za-z0-9]+)_)?"
    r"(?P<timestep>\d+)_main_(?P<main_side>left|right)_(?P<main_char>[A-Za-z0-9]+)"
    r"_exploiter_(?P<exploiter_side>left|right)_(?P<exploiter_char>[A-Za-z0-9]+)"
    # Longest-first: `lbr` prefixes `lbrgreedy`/`lbrshuffle`, and while backtracking
    # happens to resolve that correctly, ordering it explicitly keeps it obvious.
    r"(?:_(?P<exp_type>continue|dedicated|lbrminimaxshuffle|lbrminimax|lbrgreedy|lbrshuffle|lbr)_br(?P<br_idx>\d+))?"
    # Optional periodic-snapshot suffix written by
    # PeriodicLocalBREvalCallback in new_br_worker.py while a BR run is
    # still in flight. Format: "_brstep<N>_<YYYYMMDDTHHMMSS>". When
    # absent, the file is the canonical post-learn() final-eval output.
    # The selector below prefers canonical files; periodic snapshots are
    # only used as a fallback when no canonical exists for a bucket
    # (i.e. the BR run never reached its final eval — crashed / still
    # running). For periodic fallbacks, the latest snapshot (highest
    # br_step) wins.
    r"(?:_brstep(?P<br_step>\d+)_(?P<periodic_ts>\d{8}T\d{6}))?"
    # Tolerate a stray trailing suffix token (e.g. "_unknown") emitted by
    # some in-flight worker builds as a --filename_suffix. It carries no
    # aggregation meaning; capturing it here keeps such files parseable
    # instead of silently dropping the whole bucket ("No parseable reward
    # files"). Canonical (no suffix) and periodic (_brstep..) both still match.
    r"(?:_(?P<misc_suffix>[A-Za-z0-9]+))?"
    r"_\.txt$"
)

# Two families of measurement share this folder and these axes, in the same
# units (mean episode return of the exploiting side) -- but they are NOT the
# same quantity. A trained BR is an actual best response after 10M-150M env
# steps; LBR is a one-step-lookahead LOWER BOUND costing ~100x less, and is
# expected to sit well below it. Plotted undifferentiated, an LBR series reads
# as "the BR got much worse", so the two families are kept on separate master
# axes and labelled distinctly.
TRAINED_BR_EXP_TYPES = frozenset({"", "continue", "dedicated"})
LBR_EXP_TYPES = frozenset({"lbr", "lbrgreedy", "lbrshuffle",
                           "lbrminimax", "lbrminimaxshuffle"})

# Headline vs control, within the LBR family. `lbr` is the measurement;
# `lbrgreedy` (gamma=0, critic unused) and `lbrshuffle` (critic shuffled across
# branches) are controls -- if either matches `lbr`, the lookahead machinery is
# contributing nothing and that is the finding.
LBR_HEADLINE_EXP_TYPE = "lbr"

_UNKNOWN_EXP_TYPES_WARNED = set()


def _family(exp_type):
    """
    Classify an exp_type into "trained_br" or "lbr".

    Legacy files carry no exp_type ("") and are treated as trained_br, matching
    the pre-existing "legacy == de-facto continue" convention. Unknown types
    fall back to trained_br (the historical behaviour) with a one-time warning,
    so an unrecognized suffix degrades to the old rendering rather than
    vanishing from the plots.
    """
    et = exp_type or ""
    if et in TRAINED_BR_EXP_TYPES:
        return "trained_br"
    # Prefix fallback so future lbr* variants need no edit here.
    if et in LBR_EXP_TYPES or et.startswith("lbr"):
        return "lbr"
    if et not in _UNKNOWN_EXP_TYPES_WARNED:
        _UNKNOWN_EXP_TYPES_WARNED.add(et)
        print(f"  [warn] unknown exp_type {et!r}; plotting it as a trained BR")
    return "trained_br"


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
    # Keep physical left/right order from _matchup_from_record. Both
    # directional samples of the same matchup (main_left + main_right)
    # already resolve to the same (left_char, right_char) pair, so no
    # alphabetical sort is needed to canonicalize. Preserving the on-screen
    # order means main-training conventions like "Vega_left" surface as
    # "Vega_vs_<adv>" in plot titles and file names, matching the
    # "ego vs adversary" mental model.
    return f"{left_char}_vs_{right_char}"


def _parse_records(br_rewards_dir, ego_centric=True):
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
        # Which measurement family this record belongs to. Attached once here
        # so every downstream consumer filters on it rather than re-deriving
        # the exp_type -> family mapping.
        rec["family"] = _family(rec["exp_type"])
        # EGO-CENTRIC SIGN CONVENTION (LBR family only).
        #
        # local_best_response.py writes each seat's OWN return, so a main_left
        # record (LBR in the ADV seat) is in adversary units and a main_right
        # record is in ego units. Verified: the two directions' selfplay values
        # are exact negations in 20/21 LBR pairs.
        #
        # Negating main_left puts everything in EGO units, so one axis reads
        # consistently: eps_ego above the selfplay line, eps_adv below it. It
        # also repairs the selfplay overlay as a side effect -- previously
        # _compute_selfplay_means averaged x and -x and got a structurally ZERO
        # line that looked like a reference and carried no information. After
        # negation both directions agree, so the average is the actual value.
        #
        # NOT applied to trained_br: those folders are NOT negations
        # (max|left+right| = 0.36 on spar_Gu_SaRyEH, 0 of 12 pairs negate) --
        # that family records the LEFT player's return regardless of seat, so
        # flipping it would corrupt plots that are currently correct.
        if ego_centric and rec["family"] == "lbr" and rec["direction"] == "main_left":
            rec["value"] = -rec["value"]
            rec["_negated"] = True
        br_idx_str = rec.get("br_idx")
        rec["br_idx"] = int(br_idx_str) if br_idx_str is not None else None
        # Periodic-snapshot fields. br_step is the BR-side env-step count
        # at which the snapshot was taken; periodic_ts is its wall-clock
        # timestamp. Both None for canonical (final-eval) files.
        br_step_str = rec.get("br_step")
        rec["br_step"] = int(br_step_str) if br_step_str is not None else None
        rec["periodic_ts"] = rec.get("periodic_ts") or None
        # Selfplay value populated later by _attach_selfplay_values once
        # the matching sibling folder is known.
        rec["selfplay_value"] = None
        records.append(rec)
    return records


def _select_canonical_or_latest_periodic(records):
    """
    Collapse periodic-snapshot duplicates by picking, for each
    (timestep, matchup_key, direction, exp_type, br_idx) bucket, the
    single most authoritative record. Priority:

      1. Canonical (br_step is None) — the post-learn() final eval —
         wins over any periodic snapshot. This is what the canonical
         pipeline emits when a BR run completes normally.
      2. If no canonical exists, the periodic snapshot with the highest
         br_step wins — i.e. the latest mid-training eval, closest to
         where the run actually got. This is the crash-recovery path:
         the BR run never reached its final eval, so we use the freshest
         snapshot we have on disk.

    Returns: list with one record per bucket. All downstream stages
    (replicate averaging, selfplay attach, plotting) then treat the
    selected records uniformly — they no longer have to know whether a
    given value came from a canonical eval or a periodic surrogate.
    """
    grouped = defaultdict(list)
    for rec in records:
        key = (
            rec["timestep"],
            rec["matchup_key"],
            rec["direction"],
            rec["exp_type"] or "continue",
            rec["br_idx"],
        )
        grouped[key].append(rec)

    selected = []
    n_canonical = 0
    n_periodic_fallback = 0
    n_periodic_dropped = 0
    for _, group in grouped.items():
        canonical = [r for r in group if r["br_step"] is None]
        if canonical:
            # Multiple canonical files for the same bucket shouldn't
            # happen (same filename written twice) — pick the first.
            selected.append(canonical[0])
            n_canonical += 1
            n_periodic_dropped += len(group) - len(canonical)
        else:
            latest = max(group, key=lambda r: r["br_step"])
            selected.append(latest)
            n_periodic_fallback += 1
            n_periodic_dropped += len(group) - 1
    if n_periodic_fallback or n_periodic_dropped:
        print(
            f"  [info] canonical={n_canonical} periodic-as-fallback="
            f"{n_periodic_fallback} periodic-dropped={n_periodic_dropped}"
        )
    return selected


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

    One exception: local_best_response.py writes selfplay_rewards/ ONLY under
    the `_lbr_` stem, not under `_lbrgreedy_` / `_lbrshuffle_`. Since the
    selfplay marker IS the visual gap on these plots, a control without it
    renders as an uncalibrated curve. The three variants share one checkpoint,
    matchup and direction -- the self-play baseline is identical for all of
    them -- so a control falls back to its sibling `_lbr_` file.
    """
    if not selfplay_dir or not os.path.isdir(selfplay_dir):
        return 0
    paired = 0
    for rec in records:
        sp_path = os.path.join(selfplay_dir, rec["filename"])
        if not os.path.isfile(sp_path):
            et = rec.get("exp_type") or ""
            if rec.get("family") != "lbr" or et == LBR_HEADLINE_EXP_TYPE:
                continue
            sibling = rec["filename"].replace(
                f"_{et}_br", f"_{LBR_HEADLINE_EXP_TYPE}_br", 1
            )
            sp_path = os.path.join(selfplay_dir, sibling)
            if not os.path.isfile(sp_path):
                continue
        sp_value = _safe_float_from_file(sp_path)
        if sp_value is None:
            continue
        # Match the record's own sign convention -- if the value was negated
        # into ego units, its selfplay reference must be too, or the overlay
        # sits on the wrong side of the curve.
        rec["selfplay_value"] = -sp_value if rec.get("_negated") else sp_value
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


def _plot_master(records, pairs_by_timestep_matchup, out_dir, family="trained_br",
                 ego_centric=True):
    """
    Master scatter for ONE measurement family. *records* must already be
    filtered to that family by the caller -- the two families are not
    commensurable enough to share an axis (see the TRAINED_BR/LBR note above).

    family="trained_br" reproduces the historical figure byte-for-byte,
    including its filename; family="lbr" emits the parallel LBR figure.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    is_lbr = family == "lbr"

    matchup_keys = sorted({rec["matchup_key"] for rec in records})
    cmap = matplotlib.colormaps.get_cmap("tab20")
    denom = max(1, len(matchup_keys) - 1)
    color_by_matchup = {k: cmap(i / denom) for i, k in enumerate(matchup_keys)}
    marker_by_direction = {"main_left": "o", "main_right": "^"}

    # Scatter BR points. Continue-mode = filled, dedicated-mode = hollow,
    # so the master plot can show both without expanding the color palette.
    # Legacy records (exp_type == "") render as continue (filled) since
    # pre-suffix runs were de-facto continue-mode.
    #
    # Same filled/hollow budget is reused for the LBR family: headline `lbr`
    # filled, its controls hollow. Greedy and shuffle are deliberately not
    # separated here -- a control landing ON the headline is precisely the
    # signal worth seeing, and the per-matchup figures tell them apart.
    for rec in records:
        if is_lbr:
            hollow = (rec.get("exp_type") or "") != LBR_HEADLINE_EXP_TYPE
        else:
            hollow = (rec.get("exp_type") or "continue") == "dedicated"
        color = color_by_matchup[rec["matchup_key"]]
        ax.scatter(
            rec["timestep"], rec["value"],
            facecolors=("none" if hollow else color),
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

    if is_lbr:
        ax.set_title(
            "Local Best Response (lower bound): All Matchups (paired directions)"
        )
        ax.set_ylabel("Reward (LBR lower bound, EGO-centric)"
                      if ego_centric else "Reward (LBR lower bound, own-units)")
    else:
        ax.set_title("Local BR Eval: All Matchups (paired directions)")
        ax.set_ylabel("Reward")
    ax.set_xlabel("Timestep")
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
    if is_lbr:
        filled_label = "lbr (filled)"
        hollow_label = "greedy / shuffle controls (hollow)"
    else:
        filled_label = "continue (filled)"
        hollow_label = "dedicated (hollow)"
    if has_explicit_exp_type:
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="black", linestyle="None",
                markersize=7, label=filled_label,
            )
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="black", linestyle="None",
                markersize=7, markerfacecolor="none", markeredgewidth=1.2,
                label=hollow_label,
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
    stem = "master_lbr_pairs" if is_lbr else "master_local_eval_pairs"
    output_path = os.path.join(out_dir, f"{name_prefix}{stem}.png")
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_matchup_for_exp_type(m_records, matchup_key, exp_type, out_path,
                               ego_centric=True):
    """
    Render ONE per-matchup figure scoped to a single BR exp_type
    ("continue" or "dedicated"), with the matchup's selfplay average
    drawn as an overlay so the BR curves are comparable to the
    self-play baseline in either plot.

    Records arriving here have already been replicate-averaged in
    _process_run, so each (timestep, direction) bucket has exactly one
    BR value for this exp_type.
    """
    marker_by_direction = {"main_left": "o", "main_right": "^"}
    color_by_direction = {"main_left": "#1f77b4", "main_right": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(10, 6))
    m_records = sorted(m_records, key=lambda x: (x["timestep"], x["direction"]))

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
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle="-",
            linewidth=1.4,
            marker=marker_by_direction[direction],
            markersize=5,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.95,
            label=(("eps_adv (ego exploited)" if direction == "main_left"
                    else "eps_ego (adv exploited)") if ego_centric
                   else f"{exp_type} {direction}"),
        )

    # Selfplay overlay: ONE averaged line per matchup. The two
    # direction-records at the same timestep are two samples of the same
    # selfplay matchup, so we average them. Same overlay is drawn in
    # both the continue plot and the dedicated plot so each can be read
    # against the self-play baseline on its own.
    # Restrict the baseline to this figure's own family. The self-play value is
    # nominally the same quantity for both, but the two families are evaluated
    # at different checkpoint cadences -- pooling them would draw baseline
    # points at timesteps where this figure has no data, and would perturb the
    # historical trained-BR figures in any folder that also contains LBR runs.
    fam = _family(exp_type)
    sp_means = _compute_selfplay_means(
        [r for r in m_records if _family(r["exp_type"]) == fam]
    )
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

    # Family-appropriate framing. A trained-BR figure keeps its historical
    # title/ylabel exactly; an LBR figure says what it actually is, so a
    # curve sitting far below the trained-BR series is not misread as
    # regression. The caveat line is the point: LBR bounds exploitability
    # from below and its controls are diagnostics, not measurements.
    if _family(exp_type) == "lbr":
        caveat = ("one-step lookahead — a LOWER BOUND on exploitability, "
                  "not a trained best response")
        if exp_type == "lbrgreedy":
            ax.set_title(f"LBR lower bound (greedy control): {matchup_key}")
            caveat = "CONTROL: γ=0, critic unused — matching lbr means the lookahead adds nothing"
        elif exp_type == "lbrshuffle":
            ax.set_title(f"LBR lower bound (shuffle control): {matchup_key}")
            caveat = "CONTROL: critic shuffled across branches — matching lbr means V does not discriminate"
        else:
            ax.set_title(f"LBR lower bound: {matchup_key}")
        ax.set_ylabel("Reward (LBR lower bound, EGO-centric)"
                      if ego_centric else "Reward (LBR lower bound, own-units)")
        ax.text(0.5, -0.13, caveat, transform=ax.transAxes, ha="center",
                va="top", fontsize=7, color="#555555")
    else:
        ax.set_title(f"Local BR Eval ({exp_type}): {matchup_key}")
        ax.set_ylabel("Reward")
    ax.set_xlabel("Timestep")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_per_matchup(records, out_dir, ego_centric=True):
    """
    For every (matchup, exp_type) combination present in *records*, emit
    one PNG. So a matchup that has both continue and dedicated runs
    produces two files; one with only one exp_type produces just one.
    Selfplay overlay is drawn in every plot so each BR curve set can be
    read against the same self-play baseline.
    """
    paths = []
    records_by_matchup = defaultdict(list)
    for rec in records:
        records_by_matchup[rec["matchup_key"]].append(rec)

    style = _dominant_style(records)
    name_prefix = f"{style}_" if style else ""

    for matchup_key, m_records in sorted(records_by_matchup.items()):
        # Derive the exp_type set from the data rather than hardcoding it, so new
        # variants (lbr, lbrgreedy, lbrshuffle, ...) need no further edits here.
        exp_types = sorted({(r["exp_type"] or "continue") for r in m_records})
        for exp_type in exp_types:
            has_records = any(
                (r["exp_type"] or "continue") == exp_type for r in m_records
            )
            if not has_records:
                continue
            out_name = f"{name_prefix}matchup_{matchup_key}_{exp_type}.png"
            out_path = os.path.join(out_dir, out_name)
            _plot_matchup_for_exp_type(m_records, matchup_key, exp_type, out_path,
                                      ego_centric=ego_centric)
            paths.append(out_path)

    return paths


def _process_run(input_dir, output_dir, label, selfplay_dir=None,
                 ego_centric=True):
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
    records = _parse_records(input_dir, ego_centric=ego_centric)
    if not records:
        print(f"[{label}] No parseable reward files in: {input_dir}")
        return {"label": label, "count": 0}

    selfplay_paired = _attach_selfplay_values(records, selfplay_dir)
    raw_count = len(records)

    # Step 1: per-bucket selector. For each (timestep, matchup, direction,
    # exp_type, br_idx), keep the canonical (post-learn() final) record
    # if it exists; otherwise fall back to the latest periodic snapshot.
    # Without this, periodic-snapshot files (written every N env-steps by
    # PeriodicLocalBREvalCallback while the BR run is mid-flight) would
    # never plot — when no final-eval file exists yet, this pass surfaces
    # the freshest mid-training value so plots aren't empty.
    records = _select_canonical_or_latest_periodic(records)

    # Step 2: replicate averaging. Multiple (br_idx) records for the same
    # (timestep, matchup, direction, exp_type) get averaged into one
    # record. Plotters downstream then see one point per (timestep,
    # matchup, direction, exp_type) instead of N separate replicates.
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

    # One master per family, on its own axes. Pairs are rebuilt per family so
    # the connecting line never joins a trained-BR point to an LBR one at the
    # same (timestep, matchup). A family with no records emits no figure --
    # that is what keeps a trained-BR-only folder byte-identical to before,
    # and keeps an LBR-only folder from producing a blank legacy master.
    by_family = defaultdict(list)
    for rec in records:
        by_family[rec.get("family") or "trained_br"].append(rec)
    if by_family.get("lbr"):
        print(f"  Families: trained_br={len(by_family.get('trained_br', []))} "
              f"lbr={len(by_family['lbr'])} (plotted on separate axes)")

    master_path = None
    lbr_master_path = None
    for fam, target in (("trained_br", "master_path"), ("lbr", "lbr_master_path")):
        fam_records = by_family.get(fam)
        if not fam_records:
            continue
        path = _plot_master(
            fam_records, _build_pairs(fam_records), output_dir, family=fam,
            ego_centric=ego_centric
        )
        if target == "master_path":
            master_path = path
        else:
            lbr_master_path = path

    matchup_plot_paths = _plot_per_matchup(records, output_dir,
                                          ego_centric=ego_centric)

    print("  Saved plots:")
    for p in (master_path, lbr_master_path):
        if p:
            print(f"    - {p}")
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
        "lbr_master_path": lbr_master_path,
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
    parser.add_argument(
        "--sign_convention",
        type=str,
        choices=["ego", "own"],
        default="ego",
        help="LBR family only. 'ego' negates main_left records so every curve "
             "is in EGO units: eps_ego plots above the selfplay line, eps_adv "
             "below it, and the selfplay overlay stops being structurally zero. "
             "'own' keeps each seat's own return (how the .txt/.json are "
             "stored) and reproduces the pre-2026-08-05 plots. trained_br is "
             "never touched -- that family is not stored as negations.",
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
        summaries.append(_process_run(path, sub_out, label, selfplay_dir=sp_dir,
                                      ego_centric=(args.sign_convention == 'ego')))

    print("=== Aggregate summary ===")
    for s in summaries:
        line = f"  {s['label']}: parsed={s['count']}"
        if s["count"]:
            line += f", matchups={s.get('matchups', 0)}"
            line += f", selfplay_paired={s.get('selfplay_paired', 0)}"
        print(line)


if __name__ == "__main__":
    main()
