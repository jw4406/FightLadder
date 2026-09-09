#!/bin/bash
# ---------------------------------------------------------------------------
# Merge the per-workspace BR eval data into ONE flat dir per family so the
# aggregate plotter sees the whole exploitability curve.
#
# WHY: each BR workspace ($WORKDIR/$MAIN_TRAINING_DIR) writes its own
# br_rewards/<output_subdir>/*.txt, but aggregate_local_eval_data.py scans a
# SINGLE flat --br_rewards_dir. Splitting the mains/curve across workspaces
# (needed for matchup-sibling isolation) fragments the data. The eval
# filenames encode {num_timesteps}_main_{side}_{main_name}_exploiter_... and
# are globally unique, so a symlink merge reconstructs the full dataset with
# no relaunch and no data copy.
#
# Usage:
#   merge_br_rewards_for_plot.sh <family_prefix> <out_dir>
# e.g.
#   merge_br_rewards_for_plot.sh br_psro   /n/fs/magics/plots/psro_merged
#   merge_br_rewards_for_plot.sh br_league /n/fs/magics/plots/league_merged
# then:
#   python main/aggregate_local_eval_data.py --br_rewards_dir <out_dir>/br_rewards
# ---------------------------------------------------------------------------
set -u
PREFIX="${1:?usage: merge_br_rewards_for_plot.sh <family_prefix e.g. br_psro> <out_dir>}"
OUT="${2:?usage: merge_br_rewards_for_plot.sh <family_prefix> <out_dir>}"
MAGICS=/n/fs/magics

BR_OUT="$OUT/br_rewards"; SP_OUT="$OUT/selfplay_rewards"
mkdir -p "$BR_OUT" "$SP_OUT"

nbr=0; nsp=0; nws=0
for ws in "$MAGICS/${PREFIX}"*/; do
    base="$ws/FightLadder/main"
    [ -d "$base/br_rewards" ] || continue
    nws=$((nws+1))
    # flat-merge .txt ONLY from historical_step_*/ (the reeval'd curve). The
    # filename carries the main step so filenames are unique across checkpoints.
    # Excludes todo/ (stale collided originals, all mislabeled step 0) and loose
    # top-level files (unrelated older BR data). See _auto_merge_family in
    # aggregate_local_eval_data.py for the rationale.
    while IFS= read -r f; do ln -sf "$f" "$BR_OUT/$(basename "$f")" && nbr=$((nbr+1)); done \
        < <(find "$base/br_rewards"/historical_step_* -type f -name '*.txt' 2>/dev/null)
    while IFS= read -r f; do ln -sf "$f" "$SP_OUT/$(basename "$f")" && nsp=$((nsp+1)); done \
        < <(find "$base/selfplay_rewards"/historical_step_* -type f -name '*.txt' 2>/dev/null)
done

echo "merged from $nws workspace(s) matching ${PREFIX}*:"
echo "  br_rewards:       $nbr files -> $BR_OUT"
echo "  selfplay_rewards: $nsp files -> $SP_OUT"
echo
echo "now plot the full curve with:"
echo "  python main/aggregate_local_eval_data.py --br_rewards_dir $BR_OUT"
