#!/bin/bash
# EXPLOITABILITY GAP on the masked-RAM arm -- the ADVERSARY-seat half.
#
# THE METRIC (this is what gets reported from now on):
#
#     A = LBR return at the EGO seat   -> how exploitable the ADVERSARY is
#     B = LBR return at the ADV seat   -> how exploitable the EGO is
#     GAP = B - A
#
# Deliberately NOT eps. eps = LBR_return - selfplay_return, and the selfplay
# baseline is untrustworthy here: at 10.08M the whole rise in eps_ego (0.232 ->
# 0.277) came from selfplay falling (-0.057 -> -0.110) while the exploiter's
# absolute return was FLAT (+0.175 -> +0.167). Subtracting a moving, degenerate
# baseline manufactures a trend that is not in the exploiter data.
#
# GAP uses two quantities measured the SAME way, with no baseline at all:
#   GAP -> 0    neither player is exploited more than the other
#   GAP > 0     the EGO is the more exploitable one
#   GAP < 0     the ADVERSARY is
#
# The ego seat is already measured for these checkpoints (run_eps_series.sh /
# logs/lbr_masked_ego.log); this fills in the adversary seat so the pair can be
# differenced. --eval_prot True == LBR occupies the ADVERSARY seat.
#
# NOTE ON SIGN: run_lbr reads the reward slot for whichever seat LBR holds
# (lbr_reward, not -r_other) because a draw pays BOTH players +1, so the game is
# not exactly zero-sum and negating the other side's reward would be wrong.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "FATAL: no torch"; exit 3; }

ARM="${ARM:-minimax_phase0_vtoff_rammasked}"
CKDIR="${SCRIPT_DIR}/main/${ARM}/trained_models/tasks/todo"
MASK="${MASK:-${SCRIPT_DIR}/main/ram_mask.npy}"
STEPS="${STEPS:-2400000 10080000 24960000 48000000}"
MODES="${MODES:-greedy,lbr,shuffle}"
# Episodes finish in parallel across envs, so decisions-to-50-episodes scales as
# 1/n_envs while per-decision cost stays roughly flat as long as cores are free.
# With the trainer stopped there are 32 cores against 13 envs, so widening this
# is the cheapest available speedup.
N_ENVS="${N_ENVS:-16}"
OUT="${SCRIPT_DIR}/logs/gap_series.log"
: > "${OUT}"

# NOT serialized any more. The trainer is stopped, freeing ~24 of 32 cores, so
# this runs ALONGSIDE the ego-seat series rather than after it.

FAIL=0
for S in ${STEPS}; do
    CK="${CKDIR}/spar_Ry_Sa_${S}_steps.task"
    [ -f "${CK}" ] || { echo "MISSING ${CK}" | tee -a "${OUT}"; FAIL=1; continue; }
    echo "=== ADV seat ${S} $(date '+%F %H:%M:%S') ===" | tee -a "${OUT}"
    python -u "${SCRIPT_DIR}/main/local_best_response.py" \
        --main_checkpoint_model_path "${CK}" --ram_mask "${MASK}" \
        --lbr_modes "${MODES}" --eval_prot True \
        --lbr_episodes 50 --lbr_n_envs "${N_ENVS}" \
        --output_subdir "spar_masked_gap_${S}" \
        --training_style spar --br_index 0 2>&1 \
      | grep -E "^\[LBR\]    |direction" | tee -a "${OUT}"
    # PIPESTATUS, not $?: $? is grep's, so a crashed run that printed nothing
    # would look like a missing point rather than a failure.
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "  FAILED at ${S}" | tee -a "${OUT}"; FAIL=1; }
done

echo "" | tee -a "${OUT}"
echo "GAP_SERIES_DONE fail=${FAIL}" | tee -a "${OUT}"
