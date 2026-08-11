#!/bin/bash
# EXPLOITABILITY TREND on the masked-RAM arm: does eps_ego FALL as training
# proceeds, even while the ego loses more rounds?
#
# THE QUESTION THIS SETTLES. score_rollout fell 0.345 -> ~0.06 and rating_gap
# went -6.7 -> -446, but BOTH ARE RELATIVE: the config gives the adversary a
# 3.3x higher learning rate (--d_lr 1e-4 vs --c_lr 3e-5), so a dominant
# adversary is the intended direction, and ELO is self-referential -- the two
# ratings update from the same games. Both players improving with the adversary
# improving faster produces exactly the observed picture.
#
# Precedent that relative metrics mislead HERE: on the image arm eps_ego fell
# 0.384 -> 0.212 -> 0.138 -- the ego genuinely becoming LESS exploitable --
# while the relative metrics read as collapse.
#
# eps is the ABSOLUTE quantity and distinguishes the two stories:
#   eps_ego FALLING   the adversary is doing its job; the ego is getting harder
#                     to exploit even as it wins fewer rounds
#   eps_ego FLAT/UP   genuine degeneration -- losing AND not improving
#
# All three modes, matching the 2.4M point so the series is comparable:
#   greedy   gamma=0, no critic. The tightest exploiter measured so far, so the
#            best-powered trend line.
#   lbr      V-based. Says whether the VALUE FUNCTION is contributing.
#   shuffle  branch ordering ablated (branch-axis bug fixed 2026-08-10).
#
# CAVEAT that travels with every point: the gate (score in [0.3,0.7]) fails at
# all of these checkpoints. It exists because eps measured in a lopsided game is
# less reliable -- selfplay return is low, so LBR has room to look good for
# reasons other than the ego being exploitable. Read the TREND, not the level.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "FATAL: no torch"; exit 3; }

ARM="${ARM:-minimax_phase0_vtoff_rammasked}"
CKDIR="${SCRIPT_DIR}/main/${ARM}/trained_models/tasks/todo"
MASK="${MASK:-${SCRIPT_DIR}/main/ram_mask.npy}"
STEPS="${STEPS:-10080000 24960000 48000000}"
MODES="${MODES:-greedy,lbr,shuffle}"
OUT="${SCRIPT_DIR}/logs/eps_series.log"
: > "${OUT}"

# Wait for any in-flight LBR so the runs do not contend for CPU; each one is
# ~65 min and they are CPU-bound in the emulator.
for i in $(seq 1 120); do
    pgrep -f "[l]ocal_best_response.py" >/dev/null || break
    sleep 60
done

FAIL=0
for S in ${STEPS}; do
    CK="${CKDIR}/spar_Ry_Sa_${S}_steps.task"
    [ -f "${CK}" ] || { echo "MISSING ${CK}" | tee -a "${OUT}"; FAIL=1; continue; }
    echo "=== ${S} $(date '+%F %H:%M:%S') ===" | tee -a "${OUT}"
    python -u "${SCRIPT_DIR}/main/local_best_response.py" \
        --main_checkpoint_model_path "${CK}" --ram_mask "${MASK}" \
        --lbr_modes "${MODES}" --eval_prot False \
        --lbr_episodes 50 --lbr_n_envs 13 \
        --output_subdir "spar_masked_eps_${S}" \
        --training_style spar --br_index 0 2>&1 \
      | grep -E "^\[LBR\]    |eps_|NashConv|PARTIAL" | tee -a "${OUT}"
    # PIPESTATUS, not $?: $? is grep's. A crashed run that printed nothing would
    # otherwise leave a gap that reads as a missing point rather than a failure.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "  FAILED at ${S}" | tee -a "${OUT}"; FAIL=1
    fi
done

echo "" | tee -a "${OUT}"
echo "EPS_SERIES_DONE fail=${FAIL}" | tee -a "${OUT}"
