#!/bin/bash
# THE GATE: minimax vs minimaxshuffle vs greedy, on the vton Phase 0 arm.
#
#   minimax vs greedy          is Q a useful leaf evaluator where V is not?
#                              (V-based lbr loses to greedy 3/3: 0.41/0.17/0.25)
#   minimax vs minimaxshuffle  does Q carry BRANCH-level information at all?
#                              shuffle permutes Q across the 22 branches, so it
#                              keeps the value SCALE and destroys only the
#                              action ORDERING. minimax ~= minimaxshuffle means
#                              the ordering carried nothing.
#
# CHECKPOINTS satisfy BOTH gate conditions (score_rollout in [0.3,0.7] AND
# ep_len_mean < 400), verified against logs/minimax_phase0_vton.log. Balance
# alone is insufficient: at 472 steps/ep a ridge on the frozen latent scored
# +0.9636 state-only vs +0.9632 with a full per-cell slope, because the TIMER
# decides the outcome and the action cannot matter.
#
# The head is learning on this arm -- minimax_ev +0.249, target_corr +0.414
# (positive, so the ego/adversary frame flip that produced slope -0.990 in the
# first run is fixed). A gate on a head with ev<=0 would measure nothing.
#
# PRIOR: the ridge probe says this comes back null. 13 measurements from 480k to
# 12.48M, policy-sampled AND uniform-action, put the action gain within +-0.01
# of zero against a +0.02 threshold. This run is the BEHAVIORAL check of that
# representational claim -- they can disagree, and the probe is a linear read of
# a frozen latent while LBR uses the head end-to-end to CHOOSE.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# ARM selects which Phase 0 run to gate. Parameterized rather than hardcoded so
# the vtrace-off arms can be gated with the same script and the same STEP_LIST
# mechanism -- and so SWEEP_TAG/OUTPUT_SUBDIR carry the arm, which is what keeps
# one arm's results from overwriting another's.
ARM="${ARM:-vton}"
CKDIR="${SCRIPT_DIR}/main/minimax_phase0_${ARM}/trained_models/tasks/todo"
[ -d "${CKDIR}" ] || { echo "FATAL: no such arm: ${CKDIR}" >&2; exit 1; }

# Spread across the run, every one gate-qualified. 12.48M is included because it
# is where the uniform-action probe was measured, so the two disagree or agree
# on the SAME checkpoint rather than across different ones.
STEP_LIST="${STEP_LIST:-4800000 6720000 8640000 12480000}"
CKPTS=""
for S in ${STEP_LIST}; do
    F="${CKDIR}/spar_Ry_Sa_${S}_steps.task"
    [ -f "${F}" ] || { echo "FATAL: missing ${F}" >&2; exit 1; }
    CKPTS="${CKPTS} ${F}"
done

export CKPT_GLOB="${CKPTS}"
# Overridable: the shuffle-axis fix (local_best_response.py:434) invalidated
# every shuffle number produced before it, so a re-run needs `lbr,shuffle` too --
# `greedy` never consults V and is unaffected, so it is not worth re-paying for.
export LBR_MODES="${LBR_MODES:-minimax,minimaxshuffle,greedy}"
export SWEEP_TAG="${SWEEP_TAG:-minimax_trio_${ARM}}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-spar_minimax_phase0_${ARM}}"
export N_WORKERS="${N_WORKERS:-3}"
exec "${SCRIPT_DIR}/run_lbr_sweep.sh"
