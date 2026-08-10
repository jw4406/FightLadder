#!/bin/bash
# Does V-trace change WHICH STATES self-play visits?
#
# Rolls every MATCHED checkpoint of the vton and vtoff Phase 0 arms on-policy and
# records the emulator state (agent_x/y, enemy_x/y, hp, status, clock) -- an
# arm-independent descriptor, deliberately NOT the latent, since comparing two
# arms through their own encoders confounds "the representation changed" with
# "the visitation changed".
#
# TWO SEEDS PER ARM ARE MANDATORY, not a nicety: compare_visitation.py reads
# every cross-arm distance against the between-seed distance and refuses to run
# without it. A distance with no null is how this project previously produced
# three confident wrong conclusions.
#
# MATCHED STEPS ONLY. vton reached 16.32M before it was killed; vtoff is still
# training and is the binding constraint. The intersection is computed below
# rather than hardcoded, so re-running this later -- as vtoff catches up --
# automatically widens the comparison. That is the argument for leaving both
# arms running.
#
# Usage:  ./run_visitation.sh            (all matched ckpts, seeds 0 and 1)
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# conda activate is NOT optional -- three separate bugs in this project came from
# a probe running without it, crashing on `No module named torch`, and a caller
# reporting the resulting empty output as a real (reassuring) result.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="${SCRIPT_DIR}/main/visitation"
LOG="${SCRIPT_DIR}/logs/visitation.log"
mkdir -p "${OUT}" "${SCRIPT_DIR}/logs"
: > "${LOG}"

STEPS="${STEPS:-3000}"
N_ENVS="${N_ENVS:-12}"
SEEDS="${SEEDS:-0 1}"
ARMS="${ARMS:-vton vtoff vtoff_popart}"

steps_of() {   # list checkpoint step-counts available for an arm
    ls "${SCRIPT_DIR}/main/minimax_phase0_$1/trained_models/tasks/todo"/spar_Ry_Sa_*_steps.task \
        2>/dev/null | sed -E 's/.*_([0-9]+)_steps\.task/\1/' | sort -n
}

# Matched = present in BOTH primary arms. popart is a SECOND variable (it is
# vtoff+popart, not a vtrace contrast) so it rides along on whatever it has.
MATCHED=$(comm -12 <(steps_of vton) <(steps_of vtoff))
if [ -z "${MATCHED}" ]; then
    echo "FATAL: vton and vtoff share no checkpoint step -- nothing to compare" | tee -a "${LOG}"
    exit 1
fi
echo "matched ckpts: $(echo ${MATCHED} | tr '\n' ' ')" | tee -a "${LOG}"

FAIL=0
for ARM in ${ARMS}; do
    DIR="${SCRIPT_DIR}/main/minimax_phase0_${ARM}/trained_models/tasks/todo"
    for S in ${MATCHED}; do
        CK="${DIR}/spar_Ry_Sa_${S}_steps.task"
        [ -f "${CK}" ] || { echo "  skip ${ARM} @ ${S} (no checkpoint)" | tee -a "${LOG}"; continue; }
        for SEED in ${SEEDS}; do
            O="${OUT}/vis_${ARM}_${S}_s${SEED}.npz"
            [ -f "${O}" ] && { echo "  have ${ARM} ${S} s${SEED}" | tee -a "${LOG}"; continue; }
            echo "=== ${ARM} ${S} seed ${SEED} $(date +%H:%M:%S) ===" | tee -a "${LOG}"
            python -u "${SCRIPT_DIR}/main/state_visitation.py" \
                --ckpt "${CK}" --steps "${STEPS}" --n_envs "${N_ENVS}" \
                --seed "${SEED}" --out "${O}" 2>&1 | tee -a "${LOG}"
            # PIPESTATUS, not $? -- $? is tee's. A silent failure here would
            # leave a missing .npz that the comparison reads as "fewer seeds".
            if [ "${PIPESTATUS[0]}" -ne 0 ]; then
                echo "  FAILED: ${ARM} ${S} s${SEED}" | tee -a "${LOG}"; FAIL=1
            fi
        done
    done
done

echo "VISITATION_ROLLOUTS_DONE fail=${FAIL}" | tee -a "${LOG}"
[ "${FAIL}" -eq 0 ] || { echo "one or more rollouts failed -- NOT comparing" | tee -a "${LOG}"; exit 1; }

python -u "${SCRIPT_DIR}/main/compare_visitation.py" \
    --dir "${OUT}" --arms vton vtoff \
    --plot "${SCRIPT_DIR}/main/visitation/visitation.png" 2>&1 | tee -a "${LOG}"
echo "VISITATION_DONE" | tee -a "${LOG}"
