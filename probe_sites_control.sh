#!/bin/bash
# POSITIVE CONTROL for probe_sites.sh -- can the ridge read the actor sites AT ALL?
#
# WHY THIS IS REQUIRED, not optional. The uniform-action sweep found no action
# gain at conv_pi or latent_pi. That is only evidence if the probe can decode
# SOMETHING at those sites, and the state-only column says it barely can:
#
#     site        target   state-only EV
#     conv_pi     reward        0.0001
#     latent_pi   reward        0.0040
#     latent_vf   reward        0.0183     <- the site where the ridge demonstrably works
#
# At latent_vf the ridge is known-good: 0.29 (return) and 0.17 (reward) under
# POLICY actions. So a null there isolates the ACTION. At conv_pi a null could
# equally mean "a linear ridge cannot read raw CNN features", which would make
# the whole comparison uninformative -- absence of evidence, not evidence of
# absence.
#
# This runs the actor sites under POLICY actions, the condition where latent_vf
# scores 0.29/0.17, so the numbers are directly comparable.
#
#   healthy state-only EV here -> the ridge CAN read these sites; the uniform
#       nulls mean what they say, and the direction is closed.
#   still ~0 here              -> the probe cannot decode these representations
#       at all; the uniform nulls are uninformative and a NONLINEAR probe (MLP)
#       is required before any conclusion about the CNN.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "FATAL: no torch"; exit 3; }

CK="${CK:-${SCRIPT_DIR}/main/minimax_phase0_vton/trained_models/tasks/todo/spar_Ry_Sa_12480000_steps.task}"
LOG="${SCRIPT_DIR}/logs/probe_sites_control.log"
mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/main/probe_sites"
: > "${LOG}"

FAIL=0
for SITE in conv_pi latent_pi; do
  for TGT in return reward; do
    echo "=== CONTROL site=${SITE} target=${TGT} POLICY-actions $(date +%H:%M:%S) ===" | tee -a "${LOG}"
    python -u "${SCRIPT_DIR}/main/minimax_probe_ceiling.py" \
        --ckpt "${CK}" --target "${TGT}" --probe_site "${SITE}" \
        --steps 9000 --n_envs 12 \
        --out "probe_sites/ctrl_${SITE}_${TGT}.json" 2>&1 \
      | grep -E "PROBE CEILING|^  latent |ACTION GAIN|episodes from" | tee -a "${LOG}"
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "  FAILED: ${SITE}/${TGT}" | tee -a "${LOG}"; FAIL=1; }
  done
done
echo "PROBE_CONTROL_DONE fail=${FAIL}" | tee -a "${LOG}"
exit "${FAIL}"
