#!/bin/bash
# WHERE does action-conditional value live, if anywhere?
#
# The null result so far is a fact about latent_vf and nothing else. Under
# --minimax_stop_grad the critic encoder is shaped ONLY by V's objective, which
# predicts a scalar, so action-conditional structure has no reason to survive to
# that layer -- and MinimaxHead reads latent_vf exclusively. This sweeps three
# sites UPSTREAM of it on the same checkpoint, same target, same action
# distribution, so the numbers sit directly beside the existing ones:
#
#   12.48M uniform reward   latent_vf  0.0183 -> 0.0185 -> 0.0177   gain -0.0005
#   12.48M uniform return   latent_vf  0.0615 -> 0.0622 -> 0.0617   gain +0.0003
#
# --uniform_actions because the ego entropy at this checkpoint is -0.26 against a
# uniform -3.09: policy-sampled actions barely cover the 22x22 grid, and a real
# action effect could hide in cells the ridge almost never sees.
#
# READING IT:
#   gain at conv_pi/latent_pi but not latent_vf -> the information EXISTS and the
#       architecture discards it before the head can use it. Fix is the head's
#       INPUT (feed it conv features, or drop stop_grad), not its capacity.
#   gain nowhere -> the direction is closed; no representation in this network
#       carries action-conditional value at the ridge's reach.
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "FATAL: no torch"; exit 3; }

CK="${CK:-${SCRIPT_DIR}/main/minimax_phase0_vton/trained_models/tasks/todo/spar_Ry_Sa_12480000_steps.task}"
[ -f "${CK}" ] || { echo "FATAL: missing ${CK}"; exit 1; }
LOG="${SCRIPT_DIR}/logs/probe_sites.log"
mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/main/probe_sites"
: > "${LOG}"

FAIL=0
for SITE in conv_pi latent_pi conv_vf; do
  for TGT in reward return; do
    echo "=== site=${SITE} target=${TGT} $(date +%H:%M:%S) ===" | tee -a "${LOG}"
    python -u "${SCRIPT_DIR}/main/minimax_probe_ceiling.py" \
        --ckpt "${CK}" --target "${TGT}" --probe_site "${SITE}" \
        --uniform_actions --steps 9000 --n_envs 12 \
        --out "probe_sites/site_${SITE}_${TGT}.json" 2>&1 \
      | grep -E "PROBE CEILING|latent |trained |ACTION GAIN|=>|episodes from" | tee -a "${LOG}"
    # PIPESTATUS, not $?: $? is grep's. A crashed probe that printed nothing
    # would otherwise look like a clean null -- that exact fail-open bug has
    # appeared three times in this project.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "  FAILED: ${SITE}/${TGT}" | tee -a "${LOG}"; FAIL=1
    fi
  done
done
echo "PROBE_SITES_DONE fail=${FAIL}" | tee -a "${LOG}"
exit "${FAIL}"
