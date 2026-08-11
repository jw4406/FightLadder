#!/bin/bash
# Wait for the masked-RAM arm to reach 2.88M, then run the three cheap
# diagnostics that test the predictions made before the run started.
#
# 2.88M is chosen because the FULL-RAM arm was measured there, so every number
# below has a direct counterpart:
#
#   trunk DC/fluctuation   full RAM 24.56   predicted masked ~2.9   image 1.08
#   gap / spread           full RAM  1.15   predicted masked ~1.15  (UNCHANGED)
#   ||dw||/||w||           full RAM 138.5%  predicted masked ~141%  (UNCHANGED)
#   no-op SNR (ego)        full RAM  0.87   every arm so far 0.66-1.15
#
# The DC number is the falsifiable one: masking cuts input DC 8.4x while leaving
# ||delta_x|| identical at 8.5263, so a trunk that does NOT land near 2.9 refutes
# the dilution explanation. The bias numbers are predicted NOT to move -- the
# 484-row defect is in the output layer, which no observation change can touch.
# If masking fixes those, that diagnosis was wrong.
#
# FAIL-CLOSED. If the trainer dies, or the run collapses the way the pre-fix one
# did (score_rollout -> 0 and stuck, KL early-stops returning), this reports that
# and does NOT run diagnostics on a degenerate checkpoint.
set -uo pipefail
REPO=/home/jw4406/codebase/FightLadder
cd "${REPO}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder
python -c "import torch" 2>/dev/null || { echo "[watch] FATAL: no torch"; exit 3; }

TARGET="${TARGET:-2880000}"
CKDIR="${REPO}/main/minimax_phase0_vtoff_rammasked/trained_models/tasks/todo"
LOG="${REPO}/logs/minimax_phase0_vtoff_rammasked.log"
OUT="${REPO}/logs/masked_diag.log"
: > "${OUT}"

CK="${CKDIR}/spar_Ry_Sa_${TARGET}_steps.task"
for i in $(seq 1 240); do
    if [ -f "${CK}" ]; then break; fi
    if ! pgrep -f "[i]ppo.py" >/dev/null 2>&1; then
        echo "[watch] TRAINER GONE before ${TARGET} -- not running diagnostics" | tee -a "${OUT}"
        exit 2
    fi
    # Collapse guard: the pre-fix run pinned score_rollout at 0 from 159,744 on.
    # Reaching the checkpoint is not enough; it has to still be a real game.
    ZER=$(grep -oP "score_rollout *\| *\K[-0-9.e+]+" "${LOG}" 2>/dev/null | tail -30 \
          | awk '$1==0{c++} END{print c+0}')
    if [ "${ZER}" -ge 28 ]; then
        echo "[watch] COLLAPSED: score_rollout is 0 in ${ZER} of the last 30 rows." | tee -a "${OUT}"
        echo "[watch] Refusing to run diagnostics on a degenerate checkpoint." | tee -a "${OUT}"
        exit 4
    fi
    sleep 30
done
[ -f "${CK}" ] || { echo "[watch] TIMED OUT waiting for ${TARGET}" | tee -a "${OUT}"; exit 5; }

echo "[watch] ${TARGET} ready $(date '+%F %T')" | tee -a "${OUT}"
grep -c "training stopped at step" "${LOG}" 2>/dev/null \
    | sed 's/^/[watch] kl-stops so far: /' | tee -a "${OUT}"

run() {  # name, script, extra args
    echo "" | tee -a "${OUT}"
    echo "=== $1 ===" | tee -a "${OUT}"
    python -u "${REPO}/main/$2" --ckpt "${CK}" "${@:3}" 2>&1 \
        | grep -vE "^hello|Warning|warn" | tee -a "${OUT}"
    # PIPESTATUS, not $? -- $? is grep's. A crashed diagnostic that printed
    # nothing would otherwise read as a clean result.
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "[watch] FAILED: $1" | tee -a "${OUT}"
}

run "TRUNK CONTROL (predict DC/fluct ~2.9 vs full-RAM 24.56)" \
    minimax_trunk_control.py --steps 200 --n_envs 6
run "NO-OP DECOMPOSITION (predict gap/spread ~1.15, UNCHANGED)" \
    minimax_noop_decomp.py --steps 250 --n_envs 6 --out noop_masked_${TARGET}.json
run "AXIS SNR (every arm so far 0.66-1.15)" \
    minimax_axis_diag.py --steps 250 --n_envs 6 --out axis_masked_${TARGET}.json
run "BIAS MECHANISM (predict ||dw||/||w|| ~141%, UNCHANGED)" \
    minimax_bias_mech.py --steps 200 --n_envs 6

echo "" | tee -a "${OUT}"
echo "MASKED_DIAG_DONE" | tee -a "${OUT}"
