#!/usr/bin/env bash
# Contact-density chain: wait for the enumeration collections, analyse them, then
# launch the Phase-0 training arms the analysis selects. Runs unattended.
#
# WHY A CHAIN. The three interventions (start-state range, counter-hit weight,
# pressure shaping) are all measured OFFLINE from one enumeration, which takes
# minutes. The training arms that follow take hours. Chaining means the slow part
# starts the moment the fast part picks a winner, instead of waiting for a human.
#
# THE SELECTION RULE IS FIXED HERE, NOT JUDGED LATER. The arm launched is the
# variant with the highest gamma_share among the kappa and beta sweeps. Writing
# it down in advance is the point: picking the winner after seeing the training
# results is how a null gets reported as a success.
#
# CONTROL ARM ALWAYS RUNS. Every regime comparison in this programme has been
# confounded by contact rate, so the intervention is compared against a
# same-seed, same-everything baseline arm rather than against a historical run.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/main"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

NPZ_GLOB="${NPZ_GLOB:-contact_density/cd_s*.npz}"
N_EXPECT="${N_EXPECT:-4}"
SUMMARY="${SUMMARY:-contact_density/cd_summary.json}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-12000000}"
# Without this the arms build a 65536-wide observation and die on a CUDA OOM.
RAM_MASK="${RAM_MASK:-${SCRIPT_DIR}/main/ram_mask.npy}"
LAUNCH_TRAINING="${LAUNCH_TRAINING:-True}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-7200}"
LOG="${SCRIPT_DIR}/logs/contact_chain.log"

say() { echo "[chain $(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

say "waiting for ${N_EXPECT} collections matching ${NPZ_GLOB}"
t0=$SECONDS
while true; do
    n=$(ls -1 ${NPZ_GLOB} 2>/dev/null | wc -l)
    running=$(pgrep -fc "contact_density.py --mode collect" || true)
    [ "${n}" -ge "${N_EXPECT}" ] && break
    if [ "${running}" -eq 0 ]; then
        say "no collectors running and only ${n}/${N_EXPECT} npz present"
        # Proceed on what landed rather than stalling all night, but say so
        # loudly -- a quietly reduced sample size is how an underpowered result
        # gets read as a real one.
        [ "${n}" -ge 1 ] && { say "PROCEEDING WITH ${n}/${N_EXPECT} -- REDUCED SAMPLE"; break; }
        say "FAILED: no collections produced output"; exit 1
    fi
    if [ $((SECONDS - t0)) -gt "${WAIT_TIMEOUT_S}" ]; then
        say "FAILED: timed out after ${WAIT_TIMEOUT_S}s with ${n}/${N_EXPECT}"; exit 1
    fi
    sleep 60
done
say "collections ready: $(ls -1 ${NPZ_GLOB} | tr '\n' ' ')"

say "analysing"
python -u contact_density.py --mode analyze --npz ${NPZ_GLOB} --out "${SUMMARY}" \
    2>&1 | tee -a "${LOG}"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "FAILED: analysis exited nonzero"; exit 1; }

# ---- fixed selection rule ---------------------------------------------------
read -r PICK_KIND PICK_VAL PICK_SHARE PICK_CONTACT BASE_SHARE BASE_CONTACT <<<"$(
python - "${SUMMARY}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
base = d["baseline"]
cands = [("kappa", r["kappa"], r) for r in d.get("test1", []) if r["kappa"] > 0]
cands += [("beta", r["beta"], r) for r in d.get("test2", []) if r["beta"] > 0]
cands += [("trade", r["kappa"], r) for r in d.get("test3", []) if r["kappa"] > 0]
if not cands:
    print("none 0 0 0", base["gamma_share"], base["contact"]); raise SystemExit
# MUST BEAT DOING NOTHING. The first run of this rule took the max over the
# positive-weight candidates and never compared against the baseline, so it
# selected trade=0.5 at gamma_share 10.049% when the baseline was 10.073% --
# it launched a treatment measured to be WORSE than the control.
kind, val, r = max(cands, key=lambda c: c[2]["gamma_share"])
if r["gamma_share"] <= base["gamma_share"]:
    kind, val = "none", 0
    r = base
print(kind, val, r["gamma_share"], r["contact"], base["gamma_share"], base["contact"])
PY
)"
say "baseline    : gamma_share=${BASE_SHARE} contact=${BASE_CONTACT}"
say "selected    : ${PICK_KIND}=${PICK_VAL} gamma_share=${PICK_SHARE} contact=${PICK_CONTACT}"

ATK=$(python -c "import json;print(','.join(str(x) for x in json.load(open('${SUMMARY}'))['attack_statuses']))")
RNG=$(python -c "import json;print(json.load(open('${SUMMARY}'))['range_px'])")
say "attack_statuses=${ATK}  range_px=${RNG}"

if [ "${LAUNCH_TRAINING}" != "True" ]; then say "LAUNCH_TRAINING=False, stopping"; exit 0; fi
# Every reward variant was measured to DILUTE gamma's share, so the fallback is
# the one intervention that did work: the start-state range, which touches no
# reward at all and therefore cannot dilute anything.
if [ "${PICK_KIND}" = "none" ]; then
    say "no reward variant beats baseline; falling back to the START-STATE arm"
    PICK_KIND=closerange
    PICK_VAL="${CLOSE_RANGE_PX:-64}"
fi

# ---- launch the paired arms -------------------------------------------------
# Phase 0: MINIMAX_BOOTSTRAP_KAPPA=0 so the head is INERT and cannot change the
# policy. Both arms therefore differ ONLY in the reward, and any HEADROOM
# difference is attributable to the reward rather than to a changed policy.
cd "${SCRIPT_DIR}"
say "launching CONTROL arm (baseline reward)"
nohup env RUN_SUFFIX=cdctl TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    RAM_MASK="${RAM_MASK}" MINIMAX_HEAD=factored MINIMAX_BOOTSTRAP_KAPPA=0.0 CHECKPOINT_INTERVAL=200000 \
    ./run_minimax_phase0.sh vtoff > logs/cd_arm_control.log 2>&1 &
sleep 45

if [ "${PICK_KIND}" = "closerange" ]; then
    say "launching TREATMENT arm reset_close_range=${PICK_VAL}px"
    nohup env RUN_SUFFIX=cdclose TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
        RAM_MASK="${RAM_MASK}" MINIMAX_HEAD=factored MINIMAX_BOOTSTRAP_KAPPA=0.0 \
        CHECKPOINT_INTERVAL=200000 RESET_CLOSE_RANGE="${PICK_VAL}" \
        ./run_minimax_phase0.sh vtoff > logs/cd_arm_treat.log 2>&1 &
elif [ "${PICK_KIND}" = "kappa" ] || [ "${PICK_KIND}" = "trade" ]; then
    say "launching TREATMENT arm ${PICK_KIND}_kappa=${PICK_VAL}"
    nohup env RUN_SUFFIX=cd${PICK_KIND} TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
        RAM_MASK="${RAM_MASK}" MINIMAX_HEAD=factored MINIMAX_BOOTSTRAP_KAPPA=0.0 CHECKPOINT_INTERVAL=200000 \
        "$([ "${PICK_KIND}" = trade ] && echo TRADE_KAPPA || echo COUNTERHIT_KAPPA)=${PICK_VAL}" \
        ATTACK_STATUSES="${ATK}" \
        ./run_minimax_phase0.sh vtoff > logs/cd_arm_treat.log 2>&1 &
else
    say "launching TREATMENT arm pressure_beta=${PICK_VAL}"
    nohup env RUN_SUFFIX=cdbet TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
        RAM_MASK="${RAM_MASK}" MINIMAX_HEAD=factored MINIMAX_BOOTSTRAP_KAPPA=0.0 CHECKPOINT_INTERVAL=200000 \
        PRESSURE_BETA="${PICK_VAL}" PRESSURE_RANGE="${RNG}" ATTACK_STATUSES="${ATK}" \
        ./run_minimax_phase0.sh vtoff > logs/cd_arm_treat.log 2>&1 &
fi
sleep 45
say "arms launched; control=logs/cd_arm_control.log treat=logs/cd_arm_treat.log"
say "DONE"
