#!/bin/bash
# Full LBR sweep over an entire checkpoint series.
#
# run_lbr_eval.sh runs a hand-listed set of checkpoints serially. This runs the
# WHOLE series, sharded across N concurrent workers, and is resumable -- so a
# multi-day sweep survives a reboot or a kill without redoing finished work.
#
# Cost model (measured, not estimated):
#   ~0.97 s per decision point, and
#   decision_points ~= ceil(EPISODES / N_ENVS) * ep_len_mean
# With ep_len_mean 230-435 on this run that is ~20-30 min per sub-run, and each
# checkpoint is 6 sub-runs (3 modes x 2 directions) -> ~2.5-3 h per checkpoint.
#
# Resume: a checkpoint is skipped only when its marker file exists, which is
# written after ALL of its sub-runs succeed. A partially-finished checkpoint is
# redone from scratch rather than left half-measured.
#
# Keep the arg set in sync with run_lbr_eval.sh and .vscode/launch.json.
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LBR_PATH="${SCRIPT_DIR}/main/local_best_response.py"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

# --- what to sweep ----------------------------------------------------------
# All four of these are env-overridable so a second sweep can run without
# editing this file. They MUST all be set together for a different checkpoint
# series, because arm A's checkpoints share BASENAMES with the baseline's
# (spar_Ry_Sa_<N>_steps.task in both trees). Sharing any one of them silently
# corrupts the other sweep:
#   CKPT_GLOB      which series to sweep
#   SWEEP_TAG      namespaces logs/ and the resume markers -- without it the
#                  identical basenames make arm A SKIP every checkpoint the
#                  baseline already finished
#   OUTPUT_SUBDIR  br_rewards/<subdir>/ -- without it both sweeps derive the
#                  SAME subdir from the checkpoint name and overwrite each other
#   LBR_MODES      subset of {lbr,greedy,shuffle}
CKPT_GLOB="${CKPT_GLOB:-${SCRIPT_DIR}/main/trained_models/tasks/todo/spar_Ry_Sa_*_steps.task}"
SWEEP_TAG="${SWEEP_TAG:-}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"
# Measured aggregate throughput on this 32-core box (decisions/s across workers):
#   1 worker  0.89 s/decision -> 1.12
#   3 workers 1.38 s/decision -> 2.17   <- peak
#   5 workers 2.44 s/decision -> 2.05   (load 38.9, oversubscribed and NET SLOWER)
# Do not raise this without re-measuring; past 3 the per-worker slowdown is worse
# than linear and total throughput goes DOWN.
N_WORKERS="${N_WORKERS:-3}"

# Transient CUDA OOM killed 10 checkpoints on the first attempt when a training
# job held 20.7 GiB. Retrying costs a couple of minutes and saves a 40-min
# sub-run's worth of queue position.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-120}"

# --- LBR settings (full fidelity -- nothing traded away for speed) -----------
# A checkpoint records the WIDTH of its observation but not WHICH RAM bytes, so a
# masked-ram arm cannot be evaluated without the .npy used for training.
# infer_obs_kwargs() hard-exits when it sees a 1-D obs narrower than full RAM and
# no mask, so leaving this empty on a masked arm fails loudly rather than
# silently evaluating the wrong observation. Empty = image or full-ram arm.
RAM_MASK="${RAM_MASK:-}"
EVAL_PROT="both"         # both directions: eps_ego AND eps_adv
LBR_CONTROLS="True"      # legacy all-or-nothing; IGNORED when LBR_MODES is set
# Subset of {lbr,greedy,shuffle}. Empty = legacy (honour LBR_CONTROLS).
# Set to "greedy" for a 3x cheaper sweep: lbr-with-critic scored eps <= 0 on every
# checkpoint measured (vacuous), and lbr ~= shuffle established that the critic
# contributes nothing to branch selection. Greedy is the only variant producing a
# non-vacuous bound. The highest-priority mode present owns the sidecar JSON and
# the selfplay_rewards/ file, so greedy-only still yields a computable eps.
LBR_MODES="${LBR_MODES:-}"
LBR_EPISODES="50"
# Episodes finish in rounds of N_ENVS at a time, so the real cost is
# ceil(EPISODES / N_ENVS) rounds. 12 envs needs 5 rounds (60 episodes stepped for
# 50 wanted, 17% wasted); 13 needs only 4 (52 episodes, 4% wasted). One extra env
# process buys a whole round back.
LBR_N_ENVS="13"
LBR_EGO_TOPK="4"
LBR_STRIDE="1"
LBR_SEED="0"
LBR_MAX_STEPS="100000"
TRAINING_STYLE="spar"
BR_INDEX="0"
DEVICE="cuda"

LOGS_DIR="${SCRIPT_DIR}/logs/lbr_sweep${SWEEP_TAG:+_${SWEEP_TAG}}"
MARK_DIR="${LOGS_DIR}/done"
mkdir -p "${LOGS_DIR}" "${MARK_DIR}"

# Sort numerically by step count so each worker's log reads in training order.
mapfile -t ALL_CKPTS < <(ls ${CKPT_GLOB} 2>/dev/null \
    | sed -E 's/.*_([0-9]+)_steps\.task/\1 &/' | sort -n | cut -d' ' -f2-)

if [ "${#ALL_CKPTS[@]}" -eq 0 ]; then
    echo "No checkpoints matched: ${CKPT_GLOB}" >&2
    exit 1
fi

run_worker() {
    local wid="$1"
    local log="${LOGS_DIR}/worker_${wid}.log"
    : > "${log}"
    for i in "${!ALL_CKPTS[@]}"; do
        # Interleave rather than block-partition: episode length grows with
        # training, so contiguous shards would leave worker 0 idle for hours
        # while the last worker chews the long late checkpoints.
        if [ $(( i % N_WORKERS )) -ne "${wid}" ]; then continue; fi
        local ckpt="${ALL_CKPTS[$i]}"
        local base; base="$(basename "${ckpt}" .task)"
        local mark="${MARK_DIR}/${base}.done"
        if [ -f "${mark}" ]; then
            echo "[w${wid}] SKIP (done): ${base}" | tee -a "${log}"
            continue
        fi
        local rc=1 attempt=1
        while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
            echo "[w${wid}] START $(date '+%F %T'): ${base} (attempt ${attempt}/${MAX_ATTEMPTS})" | tee -a "${log}"
            python -u "${LBR_PATH}" \
                --main_checkpoint_model_path "${ckpt}" \
                --ram_mask "${RAM_MASK}" \
                --eval_prot "${EVAL_PROT}" \
                --lbr_ego_topk "${LBR_EGO_TOPK}" \
                --lbr_stride "${LBR_STRIDE}" \
                --lbr_episodes "${LBR_EPISODES}" \
                --lbr_n_envs "${LBR_N_ENVS}" \
                --lbr_seed "${LBR_SEED}" \
                --lbr_controls "${LBR_CONTROLS}" \
                --lbr_modes "${LBR_MODES}" \
                --output_subdir "${OUTPUT_SUBDIR}" \
                --lbr_max_steps "${LBR_MAX_STEPS}" \
                --training_style "${TRAINING_STYLE}" \
                --br_index "${BR_INDEX}" \
                --device "${DEVICE}" >> "${log}" 2>&1
            rc=$?
            # 130/143 are Ctrl-C / SIGTERM: the operator meant it, so do not retry.
            if [ "${rc}" -eq 0 ] || [ "${rc}" -eq 130 ] || [ "${rc}" -eq 143 ]; then break; fi
            echo "[w${wid}] RETRY $(date '+%F %T'): ${base} (rc=${rc}), sleeping ${RETRY_SLEEP_S}s" | tee -a "${log}"
            sleep "${RETRY_SLEEP_S}"
            attempt=$(( attempt + 1 ))
        done
        if [ "${rc}" -eq 0 ]; then
            touch "${mark}"
            echo "[w${wid}] OK    $(date '+%F %T'): ${base}" | tee -a "${log}"
        else
            echo "[w${wid}] FAIL  $(date '+%F %T'): ${base} (rc=${rc})" | tee -a "${log}"
        fi
    done
    echo "[w${wid}] WORKER_DONE" | tee -a "${log}"
}

echo "=== LBR sweep ==="
echo "  glob        : ${CKPT_GLOB}"
echo "  checkpoints : ${#ALL_CKPTS[@]}"
echo "  modes       : ${LBR_MODES:-<legacy: controls=${LBR_CONTROLS}>}"
echo "  out subdir  : ${OUTPUT_SUBDIR:-<derived from ckpt name>}"
echo "  workers     : ${N_WORKERS}  (${LBR_N_ENVS} envs each)"
echo "  directions  : ${EVAL_PROT}   controls: ${LBR_CONTROLS}   episodes: ${LBR_EPISODES}"
echo "  ram_mask    : ${RAM_MASK:-<none: image or full-ram arm>}"
echo "  logs        : ${LOGS_DIR}/worker_*.log"
echo "  resume      : marker files in ${MARK_DIR}"

for w in $(seq 0 $(( N_WORKERS - 1 ))); do
    run_worker "${w}" &
done
wait
echo "SWEEP_DONE $(date '+%F %T')"
