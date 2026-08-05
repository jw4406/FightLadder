#!/bin/bash
# Timescale experiment: is the ego being OUTPACED by the adversary?
#
# Baseline (spar_Ry_Sa, killed at 10.5M) had c_lr 1e-5 / d_lr 1e-4 -- a 10x
# separation -- and showed the ego frozen while the adversary ran away:
#   ego_approx_kl 0.0012 vs adv 0.022   (18x)
#   ego_clip_frac 0.021  vs adv 0.16    (8x)
#   egoH flat at -1.18 for 10M steps while advH sharpened -1.03 -> -0.33
#   rating_gap 0 -> -558, ego won 22/570 episodes (3.9%) by 10.08M
#
# KL scales ~ lr^2, so lifting ego_approx_kl 0.0012 -> ~0.010 needs ~sqrt(8) =
# 2.9x on c_lr. Hence 3e-5. Target band is ego_approx_kl 0.008-0.015; tune by
# watching that, NOT by trusting the quadratic.
#
#   arm A   c_lr 3e-5  d_lr 1e-4    separation 10x -> 3x
#   arm B   c_lr 3e-5  d_lr 3e-4    separation stays 10x, both faster
#
# A alone working implicates the RATIO; both working implicates the ego's
# ABSOLUTE rate. Everything else is held at the baseline config so the runs are
# comparable to the logs we already have.
#
# Usage:  ./run_timescale_arm.sh A|B [c_bar]
#
# Optional 2nd arg sweeps V-trace c_bar (default 1.0). c_bar truncates the TRACE
# ratio, so it sets variance / effective credit horizon and does NOT move the
# fixed point -- unlike rho_bar. Measured c_sat_frac ~0.47 at c_bar=1, i.e. half
# the traces clip, so the effective horizon is far shorter than seq_len=64 or
# gamma=0.99 imply. Sweep 1.0 -> 2.0 -> 5.0; ratio_max hit 119, so do not remove
# the bar entirely. A non-default c_bar gets its own RUN_ROOT and log, so runs
# cannot overwrite each other's checkpoints.
#
# Keep the arg set in sync with run_ippo_workers.sh and .vscode/launch.json.
set -u

ARM="${1:-}"
case "${ARM}" in
    A) C_LR="3e-5"; D_LR="1e-4" ;;
    B) C_LR="3e-5"; D_LR="3e-4" ;;
    *) echo "usage: $0 A|B [c_bar]" >&2; exit 1 ;;
esac

VTRACE_C_BAR="${2:-1.0}"
VTRACE_RHO_BAR="5.0"        # sets the FIXED POINT -- do not sweep casually
# Tag the run when c_bar is non-default so its checkpoints, task dir and log are
# distinct. Two runs sharing model_name_prefix ("spar_Ry_Sa") would otherwise
# write identical .task filenames.
if [ "${VTRACE_C_BAR}" = "1.0" ]; then TAG="${ARM}"; else TAG="${ARM}_c${VTRACE_C_BAR}"; fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IPPO_PATH="${SCRIPT_DIR}/main/ippo.py"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

# Reduce allocator fragmentation so two concurrent runs have a chance of
# coexisting on one 24 GiB card. The baseline run alone held 20.77 GiB, which is
# mostly caching-allocator reserve rather than working set.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CRITICAL: --save_dir is honored for .mp4 ONLY. .task checkpoints are written by
# FileQueueTriggerCallback to ippo.py's TASK_DIR, which is derived from the
# SCRIPT's path -- so it ignores both --save_dir and cwd. Both arms share
# model_name_prefix ("spar_Ry_Sa"), so without an override they write identical
# filenames and silently overwrite each other. This already destroyed one
# baseline checkpoint; a separate cwd does NOT prevent it.
#
# FIGHTLADDER_TASK_DIR (ippo.py:42) is the only thing that actually redirects
# them. Verify after the first checkpoint interval that .task files appear under
# RUN_ROOT and NOT under main/trained_models/tasks/todo/.
RUN_ROOT="${SCRIPT_DIR}/main/timescale_arm_${TAG}"
export FIGHTLADDER_TASK_DIR="${RUN_ROOT}/trained_models/tasks"
mkdir -p "${FIGHTLADDER_TASK_DIR}/todo" "${FIGHTLADDER_TASK_DIR}/todo_continue"

PLAYER=("Ryu")
OPPONENTS=("Sagat")
SAVE_DIR="${RUN_ROOT}/trained_models/tasks/todo"

V_LR="4e-4"
USE_MIRROR="False"
EGO_SIDE="left"
SIDE="both"
ENVS_PER_MATCHUP="24"
NUM_ENV_STEPS="512"
VTRACE_SEQ_LEN="64"
# Discount. EMPTY = leave ippo.py's per-path defaults alone (spar keeps 0.99).
# Measured on arm A's own checkpoints: the return-prediction ceiling is 0.18-0.20
# at gamma 0.75-0.9 vs 0.055 at 0.99. But that is PREDICTABILITY of the return,
# not policy quality -- a shorter horizon also makes the agent myopic. 0.94 is
# what the IPPO paths in the same file already use (~2.2 s vs ~13 s at 0.99,
# against rounds of ~27 s).
GAMMA=""
CHECKPOINT_INTERVAL="20000"
TRAINING_BATCH_SIZE="1024"
TOTAL_TIMESTEPS="150000000"
TRANSFORM_ACTION="True"
MODEL_ARCH_TYPE="spar"
NUM_ENV_TO_LOAD="1"
ENV_BATCH_SIZE="24"
NUM_PERTURBS="10"
RENDER="False"
ASYNC_UPDATE="False"
MASTER_USE_STAG="False"
USE_LR_ANNEALING="False"
LR_ANNEAL_COEFF=".995"
OBS_TYPE="image"
EGO_STYLE="learning"
ADV_STYLE="learning"
USE_STAGNATION_EARLY_STOP="False"
USE_STAGNATION_VELOCITY_SIGNAL="False"
USE_STAGNATION_ENTROPY_SIGNAL="False"
STAGNATION_PATIENCE="20000"
STAGNATION_TOLERANCE="1e-4"
STAGNATION_REL_TOLERANCE="0.05"
STAGNATION_EMA_BETA="0.99"
STAGNATION_EPS="1e-8"
STAGNATION_EVAL_GAMES="0"
ENTROPY_STAGNATION_WEIGHT="1.0"
STAGNATION_LR_FACTOR="0.999"
STAGNATION_LR_PATIENCE="150"

LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
LOG="${LOGS_DIR}/timescale_arm_${TAG}.log"

CMD=(
    python "${IPPO_PATH}"
    --player "${PLAYER[@]}"
    --opponents "${OPPONENTS[@]}"
    --ego_style "${EGO_STYLE}"
    --adv_style "${ADV_STYLE}"
    --save_dir "${SAVE_DIR}"
    --num_env_to_load "${NUM_ENV_TO_LOAD}"
    --env_batch_size "${ENV_BATCH_SIZE}"
    --c_lr "${C_LR}" --d_lr "${D_LR}" --v_lr "${V_LR}"
    --num_perturbs "${NUM_PERTURBS}"
    --use_mirror "${USE_MIRROR}"
    --ego_side "${EGO_SIDE}"
    --side "${SIDE}"
    --envs_per_matchup "${ENVS_PER_MATCHUP}"
    --num_env_steps "${NUM_ENV_STEPS}"
    --vtrace_seq_len "${VTRACE_SEQ_LEN}"
    --vtrace_c_bar "${VTRACE_C_BAR}"
    --vtrace_rho_bar "${VTRACE_RHO_BAR}"
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"
    --training_batch_size "${TRAINING_BATCH_SIZE}"
    --total_timesteps "${TOTAL_TIMESTEPS}"
    --transform_action "${TRANSFORM_ACTION}"
    --model_arch_type "${MODEL_ARCH_TYPE}"
    --use_lr_annealing "${USE_LR_ANNEALING}"
    --lr_anneal_coeff "${LR_ANNEAL_COEFF}"
    --render "${RENDER}"
    --async_update "${ASYNC_UPDATE}"
    --master_use_stag "${MASTER_USE_STAG}"
    --obs_type "${OBS_TYPE}"
    --use_stagnation_early_stop "${USE_STAGNATION_EARLY_STOP}"
    --use_stagnation_velocity_signal "${USE_STAGNATION_VELOCITY_SIGNAL}"
    --use_stagnation_entropy_signal "${USE_STAGNATION_ENTROPY_SIGNAL}"
    --stagnation_patience "${STAGNATION_PATIENCE}"
    --stagnation_tolerance "${STAGNATION_TOLERANCE}"
    --stagnation_rel_tolerance "${STAGNATION_REL_TOLERANCE}"
    --stagnation_ema_beta "${STAGNATION_EMA_BETA}"
    --stagnation_eps "${STAGNATION_EPS}"
    --stagnation_eval_games "${STAGNATION_EVAL_GAMES}"
    --entropy_stagnation_weight "${ENTROPY_STAGNATION_WEIGHT}"
    --stagnation_lr_factor "${STAGNATION_LR_FACTOR}"
    --stagnation_lr_patience "${STAGNATION_LR_PATIENCE}"
)
[ -n "${GAMMA}" ] && CMD+=( --gamma "${GAMMA}" )

echo "=== timescale arm ${TAG} ==="
echo "  c_lr (ego) : ${C_LR}      d_lr (adv) : ${D_LR}   v_lr : ${V_LR}"
echo "  vtrace     : c_bar ${VTRACE_C_BAR}   rho_bar ${VTRACE_RHO_BAR}"
echo "  gamma      : ${GAMMA:-<ippo.py default: spar 0.99>}"
echo "  task_dir   : ${FIGHTLADDER_TASK_DIR}   (isolates .task checkpoints)"
echo "  log        : ${LOG}"
nohup "${CMD[@]}" > "${LOG}" 2>&1 &
echo "  PID        : $!"
