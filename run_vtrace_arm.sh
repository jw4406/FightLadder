#!/bin/bash
# V-trace isolation experiment: is the replay buffer's implicit opponent-averaging
# the damping term that hid the rotational dynamics?
#
# The REFERENCE is the killed run (gamma 0.94, V-trace OFF, c_lr 1e-5), which
# CYCLED with 5 zero crossings in 390 iterations rather than collapsing
# monotonically like the baseline. See [[oscillating-selfplay-not-collapse]].
#
#   arm A   gamma 0.94   vtrace ON     <- changes ONLY the estimator
#   arm B   gamma 0.99   vtrace OFF    <- changes ONLY the horizon
#
# Together with the two runs already measured this closes a 2x2 at identical
# c_lr 1e-5 / d_lr 1e-4:
#
#                  vtrace ON                    vtrace OFF
#   gamma 0.99     baseline (monotone -558)     arm B
#   gamma 0.94     arm A                        reference (cycled)
#
# If cycling appears in BOTH vtrace-off cells and NEITHER vtrace-on cell, the
# replay buffer is supplying the damping. If it tracks gamma instead, the horizon
# is doing the work.
#
# Everything except GAMMA and VTRACE_ENABLED is held at the reference run's value
# so each arm differs from it by exactly one knob.
#
# Usage:  ./run_vtrace_arm.sh A|B
#
# Keep the arg set in sync with run_ippo_workers.sh and .vscode/launch.json.
set -u

ARM="${1:-}"
case "${ARM}" in
    A) GAMMA="0.94"; VTRACE_ENABLED="True";  TAG="A_g094_vton"  ;;
    B) GAMMA="0.99"; VTRACE_ENABLED="False"; TAG="B_g099_vtoff" ;;
    *) echo "usage: $0 A|B" >&2; exit 1 ;;
esac

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IPPO_PATH="${SCRIPT_DIR}/main/ippo.py"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

# Two concurrent runs on one 24 GiB card. Mostly caching-allocator reserve rather
# than working set, so this matters more than the raw footprint suggests.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CRITICAL -- checkpoint isolation. .task files are written by
# FileQueueTriggerCallback to ippo.py's TASK_DIR, derived from the SCRIPT path;
# it ignores --save_dir AND cwd. Both arms share model_name_prefix
# ("spar_Ry_Sa"), so without this override they overwrite each other AND the
# preserved cycling-run checkpoints in main/trained_models/tasks/todo/.
# That has already destroyed one baseline checkpoint in this project.
RUN_ROOT="${SCRIPT_DIR}/main/vtrace_arm_${TAG}"
export FIGHTLADDER_TASK_DIR="${RUN_ROOT}/trained_models/tasks"
mkdir -p "${FIGHTLADDER_TASK_DIR}/todo" "${FIGHTLADDER_TASK_DIR}/todo_continue"
SAVE_DIR="${FIGHTLADDER_TASK_DIR}/todo"

PLAYER=("Ryu")
OPPONENTS=("Sagat")
C_LR="1e-5"                 # reference value; NOT arm A's 3e-5 from the
D_LR="1e-4"                 # timescale experiment -- different experiment
V_LR="4e-4"
ENVS_PER_MATCHUP="24"
ENV_BATCH_SIZE="24"
NUM_ENV_TO_LOAD="1"
NUM_ENV_STEPS="512"
TRAINING_BATCH_SIZE="1024"
CHECKPOINT_INTERVAL="20000" # x24 envs = a checkpoint every 480k steps
TOTAL_TIMESTEPS="150000000"
NUM_PERTURBS="10"
# Inert when VTRACE_ENABLED=False, which is exactly why the flag now exists.
VTRACE_SEQ_LEN="64"
VTRACE_C_BAR="1.0"
VTRACE_RHO_BAR="5.0"
VTRACE_REPLAY_CAPACITY="15000"
# OFF for both arms -- popart is a separate variable and would confound this.
POPART="False"
USE_MIRROR="False"
EGO_SIDE="left"
SIDE="both"
TRANSFORM_ACTION="True"
MODEL_ARCH_TYPE="spar"
EGO_STYLE="learning"
ADV_STYLE="learning"
RENDER="False"
ASYNC_UPDATE="False"
MASTER_USE_STAG="False"
USE_LR_ANNEALING="False"
LR_ANNEAL_COEFF=".995"
OBS_TYPE="image"
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
LOG="${LOGS_DIR}/vtrace_arm_${TAG}.log"

CMD=(
    python -u "${IPPO_PATH}"
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
    --gamma "${GAMMA}"
    --vtrace_enabled "${VTRACE_ENABLED}"
    --vtrace_seq_len "${VTRACE_SEQ_LEN}"
    --vtrace_c_bar "${VTRACE_C_BAR}"
    --vtrace_rho_bar "${VTRACE_RHO_BAR}"
    --vtrace_replay_capacity "${VTRACE_REPLAY_CAPACITY}"
    --popart "${POPART}"
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

echo "=== vtrace arm ${ARM} (${TAG}) ==="
echo "  gamma          : ${GAMMA}"
echo "  vtrace_enabled : ${VTRACE_ENABLED}$([ "${VTRACE_ENABLED}" = "False" ] && echo '   (all VTRACE_* knobs INERT)')"
echo "  c_lr/d_lr/v_lr : ${C_LR} / ${D_LR} / ${V_LR}"
echo "  task_dir       : ${FIGHTLADDER_TASK_DIR}"
echo "  log            : ${LOG}"
nohup "${CMD[@]}" > "${LOG}" 2>&1 &
echo "  PID (python)   : $!"
