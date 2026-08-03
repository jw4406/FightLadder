#!/bin/bash
# Frozen-strong-ego experiment: can a FRESH adversary learn against a strong
# STATIONARY opponent?
#
# Separates the two candidate causes of the adversary's entropy collapse:
#   frozen WEAK   ego (--ego_style zero_action)  -> adversary learns, advH ~1.2   [done]
#   frozen STRONG ego (this run)                 -> ?
#   MOVING strong ego (self-play)                -> adversary collapses, advH 0.02 [done]
# Learns here   -> the collapse is caused by the opponent MOVING (non-stationarity).
# Collapses too -> caused by opponent STRENGTH making the advantage signal degenerate.
#
# --model_file loads the WHOLE policy (ego + adversary + critic), so
# --reinit_adversary True resets the adversary head to uniform (ln22 = 3.09) and
# rebuilds dstb_optimizer. Without it the adversary would resume already collapsed
# at advH ~ 0.02 and a negative result would be uninformative.
#
# Keep the arg set in sync with run_ippo_workers.sh and .vscode/launch.json.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IPPO_PATH="${SCRIPT_DIR}/main/ippo.py"

# Self-contained: run_ippo_workers.sh assumes the caller already activated the
# env, which silently fails (ModuleNotFoundError: av) when launched detached.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

PLAYER=("Ryu")
OPPONENTS=("Guile")

# Strong ego: at 1M this checkpoint's ego had rating 1400 and score_rollout 1.00
# (won every rollout game). Same architecture as the current run.
MODEL_FILE="${SCRIPT_DIR}/main/trained_models/tasks/todo/spar_Ry_Gu_1000000_steps.task"
EGO_STYLE="frozen"
ADV_STYLE="learning"
REINIT_ADVERSARY="True"

SAVE_DIR="${SCRIPT_DIR}/main/trained_models/tasks/todo/critic_fixed_base_runs"

# --- everything below matches the live self-play run so the comparison holds ---
C_LR="1e-5"
D_LR="1e-4"
V_LR="4e-4"
USE_MIRROR="False"
EGO_SIDE="left"
SIDE="both"
ENVS_PER_MATCHUP="16"
NUM_ENV_STEPS="64"
VTRACE_SEQ_LEN="64"
CHECKPOINT_INTERVAL="6250"     # x16 envs = 100k timesteps per checkpoint
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

mkdir -p "${SAVE_DIR}"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

CMD=(
    python "${IPPO_PATH}"
    --player "${PLAYER[@]}"
    --opponents "${OPPONENTS[@]}"
    --model_file "${MODEL_FILE}"
    --reinit_adversary "${REINIT_ADVERSARY}"
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

echo "=== frozen-ego experiment ==="
echo "  ego        : frozen from $(basename "${MODEL_FILE}")"
echo "  adversary  : learning, re-initialized to uniform"
echo "  save_dir   : ${SAVE_DIR}"
nohup "${CMD[@]}" > "${LOGS_DIR}/frozen_ego_exp.log" 2>&1 &
echo "  PID        : $!"
echo "  log        : ${LOGS_DIR}/frozen_ego_exp.log"
