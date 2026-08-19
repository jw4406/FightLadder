#!/bin/bash

# Number of parallel ippo instances to run
NUM_WORKERS=1
# If True, run each worker detached with nohup and log redirection.
# If False, run without nohup in the current shell.
RUN_LIVE="False"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to ippo.py
IPPO_PATH="${SCRIPT_DIR}/main/ippo.py"

# Arguments from launch.json (Python Debugger: ippo.py)
PLAYER=("Ryu")
OPPONENTS=("Sagat")
NUM_ENV_TO_LOAD="1"
ENV_BATCH_SIZE="24"
C_LR="1e-5"
D_LR="1e-4"
V_LR="4e-4"
NUM_PERTURBS="10"
USE_MIRROR="False"
EGO_SIDE="left"
# LOAD_PATH=""  # Optional: set this to pass --load_path
# TRAINING_STYLE=""  # Optional: set this to pass --training_style
# CONTINUE_TRAINING=""  # Optional: set this to pass --continue_training
# LEFT_MODEL_FILE=""  # Optional: set this to pass --left-model-file
# RIGHT_MODEL_FILE=""  # Optional: set this to pass --right-model-file
SAVE_DIR="/home/jw4406/codebase/FightLadder/main/trained_models/tasks/todo/"
USE_LR_ANNEALING="False"
LR_ANNEAL_COEFF=".995"
CHECKPOINT_INTERVAL="20000"
TOTAL_TIMESTEPS="150000000"
TRAINING_BATCH_SIZE="1024"
TRANSFORM_ACTION="True"
NUM_ENV_STEPS="512"
EGO_STYLE="learning"
ADV_STYLE="learning"
ENVS_PER_MATCHUP="24"
SIDE="both"
RENDER="False"
MODEL_FILE=""
VTRACE_SEQ_LEN="64"
# c_bar truncates the TRACE ratio: sets variance / effective credit
# horizon, does NOT move the fixed point. Measured c_sat_frac ~0.47 at 1.0,
# so ~half the traces clip. rho_bar truncates the TD-error ratio and DOES
# set the fixed point -- changing it changes what is learned.
VTRACE_C_BAR="1.0"
VTRACE_RHO_BAR="5.0"
# Discount. EMPTY = ippo.py per-path defaults (spar 0.99, ippo paths 0.94).
GAMMA="0.94"


ASYNC_UPDATE="False"
MODEL_ARCH_TYPE="spar"
OBS_TYPE="image"
# Create logs directory if it doesn't exist
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# Run NUM_WORKERS copies of ippo in parallel
for i in $(seq 1 ${NUM_WORKERS}); do
    echo "Starting ippo instance ${i}..."
    CMD=(
        python "${IPPO_PATH}"
        --player "${PLAYER[@]}"
        --opponents "${OPPONENTS[@]}"
        --num_env_to_load "${NUM_ENV_TO_LOAD}"
        --env_batch_size "${ENV_BATCH_SIZE}"
        --c_lr "${C_LR}"
        --d_lr "${D_LR}"
        --v_lr "${V_LR}"
        --num_perturbs "${NUM_PERTURBS}"
        --use_mirror "${USE_MIRROR}"
	--ego_side "${EGO_SIDE}"
        --save_dir "${SAVE_DIR}"
        --use_lr_annealing "${USE_LR_ANNEALING}"
        --lr_anneal_coeff "${LR_ANNEAL_COEFF}"
        --transform_action "${TRANSFORM_ACTION}"
	    --training_batch_size "${TRAINING_BATCH_SIZE}"
        --checkpoint_interval "${CHECKPOINT_INTERVAL}"
        --num_env_steps "${NUM_ENV_STEPS}"
        --total_timesteps "${TOTAL_TIMESTEPS}"
        --ego_style "${EGO_STYLE}"
        --adv_style "${ADV_STYLE}"
        --envs_per_matchup "${ENVS_PER_MATCHUP}"
        --side "${SIDE}"
        --render "${RENDER}"
        --model_file "${MODEL_FILE}"
        --async_update "${ASYNC_UPDATE}"
	    --model_arch_type "${MODEL_ARCH_TYPE}"
        --obs_type "${OBS_TYPE}"
	--vtrace_seq_len "${VTRACE_SEQ_LEN}"
	--vtrace_c_bar "${VTRACE_C_BAR}"
	--vtrace_rho_bar "${VTRACE_RHO_BAR}"
	${GAMMA:+--gamma "${GAMMA}"}
    )

    if [ "${RUN_LIVE}" = "False" ]; then
        nohup "${CMD[@]}" > "${LOGS_DIR}/ippo_worker_${i}.log" 2>&1 &
        echo "ippo instance ${i} started with PID $!"
    else
        "${CMD[@]}"
    fi
done

echo "Started ${NUM_WORKERS} ippo instances."
echo "Logs are being written to: ${LOGS_DIR}/"
