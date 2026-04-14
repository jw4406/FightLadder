#!/bin/bash

# Number of parallel ippo instances to run
NUM_WORKERS=1
# If True, run each worker detached with nohup and log redirection.
# If False, run without nohup in the current shell.
RUN_LIVE="True"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to ippo.py
IPPO_PATH="${SCRIPT_DIR}/main/ippo.py"

# Arguments from launch.json (Python Debugger: ippo.py)
PLAYER=("Guile")
OPPONENTS=("Sagat" "Ryu")
NUM_ENV_TO_LOAD="1"
ENV_BATCH_SIZE="24"
C_LR="1e-5"
D_LR="2e-4"
V_LR="4e-5"
NUM_PERTURBS="10"
USE_MIRROR="False"
# LOAD_PATH=""  # Optional: set this to pass --load_path
# TRAINING_STYLE=""  # Optional: set this to pass --training_style
# CONTINUE_TRAINING=""  # Optional: set this to pass --continue_training
# LEFT_MODEL_FILE=""  # Optional: set this to pass --left-model-file
# RIGHT_MODEL_FILE=""  # Optional: set this to pass --right-model-file
SAVE_DIR="/home/jw4406/codebase/FightLadder/main/trained_models/tasks/todo/"
USE_LR_ANNEALING="True"
LR_ANNEAL_COEFF=".995"
CHECKPOINT_INTERVAL="100000"
TRAINING_BATCH_SIZE="512"
NUM_ENV_STEPS="512"
EGO_STYLE="learning"
ADV_STYLE="learning"
ENVS_PER_MATCHUP="2"
SIDE="both"
RENDER="False"
MODEL_FILE=""
ASYNC_UPDATE="False"
MODEL_ARCH_TYPE="spar"
USE_ELO_STAGNATION="False"
USE_ELO_EARLY_STOP="True"
ELO_STAGNATION_PATIENCE="20000"
ELO_STAGNATION_TOLERANCE="1e-4"
ELO_STAGNATION_REL_TOLERANCE="0.05"
ELO_STAGNATION_EMA_BETA="0.99"
ELO_STAGNATION_EPS="1e-8"
ELO_ROSTER_EPOCH_SIZE="0"
ELO_ENTROPY_WEIGHT="1.0"
ELO_LR_FACTOR="0.999"
ELO_LR_PATIENCE="150"
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
        --save_dir "${SAVE_DIR}"
        --use_lr_annealing "${USE_LR_ANNEALING}"
        --lr_anneal_coeff "${LR_ANNEAL_COEFF}"
	    --training_batch_size "${TRAINING_BATCH_SIZE}"
        --checkpoint_interval "${CHECKPOINT_INTERVAL}"
        --num_env_steps "${NUM_ENV_STEPS}"
        --ego_style "${EGO_STYLE}"
        --adv_style "${ADV_STYLE}"
        --envs_per_matchup "${ENVS_PER_MATCHUP}"
        --side "${SIDE}"
        --render "${RENDER}"
        --model_file "${MODEL_FILE}"
        --async_update "${ASYNC_UPDATE}"
	    --model_arch_type "${MODEL_ARCH_TYPE}"
	    --use_elo_stagnation "${USE_ELO_STAGNATION}"
        --use_elo_early_stop "${USE_ELO_EARLY_STOP}"
        --elo_stagnation_patience "${ELO_STAGNATION_PATIENCE}"
        --elo_stagnation_tolerance "${ELO_STAGNATION_TOLERANCE}"
        --elo_stagnation_rel_tolerance "${ELO_STAGNATION_REL_TOLERANCE}"
        --elo_stagnation_ema_beta "${ELO_STAGNATION_EMA_BETA}"
        --elo_stagnation_eps "${ELO_STAGNATION_EPS}"
        --elo_roster_epoch_size "${ELO_ROSTER_EPOCH_SIZE}"
        --elo_entropy_weight "${ELO_ENTROPY_WEIGHT}"
        --elo_lr_factor "${ELO_LR_FACTOR}"
        --elo_lr_patience "${ELO_LR_PATIENCE}"
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
