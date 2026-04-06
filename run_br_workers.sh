#!/bin/bash

# Number of parallel new_br_worker instances to run
NUM_WORKERS=1
# If True, run each worker detached with nohup and log redirection.
# If False, run without nohup in the current shell.
RUN_LIVE="True"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to new_br_worker.py
BR_WORKER_PATH="${SCRIPT_DIR}/main/new_br_worker.py"

# Arguments from launch.json (Python Debugger: new_br_worker.py)
EVAL_PROT="True"
EVAL_ADV="False"
EVAL_ONLY="False"
PROJ_NAME="br_training"
ANALYSIS_UPLOAD_PROJ_NAME="br_analysis"
LOAD_BR="False"
WHICH_ENV="my_pendulum"
IS_LEAGUE="False"
USE_MIRROR="False"
NUM_FULL_EXPLOITERS="1"
NUM_CONTINUE_EXPLOITERS="1"
DEBUG="True"
N_ENVS="2"
# TASK_DIR=""  # Optional: set this to pass --task_dir
DEDICATED_EXPLOITER="True"
CONTINUE_EXPLOITERS="False"
EXPLOITER_SAVE_FREQ="10000"
RESET="round"
SIDE="both"
RENDER="False"
ENABLE_COMBO="True"
NULL_COMBO="False"
TRANSFORM_ACTION="False"
SEED="0"
LAUNCH_LOCAL_BR_EVAL="False"
# Torch device: cpu, cuda, cuda:0, etc. (must match launch.json / worker --device)
DEVICE="cuda"
if [ "${DEVICE}" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

# Create logs directory if it doesn't exist
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
#conda activate mujoco_sb3_parallel
# Run NUM_WORKERS copies of new_br_worker in parallel
for i in $(seq 1 ${NUM_WORKERS}); do
    echo "Starting new_br_worker instance ${i}..."
    CMD=(
        python "${BR_WORKER_PATH}"
        --eval_prot "${EVAL_PROT}"
        --eval_adv "${EVAL_ADV}"
        --eval_only "${EVAL_ONLY}"
        --proj_name "${PROJ_NAME}"
        --analysis_upload_proj_name "${ANALYSIS_UPLOAD_PROJ_NAME}"
        --load_br "${LOAD_BR}"
        --which_env "${WHICH_ENV}"
        --is_league "${IS_LEAGUE}"
        --use_mirror "${USE_MIRROR}"
        --num_full_exploiters "${NUM_FULL_EXPLOITERS}"
        --num_continue_exploiters "${NUM_CONTINUE_EXPLOITERS}"
        --DEBUG "${DEBUG}"
        --n_envs "${N_ENVS}"
        --dedicated_exploiter "${DEDICATED_EXPLOITER}"
        --continue_exploiters "${CONTINUE_EXPLOITERS}"
        --exploiter_save_freq "${EXPLOITER_SAVE_FREQ}"
        --reset "${RESET}"
        --side "${SIDE}"
        --render "${RENDER}"
        --enable_combo "${ENABLE_COMBO}"
        --null_combo "${NULL_COMBO}"
        --transform_action "${TRANSFORM_ACTION}"
        --seed "${SEED}"
        --device "${DEVICE}"
        --launch_local_br_eval "${LAUNCH_LOCAL_BR_EVAL}"
    )

    if [ "${RUN_LIVE}" = "False" ]; then
        nohup "${CMD[@]}" > "${LOGS_DIR}/br_worker_${i}.log" 2>&1 &

        echo "new_br_worker instance ${i} started with PID $!"
    else
        "${CMD[@]}"
    fi
done

echo "Started ${NUM_WORKERS} new_br_worker instances."
echo "Logs are being written to: ${LOGS_DIR}/"
echo "To stop all workers, create a STOP file in the task directory or kill the processes."
