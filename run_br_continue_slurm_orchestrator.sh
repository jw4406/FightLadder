#!/bin/bash
# Launch the per-matchup continue BR SLURM orchestrator as a long-running
# watchdog. Mirrors run_br_slurm_orchestrator.sh's variable layout so the
# two are easy to diff. CDS-only; league tasks are skipped with a clear
# message inside the orchestrator.

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to br_continue_slurm_orchestrator.py
ORCH_PATH="${SCRIPT_DIR}/main/br_continue_slurm_orchestrator.py"

# --- Watchdog / paths ---
# Same TODO_DIR as the dedicated orchestrator is OK — both watchers race
# to atomically rename each .task; whichever wins handles it. By
# convention you typically run only one orchestrator at a time. The
# processing/done dirs are distinct so the two never collide on
# in-flight state even when both are running.
TODO_DIR="/home/jw4406/codebase/FightLadder/main/trained_models/tasks/todo"
PROCESSING_DIR="/home/jw4406/codebase/FightLadder/main/trained_models/tasks/slurm_processing_continue"
DONE_DIR="/home/jw4406/codebase/FightLadder/main/trained_models/tasks/slurm_done_continue"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"
STOP_FILE="/home/jw4406/codebase/FightLadder/main/trained_models/tasks/STOP_SLURM_CONTINUE"

# --- Job-side parity with new_br_worker (drives shared_config_json) ---
EVAL_PROT="True"
EVAL_ADV="True"
EVAL_ONLY="False"
PROJ_NAME="br_training_slurm_continue"
ANALYSIS_UPLOAD_PROJ_NAME="br_analysis"
USE_MIRROR="False"
NUM_CONTINUE_EXPLOITERS="1"
N_ENVS="2"
EXPLOITER_SAVE_FREQ="100000"

# BR convergence tracker (used inside Exploiter; harmless in continue mode
# but threaded for parity with dedicated orchestrator).
BR_TRACKER_PATIENCE="300"
BR_TRACKER_TOLERANCE="1e-4"
BR_TRACKER_WINDOW_SIZE="50"
USE_BR_REWARD_STAGNATION="False"
USE_BR_ENTROPY_STAGNATION="True"
BR_USE_SLOPE_EARLY_STOP="False"
BR_SLOPE_WINDOW="20"
BR_SLOPE_TOLERANCE="5e-3"
BR_MIN_SLOPE_CHECKS="12"

# Entropy-window early-stop knobs (RatingStagnationTracker).
ENTROPY_STOP_RATIO="0.05"
ENTROPY_WINDOW_SIZE="50"
ENTROPY_WARMUP_CHECKS="100"

# CDS continue-mode stagnation knobs (active in this orchestrator).
USE_STAGNATION_EARLY_STOP="True"
USE_STAGNATION_VELOCITY_SIGNAL="False"
USE_STAGNATION_ENTROPY_SIGNAL="True"
STAGNATION_PATIENCE="2000"
STAGNATION_TOLERANCE="1e-4"
STAGNATION_REL_TOLERANCE="0.05"
STAGNATION_EMA_BETA="0.99"
STAGNATION_EPS="1e-8"
STAGNATION_EVAL_GAMES="20"
ENTROPY_STAGNATION_WEIGHT="1.0"
STAGNATION_LR_FACTOR="0.999"
STAGNATION_LR_PATIENCE="1000"
STAGNATION_USE_SLOPE_EARLY_STOP="False"
STAGNATION_SLOPE_WINDOW="20"
STAGNATION_SLOPE_TOLERANCE="5e-3"
STAGNATION_MIN_SLOPE_CHECKS="12"

# Game args
RESET="round"
SIDE="both"
RENDER="False"
ENABLE_COMBO="True"
NULL_COMBO="False"
TRANSFORM_ACTION="True"
SEED="0"
DEVICE="cuda"
LAUNCH_LOCAL_BR_EVAL="True"
USE_WANDB="False"

# --- SLURM resource defaults (override on the cluster as needed) ---
SLURM_PARTITION="gpu"
SLURM_TIME="12:00:00"
SLURM_MEM="16G"
SLURM_GRES="gpu:1"
SLURM_CPUS_PER_TASK="4"
SLURM_ACCOUNT=""
PYTHON_BIN="python"
ENV_SETUP=""

DRY_RUN="False"

# --- Logs ---
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

CMD=(
    "${PYTHON_BIN}" -u "${ORCH_PATH}"
    --todo_dir "${TODO_DIR}"
    --processing_dir "${PROCESSING_DIR}"
    --done_dir "${DONE_DIR}"
    --slurm_log_dir "${SLURM_LOG_DIR}"
    --stop_file "${STOP_FILE}"
    --eval_prot "${EVAL_PROT}"
    --eval_adv "${EVAL_ADV}"
    --eval_only "${EVAL_ONLY}"
    --proj_name "${PROJ_NAME}"
    --analysis_upload_proj_name "${ANALYSIS_UPLOAD_PROJ_NAME}"
    --use_mirror "${USE_MIRROR}"
    --num_continue_exploiters "${NUM_CONTINUE_EXPLOITERS}"
    --n_envs "${N_ENVS}"
    --exploiter_save_freq "${EXPLOITER_SAVE_FREQ}"
    --br_tracker_patience "${BR_TRACKER_PATIENCE}"
    --br_tracker_tolerance "${BR_TRACKER_TOLERANCE}"
    --br_tracker_window_size "${BR_TRACKER_WINDOW_SIZE}"
    --use_br_reward_stagnation "${USE_BR_REWARD_STAGNATION}"
    --use_br_entropy_stagnation "${USE_BR_ENTROPY_STAGNATION}"
    --br_use_slope_early_stop "${BR_USE_SLOPE_EARLY_STOP}"
    --br_slope_window "${BR_SLOPE_WINDOW}"
    --br_slope_tolerance "${BR_SLOPE_TOLERANCE}"
    --br_min_slope_checks "${BR_MIN_SLOPE_CHECKS}"
    --entropy_stop_ratio "${ENTROPY_STOP_RATIO}"
    --entropy_window_size "${ENTROPY_WINDOW_SIZE}"
    --entropy_warmup_checks "${ENTROPY_WARMUP_CHECKS}"
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
    --stagnation_use_slope_early_stop "${STAGNATION_USE_SLOPE_EARLY_STOP}"
    --stagnation_slope_window "${STAGNATION_SLOPE_WINDOW}"
    --stagnation_slope_tolerance "${STAGNATION_SLOPE_TOLERANCE}"
    --stagnation_min_slope_checks "${STAGNATION_MIN_SLOPE_CHECKS}"
    --reset "${RESET}"
    --side "${SIDE}"
    --render "${RENDER}"
    --enable_combo "${ENABLE_COMBO}"
    --null_combo "${NULL_COMBO}"
    --transform_action "${TRANSFORM_ACTION}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --launch_local_br_eval "${LAUNCH_LOCAL_BR_EVAL}"
    --use_wandb "${USE_WANDB}"
    --slurm_partition "${SLURM_PARTITION}"
    --slurm_time "${SLURM_TIME}"
    --slurm_mem "${SLURM_MEM}"
    --slurm_gres "${SLURM_GRES}"
    --slurm_cpus_per_task "${SLURM_CPUS_PER_TASK}"
    --python_bin "${PYTHON_BIN}"
    --dry_run "${DRY_RUN}"
)
if [ -n "${SLURM_ACCOUNT}" ]; then
    CMD+=(--slurm_account "${SLURM_ACCOUNT}")
fi
if [ -n "${ENV_SETUP}" ]; then
    CMD+=(--env_setup "${ENV_SETUP}")
fi

echo "Starting br_continue_slurm_orchestrator..."
nohup "${CMD[@]}" > "${LOGS_DIR}/br_continue_slurm_orchestrator.log" 2>&1 &
echo "Continue orchestrator started with PID $!"
echo "Log: ${LOGS_DIR}/br_continue_slurm_orchestrator.log"
echo "Touch ${STOP_FILE} to stop the watchdog (in-flight SLURM jobs are unaffected)."
