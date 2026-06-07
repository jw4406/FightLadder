#!/bin/bash
# Workstation-mode standalone launcher for the continue BR orchestrator.
# Parity with slurm_launch_files/launch_br_continue_training_orchestrator.sh
# but pointed at the .sh template and the workstation-local repo path, with
# the new --max_local_concurrent gate.

set -euo pipefail

# -----------------------------------------------------------------------------
# Workstation config -- edit these before first run.
# -----------------------------------------------------------------------------
WORKDIR=/home/jw4406/
MAIN_TRAINING_DIR=codebase
REPO_DIR=/home/jw4406/codebase/FightLadder
MAX_LOCAL_CONCURRENT=1   # how many local-bash jobs may run at once on this GPU
BR_TRAINING_STEPS=100000   # total .learn() timesteps per BR job (set small for debug)

LOGS_DIR="${WORKDIR}/${MAIN_TRAINING_DIR}/logs"
mkdir -p "${LOGS_DIR}"

TASK_BASE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks"
TODO_DIR="$TASK_BASE/todo_continue"
PROCESSING_DIR="$TASK_BASE/slurm_processing_continue"
DONE_DIR="$TASK_BASE/slurm_done_continue"
STOP_FILE="$TASK_BASE/STOP_SLURM_CONTINUE"
LOCAL_PLOT_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots_continue"

CMD=(python -u "$REPO_DIR/main/br_continue_slurm_orchestrator.py"
    --br_continue_sh_template "$REPO_DIR/ws_launch_files/br_continue_template.sh"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TODO_DIR"
    --processing_dir "$PROCESSING_DIR"
    --done_dir "$DONE_DIR"
    --stop_file "$STOP_FILE"
    --local_plot_dir "$LOCAL_PLOT_DIR"
    --slurm_log_dir '/home/jw4406'
    --max_local_concurrent "$MAX_LOCAL_CONCURRENT"
    --br_training_steps "$BR_TRAINING_STEPS"
    #--dry_run True
)

echo "Starting br_continue_slurm_orchestrator (workstation mode)..."
nohup "${CMD[@]}" > "${LOGS_DIR}/br_ws_continue_orchestrator.log" 2>&1 &
echo "Continue orchestrator started with PID $!"
echo "Log:  ${LOGS_DIR}/br_ws_continue_orchestrator.log"
echo "Stop: touch ${STOP_FILE}"
echo "Concurrency cap: ${MAX_LOCAL_CONCURRENT} local-bash job(s)"
