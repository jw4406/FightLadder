#!/bin/bash
# Launch both the dedicated and continue BR watchdog orchestrators.
#   - Dedicated: br_slurm_orchestrator.py        (todo/ -> slurm_processing/ -> slurm_done/)
#   - Continue:  br_continue_slurm_orchestrator.py (todo_continue/ -> slurm_processing_continue/ -> slurm_done_continue/)

set -euo pipefail

LAUNCH_DEDICATED='True'
LAUNCH_CONTINUE='True'
STEP_STRIDE=40000  # 0 = process all tasks; e.g. 40000 = only every 40k steps (matches ws)
BR_TRAINING_STEPS=10000000  # total .learn() timesteps per BR job

WORKDIR=/scratch/gpfs/FISAC/jw4406
MAIN_TRAINING_DIR=7500763
# The repo is rsync'd into scratch alongside MAIN_TRAINING_DIR; orchestrators,
# templates, and runners all live under it. Match ws_launch_files pattern.
REPO_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder"
TASK_BASE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks"
LOGS_DIR="$WORKDIR/$MAIN_TRAINING_DIR/logs"
mkdir -p "${LOGS_DIR}"

# ==========================================================================
# 1. Dedicated orchestrator (watchdog: todo/ -> slurm_processing/ -> slurm_done/)
# ==========================================================================
DEDICATED_CMD=(
    python -u "$REPO_DIR/main/br_slurm_orchestrator.py"
    --br_dedicated_sh_template "$REPO_DIR/slurm_launch_files/br_dedicated_template.slurm"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TASK_BASE/todo"
    --processing_dir "$TASK_BASE/slurm_processing"
    --done_dir "$TASK_BASE/slurm_done"
    --stop_file "$TASK_BASE/STOP_SLURM"
    --local_plot_dir "$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots"
    --slurm_log_dir /home/jw4406
    --step_stride "$STEP_STRIDE"
    --br_training_steps "$BR_TRAINING_STEPS"
    #--dry_run True
)

if [ "$LAUNCH_DEDICATED" = 'True' ]; then
    echo "Launching dedicated BR orchestrator..."
    nohup "${DEDICATED_CMD[@]}" > "${LOGS_DIR}/br_slurm_orchestrator.log" 2>&1 &
    echo "  PID: $!"
    echo "  Log: ${LOGS_DIR}/br_slurm_orchestrator.log"
    echo "  Stop: touch $TASK_BASE/STOP_SLURM"
fi

# ==========================================================================
# 2. Continue orchestrator (watchdog: todo_continue/ -> slurm_processing_continue/ -> slurm_done_continue/)
# ==========================================================================
CONTINUE_CMD=(
    python -u "$REPO_DIR/main/br_continue_slurm_orchestrator.py"
    --br_continue_sh_template "$REPO_DIR/slurm_launch_files/br_continue_template.slurm"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TASK_BASE/todo_continue"
    --processing_dir "$TASK_BASE/slurm_processing_continue"
    --done_dir "$TASK_BASE/slurm_done_continue"
    --stop_file "$TASK_BASE/STOP_SLURM_CONTINUE"
    --local_plot_dir "$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots_continue"
    --slurm_log_dir /home/jw4406
    --step_stride "$STEP_STRIDE"
    --br_training_steps "$BR_TRAINING_STEPS"
    #--dry_run True
)

echo ""
if [ "$LAUNCH_CONTINUE" = 'True' ]; then
    echo "Launching continue BR orchestrator..."
    nohup "${CONTINUE_CMD[@]}" > "${LOGS_DIR}/br_continue_slurm_orchestrator.log" 2>&1 &
    echo "  PID: $!"
    echo "  Log: ${LOGS_DIR}/br_continue_slurm_orchestrator.log"
    echo "  Stop: touch $TASK_BASE/STOP_SLURM_CONTINUE"
fi

echo ""
echo "To stop both: touch $TASK_BASE/STOP_SLURM $TASK_BASE/STOP_SLURM_CONTINUE"
