#!/bin/bash
# Workstation-mode "fire both" launcher: starts the dedicated AND continue
# BR orchestrators as nohup'd background processes against this machine's
# single GPU.
#
# Parity with slurm_launch_files/launch_br_orchestrators.sh but:
#   - points at ws_launch_files/*.sh templates
#   - uses workstation-local repo path
#   - threads MAX_LOCAL_CONCURRENT into both orchestrators so they share
#     the GPU according to your cap instead of all launching at once
#
# The two orchestrators each have their OWN concurrency cap (no shared
# coordination across them yet). If MAX_LOCAL_CONCURRENT=1 for both, you
# can have up to 2 jobs running simultaneously on the GPU (one dedicated,
# one continue). Set LAUNCH_DEDICATED='False' or LAUNCH_CONTINUE='False'
# to run only one side.

set -euo pipefail

# -----------------------------------------------------------------------------
# Workstation config -- edit these before first run.
# -----------------------------------------------------------------------------
LAUNCH_DEDICATED='True'
LAUNCH_CONTINUE='False'
STEP_STRIDE=40000        # 0 = process all tasks; e.g. 500000 = only every 500k steps
MAX_LOCAL_CONCURRENT=1    # per-orchestrator cap on simultaneous local-bash jobs
BR_TRAINING_STEPS=250000  # total .learn() timesteps per BR job (set small for debug)

WORKDIR=/home/jw4406/
MAIN_TRAINING_DIR=codebase
REPO_DIR=/home/jw4406/codebase/FightLadder

TASK_BASE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks"
LOGS_DIR="$WORKDIR/$MAIN_TRAINING_DIR/logs"
mkdir -p "${LOGS_DIR}"

# ==========================================================================
# 1. Dedicated orchestrator (watchdog: todo/ -> slurm_processing/ -> slurm_done/)
# ==========================================================================
DEDICATED_CMD=(
    python -u "$REPO_DIR/main/br_slurm_orchestrator.py"
    --br_dedicated_sh_template "$REPO_DIR/ws_launch_files/br_dedicated_template.sh"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TASK_BASE/todo"
    --processing_dir "$TASK_BASE/slurm_processing"
    --done_dir "$TASK_BASE/slurm_done"
    --stop_file "$TASK_BASE/STOP_SLURM"
    --local_plot_dir "$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots"
    --slurm_log_dir /home/jw4406
    --step_stride "$STEP_STRIDE"
    --max_local_concurrent "$MAX_LOCAL_CONCURRENT"
    --br_training_steps "$BR_TRAINING_STEPS"
    #--dry_run True
)

if [ "$LAUNCH_DEDICATED" = 'True' ]; then
    echo "Launching dedicated BR orchestrator (workstation mode)..."
    nohup "${DEDICATED_CMD[@]}" > "${LOGS_DIR}/br_ws_orchestrator.log" 2>&1 &
    echo "  PID: $!"
    echo "  Log: ${LOGS_DIR}/br_ws_orchestrator.log"
    echo "  Stop: touch $TASK_BASE/STOP_SLURM"
    echo "  Concurrency cap: $MAX_LOCAL_CONCURRENT"
fi

# ==========================================================================
# 2. Continue orchestrator (watchdog: todo_continue/ -> slurm_processing_continue/ -> slurm_done_continue/)
# ==========================================================================
CONTINUE_CMD=(
    python -u "$REPO_DIR/main/br_continue_slurm_orchestrator.py"
    --br_continue_sh_template "$REPO_DIR/ws_launch_files/br_continue_template.sh"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TASK_BASE/todo_continue"
    --processing_dir "$TASK_BASE/slurm_processing_continue"
    --done_dir "$TASK_BASE/slurm_done_continue"
    --stop_file "$TASK_BASE/STOP_SLURM_CONTINUE"
    --local_plot_dir "$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots_continue"
    --slurm_log_dir /home/jw4406
    --step_stride "$STEP_STRIDE"
    --max_local_concurrent "$MAX_LOCAL_CONCURRENT"
    --br_training_steps "$BR_TRAINING_STEPS"
    #--dry_run True
)

echo ""
if [ "$LAUNCH_CONTINUE" = 'True' ]; then
    echo "Launching continue BR orchestrator (workstation mode)..."
    nohup "${CONTINUE_CMD[@]}" > "${LOGS_DIR}/br_ws_continue_orchestrator.log" 2>&1 &
    echo "  PID: $!"
    echo "  Log: ${LOGS_DIR}/br_ws_continue_orchestrator.log"
    echo "  Stop: touch $TASK_BASE/STOP_SLURM_CONTINUE"
    echo "  Concurrency cap: $MAX_LOCAL_CONCURRENT"
fi

echo ""
echo "To stop both: touch $TASK_BASE/STOP_SLURM $TASK_BASE/STOP_SLURM_CONTINUE"
