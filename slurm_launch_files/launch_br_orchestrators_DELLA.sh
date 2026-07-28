#!/bin/bash
# Launch both the dedicated and continue BR watchdog orchestrators.
#   - Dedicated: br_slurm_orchestrator.py        (todo/ -> slurm_processing/ -> slurm_done/)
#   - Continue:  br_continue_slurm_orchestrator.py (todo_continue/ -> slurm_processing_continue/ -> slurm_done_continue/)

set -euo pipefail

LAUNCH_DEDICATED='True'
LAUNCH_CONTINUE='False'
LAUNCH_LOCAL_BR_EVAL='True'  # forwarded to both orchestrators via --launch_local_br_eval
PERIODIC_EVAL_FREQ=500000   # env-steps between mid-training local_br_eval snapshots (PeriodicLocalBREvalCallback)
STEP_STRIDE=0  # 0 = process all tasks; e.g. 40000 = only every 40k steps (matches ws)
BR_TRAINING_STEPS=150000000  # total .learn() timesteps per BR job
SLURM_TIME=024:00:00  # #SBATCH --time for each BR sbatch (HH:MM:SS)
EXPLOITER_SAVE_FREQ=1000000  # env-steps between BR exploiter checkpoints

# ---- GPU packing (opt-in; dedicated orchestrator only). ----
# EXPLOITERS_PER_JOB=1 => current behavior (one exploiter per GPU, no cap).
EXPLOITERS_PER_JOB=1              # >1 co-locates N exploiters on one GPU
GPU_MEM_FRACTION=0.45             # per-process VRAM cap (used only when >1); keep N*fraction <= ~0.85
PACK_ACROSS_CHECKPOINTS='False'   # 'True' packs exploiters from DIFFERENT checkpoints (fills GPU when <N specs/ckpt)
PACK_FLUSH_TIMEOUT=300            # partial-pack timeout in seconds (cross-checkpoint mode only)
RESOURCE_SCALE=1                  # absolute cpu multiplier for PACKED jobs (cpu only; mem still scales by N); 1 = template base

WORKDIR=/scratch/gpfs/FISAC/jw4406
MAIN_TRAINING_DIR=10937422
# The repo is rsync'd into scratch alongside MAIN_TRAINING_DIR; orchestrators,
# templates, and runners all live under it. Match ws_launch_files pattern.
REPO_DIR="$HOME/FightLadder"
TASK_BASE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks"
LOGS_DIR="$WORKDIR/$MAIN_TRAINING_DIR/logs"
mkdir -p "${LOGS_DIR}"

# ==========================================================================
# 1. Dedicated orchestrator (watchdog: todo/ -> slurm_processing/ -> slurm_done/)
# ==========================================================================
DEDICATED_CMD=(
    python -u "$REPO_DIR/main/br_slurm_orchestrator.py"
    --br_dedicated_sh_template "$REPO_DIR/slurm_launch_files/br_dedicated_template_DELLA.slurm"
    --exploiters_per_job "$EXPLOITERS_PER_JOB"
    --gpu_mem_fraction "$GPU_MEM_FRACTION"
    --pack_across_checkpoints "$PACK_ACROSS_CHECKPOINTS"
    --pack_flush_timeout "$PACK_FLUSH_TIMEOUT"
    --resource_scale "$RESOURCE_SCALE"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TASK_BASE/todo"
    --processing_dir "$TASK_BASE/slurm_processing"
    --done_dir "$TASK_BASE/slurm_done"
    --stop_file "$TASK_BASE/STOP_SLURM"
    --local_plot_dir "$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots"
    --slurm_log_dir /home/jw4406/
    --step_stride "$STEP_STRIDE"
    --br_training_steps "$BR_TRAINING_STEPS"
    --slurm_time "$SLURM_TIME"
    --exploiter_save_freq "$EXPLOITER_SAVE_FREQ"
    --launch_local_br_eval "$LAUNCH_LOCAL_BR_EVAL"
    --periodic_eval_freq "$PERIODIC_EVAL_FREQ"
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
    --br_continue_sh_template "$REPO_DIR/slurm_launch_files/br_continue_template_DELLA.slurm"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TASK_BASE/todo_continue"
    --processing_dir "$TASK_BASE/slurm_processing_continue"
    --done_dir "$TASK_BASE/slurm_done_continue"
    --stop_file "$TASK_BASE/STOP_SLURM_CONTINUE"
    --local_plot_dir "$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots_continue"
    --slurm_log_dir /u/jw4406/
    --step_stride "$STEP_STRIDE"
    --br_training_steps "$BR_TRAINING_STEPS"
    --slurm_time "$SLURM_TIME"
    --exploiter_save_freq "$EXPLOITER_SAVE_FREQ"
    --launch_local_br_eval "$LAUNCH_LOCAL_BR_EVAL"
    --periodic_eval_freq "$PERIODIC_EVAL_FREQ"
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
