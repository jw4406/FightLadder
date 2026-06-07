#!/bin/bash

WORKDIR=/scratch/gpfs/FISAC/jw4406
MAIN_TRAINING_DIR=7500763
BR_TRAINING_STEPS=10000000   # total .learn() timesteps per BR job
LOGS_DIR="${WORKDIR}/${MAIN_TRAINING_DIR}/logs"
mkdir -p "${LOGS_DIR}"

TODO_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/todo"
PROCESSING_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/slurm_processing"
DONE_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/slurm_done"
STOP_FILE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/STOP_SLURM"
LOCAL_PLOT_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots"

CMD=(python -u /home/jw4406/FightLadder/main/br_slurm_orchestrator.py
	--br_dedicated_sh_template /home/jw4406/FightLadder/slurm_launch_files/br_dedicated_template.slurm
	--main_training_dir "$MAIN_TRAINING_DIR"
	--workdir "$WORKDIR"
	--todo_dir "$TODO_DIR"
	--processing_dir "$PROCESSING_DIR"
	--done_dir "$DONE_DIR"
	--stop_file "$STOP_FILE"
	--local_plot_dir "$LOCAL_PLOT_DIR"
	--slurm_log_dir /home/jw4406
	--br_training_steps "$BR_TRAINING_STEPS"
	#--dry_run True
)

echo "Starting br_slurm_orchestrator..."
nohup "${CMD[@]}" > "${LOGS_DIR}/br_slurm_orchestrator.log" 2>&1 &
echo "Dedicated orchestrator started with PID $!"
echo "Log: ${LOGS_DIR}/br_slurm_orchestrator.log"
echo "Touch ${STOP_FILE} to stop the watchdog (in-flight SLURM jobs are unaffected)."
