#!/bin/bash

WORKDIR=/scratch/gpfs/FISAC/jw4406
MAIN_TRAINING_DIR=7500763
BR_TRAINING_STEPS=10000000   # total .learn() timesteps per BR job

# --- ADAPTIVE exploiter mode (default ON; see ws_launch_files for the full rationale) ---
# NOTE: adaptive mode runs LOCAL controllers on the node the orchestrator runs on;
# it does NOT submit per-spec sbatch jobs. On a real multi-node SLURM allocation set
# USE_ADAPTIVE=False to keep the sbatch fan-out. Concurrency+envs auto-tune to the node.
USE_ADAPTIVE=True
MAX_CONCURRENT_ADAPTIVES=auto     # 'auto' = one per visible GPU (GPU-mem gated); or an integer
ADAPTIVE_N_ENVS=auto              # 'auto' = fill cores given the concurrency, clamped [2,8]; or pin an integer
ADAPTIVE_RESERVE_CORES=2
ADAPTIVE_OUT_DIR=""               # default <WORKDIR>/<MAIN_TRAINING_DIR>/adaptive_out
ADAPTIVE_LEAGUE_STATES=""         # CSV state override for non-standard league members (PSRO *_historical_*)
# The repo is rsync'd into scratch alongside MAIN_TRAINING_DIR; orchestrators,
# templates, and runners all live under it. Matches ws_launch_files pattern.
REPO_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder"
LOGS_DIR="${WORKDIR}/${MAIN_TRAINING_DIR}/logs"
mkdir -p "${LOGS_DIR}"

TODO_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/todo"
PROCESSING_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/slurm_processing"
DONE_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/slurm_done"
STOP_FILE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/STOP_SLURM"
LOCAL_PLOT_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots"

CMD=(python -u "$REPO_DIR/main/br_slurm_orchestrator.py"
	--br_dedicated_sh_template "$REPO_DIR/slurm_launch_files/br_dedicated_template.slurm"
	--main_training_dir "$MAIN_TRAINING_DIR"
	--workdir "$WORKDIR"
	--todo_dir "$TODO_DIR"
	--processing_dir "$PROCESSING_DIR"
	--done_dir "$DONE_DIR"
	--stop_file "$STOP_FILE"
	--local_plot_dir "$LOCAL_PLOT_DIR"
	--slurm_log_dir /home/jw4406
	--br_training_steps "$BR_TRAINING_STEPS"
	--use_adaptive "$USE_ADAPTIVE"
	--max_concurrent_adaptives "$MAX_CONCURRENT_ADAPTIVES"
	--adaptive_n_envs "$ADAPTIVE_N_ENVS"
	--adaptive_reserve_cores "$ADAPTIVE_RESERVE_CORES"
	--adaptive_out_dir "$ADAPTIVE_OUT_DIR"
	--adaptive_league_states "$ADAPTIVE_LEAGUE_STATES"
	#--dry_run True
)

echo "Starting br_slurm_orchestrator..."
nohup "${CMD[@]}" > "${LOGS_DIR}/br_slurm_orchestrator.log" 2>&1 &
echo "Dedicated orchestrator started with PID $!"
echo "Log: ${LOGS_DIR}/br_slurm_orchestrator.log"
echo "Touch ${STOP_FILE} to stop the watchdog (in-flight SLURM jobs are unaffected)."
