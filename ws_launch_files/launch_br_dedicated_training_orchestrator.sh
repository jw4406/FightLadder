#!/bin/bash
# Workstation-mode standalone launcher for the dedicated BR orchestrator.
# Parity with slurm_launch_files/launch_br_dedicated_training_orchestrator.sh
# but pointed at the .sh template and the workstation-local repo path, with
# the new --max_local_concurrent gate.

set -euo pipefail

# -----------------------------------------------------------------------------
# Workstation config -- edit these before first run.
# -----------------------------------------------------------------------------
WORKDIR=/home/jw4406
MAIN_TRAINING_DIR=codebase
REPO_DIR=/home/jw4406/FightLadder
MAX_LOCAL_CONCURRENT=1   # how many local-bash jobs may run at once on this GPU (legacy per-spec path)
BR_TRAINING_STEPS=100000   # total .learn() timesteps per BR job (set small for debug)

# --- ADAPTIVE exploiter mode (default ON) ---
# Per checkpoint, run adaptive_exploiter.py (portfolio + successive halving + kill
# bar + escalation ladder) instead of one blind BR per (matchup, side, replicate).
# Concurrency + envs/worker are auto-tuned to THIS machine at startup and frozen
# for the run; search budget (MAX_CONFIGS/CAP_STEPS in adaptive_exploiter.py) stays
# fixed so exploitability curve points stay comparable.
USE_ADAPTIVE=True                 # False = legacy per-spec dedicated path
MAX_CONCURRENT_ADAPTIVES=auto     # 'auto' = one per visible GPU (GPU-mem gated); or an integer
ADAPTIVE_N_ENVS=auto              # 'auto' = fill cores given the concurrency, clamped [2,8]; or pin an integer
ADAPTIVE_RESERVE_CORES=2          # cores held back for controller(s)+system
ADAPTIVE_OUT_DIR=""               # default <WORKDIR>/<MAIN_TRAINING_DIR>/adaptive_out
ADAPTIVE_LEAGUE_STATES=""         # CSV state override for non-standard league members (PSRO *_historical_*)

LOGS_DIR="${WORKDIR}/${MAIN_TRAINING_DIR}/logs"
mkdir -p "${LOGS_DIR}"

TASK_BASE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks"
TODO_DIR="$TASK_BASE/todo"
PROCESSING_DIR="$TASK_BASE/slurm_processing"
DONE_DIR="$TASK_BASE/slurm_done"
STOP_FILE="$TASK_BASE/STOP_SLURM"
LOCAL_PLOT_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots"

CMD=(python -u "$REPO_DIR/main/br_slurm_orchestrator.py"
    --br_dedicated_sh_template "$REPO_DIR/ws_launch_files/br_dedicated_template.sh"
    --main_training_dir "$MAIN_TRAINING_DIR"
    --workdir "$WORKDIR"
    --todo_dir "$TODO_DIR"
    --processing_dir "$PROCESSING_DIR"
    --done_dir "$DONE_DIR"
    --stop_file "$STOP_FILE"
    --local_plot_dir "$LOCAL_PLOT_DIR"
    --slurm_log_dir /home/jw4406
    --max_local_concurrent "$MAX_LOCAL_CONCURRENT"
    --br_training_steps "$BR_TRAINING_STEPS"
    --use_adaptive "$USE_ADAPTIVE"
    --max_concurrent_adaptives "$MAX_CONCURRENT_ADAPTIVES"
    --adaptive_n_envs "$ADAPTIVE_N_ENVS"
    --adaptive_reserve_cores "$ADAPTIVE_RESERVE_CORES"
    --adaptive_out_dir "$ADAPTIVE_OUT_DIR"
    --adaptive_league_states "$ADAPTIVE_LEAGUE_STATES"
    #--dry_run True
)

echo "Starting br_slurm_orchestrator (workstation mode)..."
nohup "${CMD[@]}" > "${LOGS_DIR}/br_ws_orchestrator.log" 2>&1 &
echo "Dedicated orchestrator started with PID $!"
echo "Log:  ${LOGS_DIR}/br_ws_orchestrator.log"
echo "Stop: touch ${STOP_FILE}"
echo "Concurrency cap: ${MAX_LOCAL_CONCURRENT} local-bash job(s)"
