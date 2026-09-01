#!/bin/bash
# Workstation-mode BR dedicated job template.
#
# Parity with slurm_launch_files/br_dedicated_template.slurm but stripped of
# all #SBATCH/module/mail directives and pointed at a local conda install.
# The orchestrator spawns this via `bash <script>` (non-login, non-interactive)
# so we cannot rely on ~/.bashrc to put conda on PATH — we source it manually.
#
# Placeholders (do NOT use {{...}} in any comment line of this file -- the
# orchestrator's text-replace pass would expand them in-place, and a
# multi-line value like PYTHON_CMD would spill un-commented code below this
# header):
#   JOB_NAME, OUT_LOG, ERR_LOG, PYTHON_CMD       -- filled by render_template_sbatch
#   WS_WORKDIR, MAIN_TRAINING_DIR, WS_REPO_DIR   -- passed from the high-level launcher

set -euo pipefail

# -----------------------------------------------------------------------------
# Workstation paths (edit to match this machine).
# -----------------------------------------------------------------------------
WORKDIR={{WS_WORKDIR}}
MAIN_TRAINING_DIR={{MAIN_TRAINING_DIR}}
REPO_DIR={{WS_REPO_DIR}}

export WORKDIR
export MAIN_TRAINING_DIR

cd "$REPO_DIR"

# -----------------------------------------------------------------------------
# Conda activation (Popen'd bash does not source ~/.bashrc).
# -----------------------------------------------------------------------------
source /home/jw4406/anaconda3/etc/profile.d/conda.sh
conda activate fightladder

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Job-side config (parity with new_br_worker) ---
EVAL_PROT="True"
EVAL_ADV="True"
EVAL_ONLY="False"
PROJ_NAME="br_training_ws"
ANALYSIS_UPLOAD_PROJ_NAME="br_analysis"
USE_MIRROR="True"
NUM_FULL_EXPLOITERS="1"
N_ENVS="2"
EXPLOITER_SAVE_FREQ="5000"

# --- BR convergence tracker ---
BR_TRACKER_PATIENCE="300"
BR_TRACKER_TOLERANCE="1e-4"
BR_TRACKER_WINDOW_SIZE="50"
USE_BR_REWARD_STAGNATION="False"
USE_BR_ENTROPY_STAGNATION="True"
BR_USE_SLOPE_EARLY_STOP="False"
BR_SLOPE_WINDOW="20"
BR_SLOPE_TOLERANCE="5e-3"
BR_MIN_SLOPE_CHECKS="12"

# --- Entropy-window early-stop (RatingStagnationTracker) ---
ENTROPY_STOP_RATIO="0.05"
ENTROPY_WINDOW_SIZE="50"
ENTROPY_WARMUP_CHECKS="100"
ENTROPY_RATIO_ONLY="True"
# --- RLHF-style KL-to-reference locality (parity with new_br_worker; default off) ---
BR_KL_REF_COEF="0.0"
BR_KL_REF_DIRECTION="reverse"
BR_KL_REF_DROP_ENTROPY="True"

# --- CDS stagnation (unused in dedicated mode, threaded for parity) ---
USE_STAGNATION_EARLY_STOP="False"
USE_STAGNATION_VELOCITY_SIGNAL="False"
USE_STAGNATION_ENTROPY_SIGNAL="False"
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

# --- Game args ---
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
ENABLE_LOCAL_KL_PLOT="True"

# --- SLURM resource defaults (unused on workstation, kept for parity) ---
SLURM_ACCOUNT=""
PYTHON_BIN="python"

DRY_RUN="False"

# --- Per-job command (substituted by the orchestrator) ---
{{PYTHON_CMD}}
