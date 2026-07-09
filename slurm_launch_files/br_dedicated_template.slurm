#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=024:00:00
#SBATCH --output={{OUT_LOG}}
#SBATCH --error={{ERR_LOG}}
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jw4406@princeton.edu

set -euo pipefail

# Placeholders WS_WORKDIR / MAIN_TRAINING_DIR / WS_REPO_DIR are filled in by
# render_template_sbatch (orchestrator-side); see ws_launch_files/*.sh for the
# same pattern. The launcher (launch_br_*.sh) is the single source of truth.
WORKDIR={{WS_WORKDIR}}
MAIN_TRAINING_DIR={{MAIN_TRAINING_DIR}}
REPO_DIR={{WS_REPO_DIR}}

export WORKDIR
export MAIN_TRAINING_DIR

cd "$REPO_DIR"

module purge
module load anaconda3/2024.02
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3/2024.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/2024.02/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate fightladder

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Job-side config (parity with new_br_worker) ---
# These KEY="value" lines are parsed by load_template_config in br_slurm_common.py
# and applied to argparse defaults (CLI flags still override). NOT shell vars:
# do not rely on $KEY expansion below; the only path to the runner is the
# python command block at the bottom, built from the orchestrator's parsed args.
# (Do not write the literal placeholder token in this comment — render_template_sbatch
# does a blind text-replace and would inject the CMD block here.)
EVAL_PROT="True"
EVAL_ADV="True"
EVAL_ONLY="False"
PROJ_NAME="br_training_slurm"
ANALYSIS_UPLOAD_PROJ_NAME="br_analysis"
USE_MIRROR="False"
NUM_FULL_EXPLOITERS="1"
N_ENVS="2"
EXPLOITER_SAVE_FREQ="1000000"

# (BR Exploiter tracker is record-only after the gutting; its ema_beta and
# eval_games are hardcoded in algorithms.py, so there are no tunable
# tracker knobs for the dedicated path. All BR_TRACKER_*, BR_SLOPE_*,
# ENTROPY_*, STAGNATION_*, USE_BR_*_STAGNATION, USE_STAGNATION_* lines
# were removed -- they were all inert.)

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

# --- SLURM resource defaults ---
SLURM_ACCOUNT=""
PYTHON_BIN="python"

DRY_RUN="False"

# --- Per-job command (substituted by the orchestrator) ---
{{PYTHON_CMD}}
