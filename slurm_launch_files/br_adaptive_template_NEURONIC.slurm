#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{SBATCH_CPUS}}
#SBATCH --mem={{SBATCH_MEM}}
#SBATCH --gres={{SBATCH_GRES}}
#SBATCH --time={{SBATCH_TIME}}
#SBATCH --output={{OUT_LOG}}
#SBATCH --error={{ERR_LOG}}
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jw4406@princeton.edu

# ADAPTIVE controller job (neuronic): ONE sbatch per checkpoint, SIZED DOWN so
# SLURM packs several onto each 8-GPU / 52-core node. This node slice runs the
# WHOLE portfolio (controller + its br_single workers). The controller self-tunes
# n_envs to this cgroup (--cpus-per-task) via len(sched_getaffinity), and its
# workers inherit the SLURM-set CUDA_VISIBLE_DEVICES (from --gres), so nothing
# reaches outside the allocation. Resource sizing (cpus/gres/mem/time) is filled
# by the orchestrator from --slurm_cpus_per_task/--slurm_gres/--slurm_mem/
# --slurm_time -- set them in launch_br_orchestrators_NEURONIC.sh.

set -euo pipefail

# Filled by render_template_sbatch (orchestrator-side). The launcher is the single
# source of truth. (Do not write the literal PYTHON_CMD placeholder token in any
# comment -- render does a blind text-replace and would inject the CMD block here.)
WORKDIR={{WS_WORKDIR}}
MAIN_TRAINING_DIR={{MAIN_TRAINING_DIR}}
REPO_DIR={{WS_REPO_DIR}}

export MAIN_TRAINING_DIR

cd "$REPO_DIR"

module purge
module load anaconda3/2024.2
# >>> conda initialize >>>
__conda_setup="$('/usr/licensed/anaconda3/2024.2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/licensed/anaconda3/2024.2/etc/profile.d/conda.sh" ]; then
        . "/usr/licensed/anaconda3/2024.2/etc/profile.d/conda.sh"
    else
        export PATH="/usr/licensed/anaconda3/2024.2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate fightladder_della

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Adaptive controller command (substituted by the orchestrator): a block of
# `export ADAPT_*=...` lines (incl. ADAPT_N_ENVS=auto for cgroup self-tuning and
# the per-checkpoint WORKDIR) followed by `python -u main/adaptive_exploiter.py`.
# NOTE: the block sets its own WORKDIR (per-checkpoint adaptive_out subtree), so it
# intentionally overrides the WORKDIR line above. ---
{{PYTHON_CMD}}
