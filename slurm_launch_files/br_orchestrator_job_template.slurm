#!/bin/bash
# Dedicated BR watchdog run AS A SLURM JOB (not a login-node process).
#
# br_slurm_orchestrator.py is a stop-file-bounded polling loop
# (`while not exists(STOP_SLURM)`) that submits exploiter sbatch jobs as
# checkpoints stream into TASK_BASE/todo. Running it here — on a compute
# node — keeps long-running work off the login node. It ends when
# TASK_BASE/STOP_SLURM is touched, or when this job's walltime expires.
#
# lr_ratio_sweep.py substitutes the marked VAR="..." lines per config. Paths
# derive from $WORKDIR/$MAIN_TRAINING_DIR (the same per-run tree the training
# job populates), so tasks + br_rewards + br_models all resolve consistently.
# CPU-only: the orchestrator does no GPU work; exploiter jobs request their own.
#SBATCH --job-name=brorch          # per-config tag substituted by the driver
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=096:00:00           # >= the training run so late checkpoints are caught
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jw4406@princeton.edu
module purge
# >>> conda initialize >>>
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
echo "fightladder environment activated"

# The orchestrator runs from a stable checkout (it only submits sbatch jobs;
# each exploiter job does its own repo handling via the dedicated template).
REPO_DIR="$HOME/FightLadder"
cd "$REPO_DIR"

# ---- per-config values (substituted by lr_ratio_sweep.py) ----
WORKDIR="/scratch/gpfs/FISAC/jw4406/"
MAIN_TRAINING_DIR="lr_sweep/CONFIG_TAG"
BR_DEDICATED_TEMPLATE="$REPO_DIR/slurm_launch_files/br_dedicated_template_NEURONIC.slurm"
SLURM_LOG_DIR="/home/jw4406/"
STEP_STRIDE="0"
BR_TRAINING_STEPS="99999"
EXPLOITER_SAVE_FREQ="1000"
PERIODIC_EVAL_FREQ="500000"
BR_SLURM_TIME="000:12:00"
ORCH_DRY_RUN="False"
EXPLOITERS_PER_JOB="1"
GPU_MEM_FRACTION="0.45"

# Derived — the per-config tree the training job populates.
export WORKDIR MAIN_TRAINING_DIR
TASK_BASE="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks"
LOCAL_PLOT_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/logs/local_entropy_plots"
mkdir -p "$LOCAL_PLOT_DIR"

# Don't start polling until the training job has created the task dir. This
# covers the brief window after training begins while it copies the repo /
# writes its first checkpoint, complementing the sbatch --dependency=after:
# gate (which the driver adds in --phase both). Bounded so a dead training run
# doesn't idle to walltime.
_WAIT=0
until [ -d "$TASK_BASE/todo" ] || [ "$_WAIT" -ge 1800 ]; do
    sleep 15
    _WAIT=$((_WAIT + 15))
done

python -u "$REPO_DIR/main/br_slurm_orchestrator.py" \
    --br_dedicated_sh_template "$BR_DEDICATED_TEMPLATE" \
    --workdir "$WORKDIR" \
    --main_training_dir "$MAIN_TRAINING_DIR" \
    --todo_dir "$TASK_BASE/todo" \
    --processing_dir "$TASK_BASE/slurm_processing" \
    --done_dir "$TASK_BASE/slurm_done" \
    --stop_file "$TASK_BASE/STOP_SLURM" \
    --local_plot_dir "$LOCAL_PLOT_DIR" \
    --slurm_log_dir "$SLURM_LOG_DIR" \
    --step_stride "$STEP_STRIDE" \
    --br_training_steps "$BR_TRAINING_STEPS" \
    --slurm_time "$BR_SLURM_TIME" \
    --exploiter_save_freq "$EXPLOITER_SAVE_FREQ" \
    --launch_local_br_eval True \
    --periodic_eval_freq "$PERIODIC_EVAL_FREQ" \
    --exploiters_per_job "$EXPLOITERS_PER_JOB" \
    --gpu_mem_fraction "$GPU_MEM_FRACTION" \
    --dry_run "$ORCH_DRY_RUN"
