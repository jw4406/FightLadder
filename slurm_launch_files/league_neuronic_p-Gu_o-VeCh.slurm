#!/bin/bash
#SBATCH --job-name=league_Gu_VeCh  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jw4406@princeton.edu
echo $CUDA_VISIBLE_DEVICES
JOBID=$SLURM_JOB_ID
WORKDIR=/n/fs/magics/
mkdir -p $WORKDIR/$JOBID
cd $WORKDIR/$JOBID
echo "starting copy op"
cp -r $HOME/FightLadder/ ./
echo "copy op successful"
cd FightLadder/
#module purge
#module load anaconda3/2024.02
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
echo "fightladder environment activated"

# Number of parallel train_ma instances to run
NUM_WORKERS=1
# If True, run in the current shell (slurm captures stdout to slurm-%j.out).
RUN_LIVE="True"

# Path to train_ma.py
LEAGUE_PATH="$WORKDIR/$JOBID/FightLadder/main/train_ma.py"

# Guile vs {Vega, ChunLi} -- LEAGUE (plain league: NO --psro-league).
PLAYER=("Guile")
OPPONENTS=("Vega" "ChunLi")
SAVE_DIR="$WORKDIR/$JOBID/FightLadder/main/trained_models/tasks/todo/"
# Long-running: bounded by #SBATCH --time (72h), not by the step count.
TOTAL_STEPS="2000000000"
TRANSFORM_ACTION="True"
# Run NUM_WORKERS copies in parallel
for i in $(seq 1 ${NUM_WORKERS}); do
    echo "Starting train_ma instance ${i}..."
    CMD=(
        python "${LEAGUE_PATH}"
        --player "${PLAYER[@]}"
        --opponent-list "${OPPONENTS[@]}"
	--enable-combo
	--transform-action
	--save-dir "${SAVE_DIR}"
	--total-steps "${TOTAL_STEPS}"
    )

    if [ "${RUN_LIVE}" = "False" ]; then
        nohup "${CMD[@]}" > "${LOGS_DIR}/ippo_worker_${i}.log" 2>&1 &
        echo "train_ma instance ${i} started with PID $!"
    else
        "${CMD[@]}"
    fi
done

echo "Started ${NUM_WORKERS} train_ma instances."
