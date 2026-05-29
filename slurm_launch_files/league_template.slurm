#!/bin/bash
#SBATCH --job-name=spar_Ry_GuChZaSaRyEHBlDhKeMBBaVe  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jw4406@princeton.edu
echo $CUDA_VISIBLE_DEVICES
JOBID=$SLURM_JOB_ID
WORKDIR=/scratch/gpfs/FISAC/jw4406/
mkdir -p $WORKDIR/$JOBID
#cd /n/fs/magics/2415498/FightLadder
cd $WORKDIR/$JOBID
echo "starting copy op"
cp -r $HOME/FightLadder/ ./
echo "copy op successful"
cd FightLadder/
module purge
module load anaconda3/2024.2
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
# options: my_pendulum, my_half_cheetah, my_hopper, my_ant
#!/bin/bash

# Number of parallel ippo instances to run
#!/bin/bash

# Number of parallel ippo instances to run
NUM_WORKERS=1
# If True, run each worker detached with nohup and log redirection.
# If False, run without nohup in the current shell.
RUN_LIVE="True"

# Get the directory where this script is located
#SCRIPT_DIR=$WORKDIR/$JOBID/FightLadder/main/train_ma/py

# Path to ippo.py
LEAGUE_PATH="$WORKDIR/$JOBID/FightLadder/main/train_ma.py"

# Arguments from launch.json (Python Debugger: ippo.py)
PLAYER=("Ryu")
OPPONENTS=("Guile" "ChunLi" "Zangief" "Sagat" "Ryu" "EHonda" "Blanka" "Dhalsim" "Ken" "MBison" "Balrog" "Vega")
# RIGHT_MODEL_FILE=""  # Optional: set this to pass --right-model-file
SAVE_DIR="$WORKDIR/$JOBID/FightLadder/main/trained_models/tasks/todo/"
TOTAL_STEPS="10000"
TRANSFORM_ACTION="True"
# Run NUM_WORKERS copies of leaguge in parallel
for i in $(seq 1 ${NUM_WORKERS}); do
    echo "Starting ippo instance ${i}..."
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
        echo "ippo instance ${i} started with PID $!"
    else
        "${CMD[@]}"
    fi
done

echo "Started ${NUM_WORKERS} ippo instances."
echo "Logs are being written to: ${LOGS_DIR}/"
