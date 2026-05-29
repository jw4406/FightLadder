#!/bin/bash
#SBATCH --job-name=br_spar_Gu_SaRy_2908732

#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=52       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jw4406@princeton.edu
echo $CUDA_VISIBLE_DEVICES
JOBID=$SLURM_JOB_ID
WORKDIR=/scratch/gpfs/FISAC/jw4406

MAIN_TRAINING_DIR=2908732

proj_name=guile_exploiter_1v6_noanneal
analysis_upload_proj_name=guile_exploiter_1v6_noanneal_analysis

mkdir -p $WORKDIR/$JOBID
#cp -r /n/fs/magics/$MAIN_TRAINING_DIR/FightLadder ./
cd $WORKDIR/$JOBID
cp -r $HOME/FightLadder ./
cd FightLadder
cd main/trained_models/tasks/todo

#cp -r /n/fs/magics/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/todo/*.task ./

cd $WORKDIR/$JOBID/FightLadder

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
cd main

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Number of parallel new_br_worker instances to run#!/bin/bash

# Number of parallel new_br_worker instances to run

# Number of parallel new_br_worker instances to run

# Number of parallel new_br_worker instances to run
NUM_WORKERS=2
# If True, run each worker detached with nohup and log redirection.
# If False, run without nohup in the current shell.
RUN_LIVE="True"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to new_br_worker.py
BR_WORKER_PATH="$WORKDIR/$JOBID/FightLadder/main/new_br_worker.py"

# Arguments from launch.json (Python Debugger: new_br_worker.py)
EVAL_PROT="True"
EVAL_ADV="True"
EVAL_ONLY="False"
PROJ_NAME="br_training_ema_test_2"
ANALYSIS_UPLOAD_PROJ_NAME="br_analysis"
LOAD_BR="False"
WHICH_ENV="my_pendulum"
IS_LEAGUE="False"
LEAGUE_DIR="/home/jw4406/codebase/FightLadder/main/trained_models/ma/"
# League matchup states (required when IS_LEAGUE="True"). Space-separated retro state strings.
# Example for Ryu vs {Guile, Sagat, EHonda}:
# LEAGUE_MATCHUP_STATES=(
#     "two_player/Ryu_left/Champion.Level1.RyuVsGuile.2Player.state"
#     "two_player/Ryu_left/Champion.Level1.RyuVsSagat.2Player.state"
#     "two_player/Ryu_left/Champion.Level1.RyuVsEhonda.2Player.state"
# )
LEAGUE_MATCHUP_STATES=()
USE_MIRROR="False"
NUM_FULL_EXPLOITERS="3"
NUM_CONTINUE_EXPLOITERS="1"
MAX_CONCURRENT_JOBS="0"  # 0 = auto-compute from NUM_CORES. -1 = unlimited.
NUM_CORES="26"  # CPU cores for this worker. 0 = auto-detect. Divide total cores by NUM_WORKERS.
DEBUG="False"
BR_TRACKER_PATIENCE="300"
USE_BR_REWARD_STAGNATION="False"
USE_BR_ENTROPY_STAGNATION="True"
BR_USE_SLOPE_EARLY_STOP="False"
BR_SLOPE_WINDOW="20"
BR_SLOPE_TOLERANCE="5e-3"
BR_MIN_SLOPE_CHECKS="12"
# Continue-exploiter CDS stagnation controls
USE_STAGNATION_EARLY_STOP="True"
USE_STAGNATION_VELOCITY_SIGNAL="False"
USE_STAGNATION_ENTROPY_SIGNAL="True"
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
N_ENVS="4"
TASK_DIR="$WORKDIR/$MAIN_TRAINING_DIR/FightLadder/main/trained_models/tasks/"  # Optional: set this to pass --task_dir
echo $TASK_DIR
DEDICATED_EXPLOITER="True"
CONTINUE_EXPLOITERS="False"
EXPLOITER_SAVE_FREQ="500000"
RESET="round"
SIDE="both"
RENDER="False"
ENABLE_COMBO="True"
NULL_COMBO="False"
TRANSFORM_ACTION="True"
SEED="0"
LAUNCH_LOCAL_BR_EVAL="True"
USE_WANDB="False"
# Torch device: cpu, cuda, cuda:0, etc. (must match launch.json / worker --device)
DEVICE="cuda"
if [ "${DEVICE}" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

# Create logs directory if it doesn't exist
#LOGS_DIR="${SCRIPT_DIR}/logs"
#mkdir -p "${LOGS_DIR}"
#conda activate mujoco_sb3_parallel
# Run NUM_WORKERS copies of new_br_worker in parallel
set -x                              # trace every command
echo "DEBUG: pwd=$(pwd)"
echo "DEBUG: which python=$(which python)"
echo "DEBUG: python --version=$(python --version 2>&1)"
echo "DEBUG: conda env=$CONDA_DEFAULT_ENV"
echo "DEBUG: ls TASK_DIR=$(ls -la "${TASK_DIR}" 2>&1 | head)"
for i in $(seq 1 ${NUM_WORKERS}); do
    echo "Starting new_br_worker instance ${i}..."
    CMD=(
        python "${BR_WORKER_PATH}"
        --eval_prot "${EVAL_PROT}"
        --eval_adv "${EVAL_ADV}"
        --eval_only "${EVAL_ONLY}"
        --proj_name "${PROJ_NAME}"
        --analysis_upload_proj_name "${ANALYSIS_UPLOAD_PROJ_NAME}"
        --load_br "${LOAD_BR}"
        --which_env "${WHICH_ENV}"
        --is_league "${IS_LEAGUE}"
        --league_dir "${LEAGUE_DIR}"
	--task_dir "${TASK_DIR}"
        --use_mirror "${USE_MIRROR}"
        --num_full_exploiters "${NUM_FULL_EXPLOITERS}"
        --num_continue_exploiters "${NUM_CONTINUE_EXPLOITERS}"
        --max_concurrent_jobs "${MAX_CONCURRENT_JOBS}"
        --num_cores "${NUM_CORES}"
        --DEBUG "${DEBUG}"
        --n_envs "${N_ENVS}"
        --dedicated_exploiter "${DEDICATED_EXPLOITER}"
        --continue_exploiters "${CONTINUE_EXPLOITERS}"
        --exploiter_save_freq "${EXPLOITER_SAVE_FREQ}"
        --br_tracker_patience "${BR_TRACKER_PATIENCE}"
        --use_br_reward_stagnation "${USE_BR_REWARD_STAGNATION}"
        --use_br_entropy_stagnation "${USE_BR_ENTROPY_STAGNATION}"
        --br_use_slope_early_stop "${BR_USE_SLOPE_EARLY_STOP}"
        --br_slope_window "${BR_SLOPE_WINDOW}"
        --br_slope_tolerance "${BR_SLOPE_TOLERANCE}"
        --br_min_slope_checks "${BR_MIN_SLOPE_CHECKS}"
        --use_stagnation_early_stop "${USE_STAGNATION_EARLY_STOP}"
        --use_stagnation_velocity_signal "${USE_STAGNATION_VELOCITY_SIGNAL}"
        --use_stagnation_entropy_signal "${USE_STAGNATION_ENTROPY_SIGNAL}"
        --stagnation_patience "${STAGNATION_PATIENCE}"
        --stagnation_tolerance "${STAGNATION_TOLERANCE}"
        --stagnation_rel_tolerance "${STAGNATION_REL_TOLERANCE}"
        --stagnation_ema_beta "${STAGNATION_EMA_BETA}"
        --stagnation_eps "${STAGNATION_EPS}"
        --stagnation_eval_games "${STAGNATION_EVAL_GAMES}"
        --entropy_stagnation_weight "${ENTROPY_STAGNATION_WEIGHT}"
        --stagnation_lr_factor "${STAGNATION_LR_FACTOR}"
        --stagnation_lr_patience "${STAGNATION_LR_PATIENCE}"
        --stagnation_use_slope_early_stop "${STAGNATION_USE_SLOPE_EARLY_STOP}"
        --stagnation_slope_window "${STAGNATION_SLOPE_WINDOW}"
        --stagnation_slope_tolerance "${STAGNATION_SLOPE_TOLERANCE}"
        --stagnation_min_slope_checks "${STAGNATION_MIN_SLOPE_CHECKS}"
        --reset "${RESET}"
        --side "${SIDE}"
        --render "${RENDER}"
        --enable_combo "${ENABLE_COMBO}"
        --null_combo "${NULL_COMBO}"
        --transform_action "${TRANSFORM_ACTION}"
        --seed "${SEED}"
        --device "${DEVICE}"
        --launch_local_br_eval "${LAUNCH_LOCAL_BR_EVAL}"
        --use_wandb "${USE_WANDB}"
    )

    # Append league matchup states when running in league mode.
    if [ "${IS_LEAGUE}" = "True" ] && [ ${#LEAGUE_MATCHUP_STATES[@]} -gt 0 ]; then
        CMD+=(--league_matchup_states "${LEAGUE_MATCHUP_STATES[@]}")
    fi

    if [ "${RUN_LIVE}" = "False" ]; then
        nohup "${CMD[@]}" > "${LOGS_DIR}/br_worker_${i}.log" 2>&1 &

        echo "new_br_worker instance ${i} started with PID $!"
    else
        "${CMD[@]}"
    fi
done

echo "Started ${NUM_WORKERS} new_br_worker instances."
#echo "Logs are being written to: ${LOGS_DIR}/"
echo "To stop all workers, create a STOP file in the task directory or kill the processes."
