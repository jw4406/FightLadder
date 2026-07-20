#!/bin/bash
#SBATCH --job-name=spar_Ry_GuChZaSaRyEHBlDhKeMBBaVe  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=72      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jw4406@princeton.edu
echo $CUDA_VISIBLE_DEVICES
JOBID=$SLURM_JOB_ID
WORKDIR=/n/fs/magics/
mkdir -p $WORKDIR/$JOBID
#cd /n/fs/magics/2415498/FightLadder
cd $WORKDIR/$JOBID
echo "starting copy op"
cp -r $HOME/FightLadder/ ./
echo "copy op successful"
cd FightLadder/
module purge
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
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to ippo.py
IPPO_PATH="$WORKDIR/$JOBID/FightLadder/main/ippo.py"

# Arguments from launch.json (Python Debugger: ippo.py)
PLAYER=("Ryu")
OPPONENTS=("Guile" "ChunLi" "Zangief" "Sagat" "Ryu" "EHonda" "Blanka" "Dhalsim" "Ken" "MBison" "Balrog" "Vega")
NUM_ENV_TO_LOAD="1"
ENV_BATCH_SIZE="24"
C_LR="1e-5"
D_LR="2e-5"
V_LR="4e-4"
# Ego value-head LR. Leave empty for spar (optional/ignored). The orchestrator
# sets this for ippo/2timescale runs, where ippo.py requires it.
EGO_VALUE_HEAD_LR=""
NUM_PERTURBS="10"
USE_MIRROR="False"
EGO_SIDE="left"
# LOAD_PATH=""  # Optional: set this to pass --load_path
# TRAINING_STYLE=""  # Optional: set this to pass --training_style
# CONTINUE_TRAINING=""  # Optional: set this to pass --continue_training
# LEFT_MODEL_FILE=""  # Optional: set this to pass --left-model-file
# RIGHT_MODEL_FILE=""  # Optional: set this to pass --right-model-file
SAVE_DIR="$WORKDIR/$JOBID/FightLadder/main/trained_models/tasks/todo/"
USE_LR_ANNEALING="False"
LR_ANNEAL_COEFF=".995"
CHECKPOINT_INTERVAL="50000"
TOTAL_TIMESTEPS="999"
TRAINING_BATCH_SIZE="512"
NUM_ENV_STEPS="1024"
EGO_STYLE="learning"
ADV_STYLE="learning"
ENVS_PER_MATCHUP="2"
SIDE="both"
RENDER="False"
MODEL_FILE=""

USE_WANDB="False"

MASTER_USE_STAG="False"
TRANSFORM_ACTION="True"
ASYNC_UPDATE="False"
MODEL_ARCH_TYPE="spar"
USE_STAGNATION_EARLY_STOP="False"
USE_STAGNATION_VELOCITY_SIGNAL="False"
USE_STAGNATION_ENTROPY_SIGNAL="False"
STAGNATION_PATIENCE="20000"
STAGNATION_TOLERANCE="1e-4"
STAGNATION_REL_TOLERANCE="0.05"
STAGNATION_EMA_BETA="0.99"
STAGNATION_EPS="1e-8"
STAGNATION_EVAL_GAMES="0"
ENTROPY_STAGNATION_WEIGHT="1.0"
STAGNATION_LR_FACTOR="0.999"
STAGNATION_LR_PATIENCE="150"
# Create logs directory if it doesn't exist
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# Run NUM_WORKERS copies of ippo in parallel
for i in $(seq 1 ${NUM_WORKERS}); do
    echo "Starting ippo instance ${i}..."
    CMD=(
        python "${IPPO_PATH}"
        --player "${PLAYER[@]}"
        --opponents "${OPPONENTS[@]}"
        --num_env_to_load "${NUM_ENV_TO_LOAD}"
        --env_batch_size "${ENV_BATCH_SIZE}"
        --c_lr "${C_LR}"
        --d_lr "${D_LR}"
        --v_lr "${V_LR}"
        --num_perturbs "${NUM_PERTURBS}"
        --use_mirror "${USE_MIRROR}"
        --ego_side "${EGO_SIDE}"
        --save_dir "${SAVE_DIR}"
        --use_lr_annealing "${USE_LR_ANNEALING}"
        --lr_anneal_coeff "${LR_ANNEAL_COEFF}"
	    --training_batch_size "${TRAINING_BATCH_SIZE}"
        --checkpoint_interval "${CHECKPOINT_INTERVAL}"
	--total_timesteps "${TOTAL_TIMESTEPS}"
        --num_env_steps "${NUM_ENV_STEPS}"
        --ego_style "${EGO_STYLE}"
        --adv_style "${ADV_STYLE}"
        --envs_per_matchup "${ENVS_PER_MATCHUP}"
	--transform_action "${TRANSFORM_ACTION}"
        --side "${SIDE}"
	--use_wandb "${USE_WANDB}"
        --render "${RENDER}"
        --model_file "${MODEL_FILE}"
        --async_update "${ASYNC_UPDATE}"
	    --model_arch_type "${MODEL_ARCH_TYPE}"
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
    )

    # Only pass ego value-head LR when set (ippo/2timescale). Omitted for spar,
    # where ippo.py treats it as optional.
    if [ -n "${EGO_VALUE_HEAD_LR}" ]; then
        CMD+=( --ego_value_head_lr "${EGO_VALUE_HEAD_LR}" )
    fi

    if [ "${RUN_LIVE}" = "False" ]; then
        nohup "${CMD[@]}" > "${LOGS_DIR}/ippo_worker_${i}.log" 2>&1 &
        echo "ippo instance ${i} started with PID $!"
    else
        "${CMD[@]}"
    fi
done

echo "Started ${NUM_WORKERS} ippo instances."
echo "Logs are being written to: ${LOGS_DIR}/"
