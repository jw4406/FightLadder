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
# V-trace worker sequence length T (spar arch only). "64" reproduces the prior
# default at NUM_ENV_STEPS>=256; lower it (e.g. 16/32) for more critic updates/sec.
VTRACE_SEQ_LEN="64"
# Three ippo.py knobs that were not reachable from SLURM. All EMPTY by default,
# which leaves ippo.py's own defaults in force -- so generated jobs behave
# exactly as before unless one of these is set.
#
#   GAMMA           unset => spar 0.99, ippo/2timescale 0.94. ippo.py's SPAR
#                   constructor passed nothing and silently inherited the class
#                   default 0.99 while the other two paths hardcoded 0.94 -- a
#                   ~13 s vs ~2.2 s horizon on ~27 s rounds. Measured ceiling on
#                   return prediction is 0.18-0.20 at gamma 0.75-0.9 vs 0.055 at
#                   0.99. NOTE that is PREDICTABILITY, not policy quality.
#   VTRACE_C_BAR    unset => 1.0. Truncates the TRACE ratio: sets variance and
#                   effective credit horizon, does NOT move the fixed point.
#                   Measured c_sat_frac ~0.5, i.e. half the traces clip.
#   VTRACE_RHO_BAR  unset => 5.0. Truncates the TD-error ratio and DOES set the
#                   fixed point -- changing it changes what is learned.
#   VTRACE_ENABLED  unset => True (ippo.py default, == the committed behaviour).
#                   When False the four VTRACE_* knobs below are INERT, so the
#                   cbar/replay levers become exact copies of their base config.
VTRACE_ENABLED=""
GAMMA=""
VTRACE_C_BAR=""
VTRACE_RHO_BAR=""
#   POPART          unset => False. PopArt value-target normalization. Enabling it
#                   renames value_net state_dict keys, so a POPART="True" job
#                   CANNOT resume from or be compared parameter-wise against any
#                   earlier checkpoint -- start it from scratch.
#   POPART_BETA     unset => 3e-4 (van Hasselt et al. 2016).
POPART=""
POPART_BETA=""
# ---------------------------------------------------------------------------
# OBSERVATION + MINIMAX-Q knobs. ALL EMPTY BY DEFAULT, so a rendered job behaves
# EXACTLY as before unless the sweep sets them -- same contract as GAMMA/POPART
# above (empty => flag omitted => ippo.py's own default applies).
#
#   OBS_TYPE / RAM_MASK  'ram' + a byte-index .npy. THE MASK IS LOAD-BEARING: a
#                   checkpoint records the WIDTH of its observation but not WHICH
#                   bytes, so a masked run is unevaluable without the exact file
#                   it trained with, and a DIFFERENT mask of the same width would
#                   load silently and feed the wrong bytes. RAM_MASK must
#                   therefore point INSIDE the per-config repo copy, not at a
#                   submitting-host path that will not exist on the compute node.
#
#   GAE_LAMBDA      unset => 0.95. Set 0 for a minimax bootstrap: that target is
#                   OFF-POLICY and a lambda-return splices ON-POLICY intermediate
#                   rewards into it, which is unsound without a Retrace trace
#                   (not built). lambda 0 ALSO changes the bias/variance
#                   tradeoff, so a kappa=0 arm at lambda 0 is the required
#                   control.
#
#   MINIMAX_BOOTSTRAP_KAPPA   0.0 = PHASE 0, the joint-action head trains and
#                   feeds NOTHING; the code path is not entered, so behaviour is
#                   bitwise identical to not having a head. >0 = PHASE 1, the
#                   head moves the policy.
OBS_TYPE=""
RAM_MASK=""
GAE_LAMBDA=""
MINIMAX_Q=""
MINIMAX_HEAD=""
MINIMAX_RANK=""
MINIMAX_TARGET=""
MINIMAX_BOOTSTRAP_KAPPA=""
MINIMAX_BOOTSTRAP_WARMUP=""
# ---------------------------------------------------------------------------
# REWARD SCALE + ZERO-SUM knobs. SET HERE TO RUN ac=1, UNSCALED.
#   REWARD_SCALE=1.0         raw hp-unit reward (ippo.py default 0.001 = "scaled").
#                            Trains the value function at full magnitude and
#                            un-binds Adam's second moment (sqrt(v) rises above
#                            eps=1e-8, restoring self-adaptation on the value head).
#   VALUE_CLIP_SEPARATE=True REQUIRED with unscaled reward: at reward_scale 1.0 the
#                            value grads are ~1000x larger and would dominate the
#                            joint clip_grad_norm(0.5), throttling the policy. Omit
#                            it and the unscaled run is NOT comparable to a scaled one.
#   AGGRESIVE_COEFF=1.0      ac=1 keeps the reward antisymmetric (zero-sum); a!=1
#                            makes the game general-sum and invalidates minimax-Q.
# Empty => flag omitted => ippo.py defaults (reward_scale 0.001, ac 1.0, vclip off).
# NOTE: these are NEW ippo.py args -- the compute node's ippo.py MUST be synced
# (git pull) or the job dies with "unrecognized arguments".
REWARD_SCALE="1.0"
AGGRESIVE_COEFF="1.0"
VALUE_CLIP_SEPARATE="True"
# Blend the multi-head adversary trunk update (spar arch only). "False" keeps the
# unchanged sequential per-head update; "True" gives one mean-over-heads trunk step.
BLEND_ADVERSARY_HEADS="False"
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
        --vtrace_seq_len "${VTRACE_SEQ_LEN}"
        --blend_adversary_heads "${BLEND_ADVERSARY_HEADS}"
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
    # Optional overrides -- omitted entirely when empty so ippo.py's defaults apply.
    if [ -n "${GAMMA}" ]; then
        CMD+=( --gamma "${GAMMA}" )
    fi
    if [ -n "${VTRACE_C_BAR}" ]; then
        CMD+=( --vtrace_c_bar "${VTRACE_C_BAR}" )
    fi
    if [ -n "${VTRACE_RHO_BAR}" ]; then
        CMD+=( --vtrace_rho_bar "${VTRACE_RHO_BAR}" )
    fi
    if [ -n "${VTRACE_ENABLED}" ]; then
        CMD+=( --vtrace_enabled "${VTRACE_ENABLED}" )
    fi
    if [ -n "${POPART}" ]; then
        CMD+=( --popart "${POPART}" )
    fi
    if [ -n "${POPART_BETA}" ]; then
        CMD+=( --popart_beta "${POPART_BETA}" )
    fi
    if [ -n "${OBS_TYPE}" ]; then
        CMD+=( --obs_type "${OBS_TYPE}" )
    fi
    # Resolved against the PER-CONFIG repo copy, so the mask travels with the job.
    if [ -n "${RAM_MASK}" ]; then
        CMD+=( --ram_mask "$WORKDIR/$JOBID/FightLadder/${RAM_MASK}" )
    fi
    if [ -n "${GAE_LAMBDA}" ]; then
        CMD+=( --gae_lambda "${GAE_LAMBDA}" )
    fi
    if [ -n "${MINIMAX_Q}" ]; then
        CMD+=( --minimax_q "${MINIMAX_Q}" )
    fi
    if [ -n "${MINIMAX_HEAD}" ]; then
        CMD+=( --minimax_head "${MINIMAX_HEAD}" )
    fi
    if [ -n "${MINIMAX_RANK}" ]; then
        CMD+=( --minimax_rank "${MINIMAX_RANK}" )
    fi
    if [ -n "${MINIMAX_TARGET}" ]; then
        CMD+=( --minimax_target "${MINIMAX_TARGET}" )
    fi
    if [ -n "${MINIMAX_BOOTSTRAP_KAPPA}" ]; then
        CMD+=( --minimax_bootstrap_kappa "${MINIMAX_BOOTSTRAP_KAPPA}" )
    fi
    if [ -n "${MINIMAX_BOOTSTRAP_WARMUP}" ]; then
        CMD+=( --minimax_bootstrap_warmup "${MINIMAX_BOOTSTRAP_WARMUP}" )
    fi
    # Reward scale / zero-sum knobs (empty => ippo.py default). Set above for ac=1 unscaled.
    if [ -n "${REWARD_SCALE}" ]; then
        CMD+=( --reward_scale "${REWARD_SCALE}" )
    fi
    if [ -n "${AGGRESIVE_COEFF}" ]; then
        CMD+=( --aggresive_coeff "${AGGRESIVE_COEFF}" )
    fi
    if [ -n "${VALUE_CLIP_SEPARATE}" ]; then
        CMD+=( --value_clip_separate "${VALUE_CLIP_SEPARATE}" )
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
