#!/bin/bash
#SBATCH --job-name=vtA_g094_vton_RySa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80|gpu40
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/vtA_g094_vton_%j.out
#SBATCH --error=slurm_logs/vtA_g094_vton_%j.err
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jw4406@princeton.edu
#
# V-TRACE ISOLATION, ARM A -- gamma 0.94 with V-trace ON.
#
# WHY: the local reference run (gamma 0.94, V-trace OFF, c_lr 1e-5) CYCLED --
# 5 rating_gap zero crossings in 390 iterations, amplitude -121 -> +307 -> -324 --
# instead of collapsing monotonically like the gamma 0.99 + V-trace baseline
# (-558, ego won 3.9%, and it stayed there).
#
# Hypothesis: V-trace's replay buffer (capacity 15000, mean_age ~7400) makes the
# critic fit a TIME-AVERAGE of recent opponents. Averaging over past opponents is
# what fictitious play does, and it DAMPS rotation -- so the buffer was supplying
# artificial damping that hid the antisymmetric part of the game Jacobian, at the
# cost of a shared lag that compresses effective timescale separation toward 1.
# Removing it revealed the rotational dynamics.
#
# This arm changes ONLY the estimator. Together with runs already in hand it
# closes a 2x2 at identical c_lr 1e-5 / d_lr 1e-4:
#
#                  vtrace ON                    vtrace OFF
#   gamma 0.99     baseline (monotone -558)     arm B (running locally)
#   gamma 0.94     THIS JOB                     reference (cycled)
#
# READ IT ON: rating_gap zero crossings and swing amplitude, NOT on final score.
#   Cycling in BOTH vtrace-off cells and NEITHER vtrace-on cell => the buffer is
#   the damping term. Tracking gamma instead => the horizon is doing the work.
#   Also watch steps/episode: >400 means the game is STALLING and every EV number
#   on that checkpoint is uninterpretable (see critic_diagnostics.py).
#
# NOTE c_lr is 1e-5 here -- the REFERENCE run's value, deliberately NOT the 3e-5
# from the timescale experiment. Changing it would confound this with that one.
#
# RESOURCES (measured locally, not guessed):
#   GPU  ~19.3 GiB with V-trace ON. Two such runs do NOT fit on one 24 GiB card;
#        this job needs a GPU to itself.
#   RAM  ~28 GB for the V-trace replay buffers alone: two buffers (ego + adv) at
#        capacity 15000 x 24 envs x 37.5 KiB uint8 = 13.8 GB each, allocated as
#        numpy on the HOST, not on the device. Hence --mem=96G.

set -u

module purge
module load anaconda3/2024.02
conda activate fightladder

WORKDIR="${WORKDIR:-/home/jw4406/codebase/FightLadder}"
cd "${WORKDIR}"
mkdir -p slurm_logs

# CRITICAL -- checkpoint isolation.
# .task checkpoints are written by FileQueueTriggerCallback to ippo.py's
# TASK_DIR, which is derived from the SCRIPT's path. It ignores --save_dir AND
# cwd. Every spar Ryu-vs-Sagat run produces the SAME filenames
# (model_name_prefix = "spar_Ry_Sa"), so two jobs sharing a TASK_DIR silently
# overwrite each other. That already destroyed a baseline checkpoint locally.
# FIGHTLADDER_TASK_DIR is the only override.
export FIGHTLADDER_TASK_DIR="${WORKDIR}/main/slurm_runs/${SLURM_JOB_ID}/trained_models/tasks"
mkdir -p "${FIGHTLADDER_TASK_DIR}/todo" "${FIGHTLADDER_TASK_DIR}/todo_continue"
SAVE_DIR="${FIGHTLADDER_TASK_DIR}/todo"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IPPO_PATH="${WORKDIR}/main/ippo.py"

# --- the experiment ---------------------------------------------------------
GAMMA="0.94"
# EXPLICIT. This used to be a hardcoded literal in ippo.py that differed between
# HEAD (True) and the local working tree (False), so nothing in any launcher or
# log revealed which one a job got -- and every --vtrace_* flag below is INERT
# when it is False. Never leave this to the default for this experiment.
VTRACE_ENABLED="True"

# --- everything below matches the local reference run -----------------------
PLAYER=("Ryu")
OPPONENTS=("Sagat")
C_LR="1e-5"                 # reference value; NOT the timescale arm's 3e-5
D_LR="1e-4"
V_LR="4e-4"
ENVS_PER_MATCHUP="24"
ENV_BATCH_SIZE="24"
NUM_ENV_TO_LOAD="1"
NUM_ENV_STEPS="512"
TRAINING_BATCH_SIZE="1024"
VTRACE_SEQ_LEN="64"
VTRACE_C_BAR="1.0"
VTRACE_RHO_BAR="5.0"
VTRACE_REPLAY_CAPACITY="15000"
POPART="False"              # separate variable; would confound this experiment
CHECKPOINT_INTERVAL="20000" # x24 envs = a checkpoint every 480k steps
TOTAL_TIMESTEPS="150000000"
NUM_PERTURBS="10"
EGO_SIDE="left"
SIDE="both"
USE_MIRROR="False"
TRANSFORM_ACTION="True"
MODEL_ARCH_TYPE="spar"
EGO_STYLE="learning"
ADV_STYLE="learning"
RENDER="False"
ASYNC_UPDATE="False"
MASTER_USE_STAG="False"
USE_LR_ANNEALING="False"
LR_ANNEAL_COEFF=".995"
OBS_TYPE="image"
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

CMD=(
    python -u "${IPPO_PATH}"
    --player "${PLAYER[@]}"
    --opponents "${OPPONENTS[@]}"
    --ego_style "${EGO_STYLE}"
    --adv_style "${ADV_STYLE}"
    --save_dir "${SAVE_DIR}"
    --num_env_to_load "${NUM_ENV_TO_LOAD}"
    --env_batch_size "${ENV_BATCH_SIZE}"
    --c_lr "${C_LR}" --d_lr "${D_LR}" --v_lr "${V_LR}"
    --num_perturbs "${NUM_PERTURBS}"
    --use_mirror "${USE_MIRROR}"
    --ego_side "${EGO_SIDE}"
    --side "${SIDE}"
    --envs_per_matchup "${ENVS_PER_MATCHUP}"
    --num_env_steps "${NUM_ENV_STEPS}"
    --gamma "${GAMMA}"
    --vtrace_enabled "${VTRACE_ENABLED}"
    --vtrace_seq_len "${VTRACE_SEQ_LEN}"
    --vtrace_c_bar "${VTRACE_C_BAR}"
    --vtrace_rho_bar "${VTRACE_RHO_BAR}"
    --vtrace_replay_capacity "${VTRACE_REPLAY_CAPACITY}"
    --popart "${POPART}"
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"
    --training_batch_size "${TRAINING_BATCH_SIZE}"
    --total_timesteps "${TOTAL_TIMESTEPS}"
    --transform_action "${TRANSFORM_ACTION}"
    --model_arch_type "${MODEL_ARCH_TYPE}"
    --use_lr_annealing "${USE_LR_ANNEALING}"
    --lr_anneal_coeff "${LR_ANNEAL_COEFF}"
    --render "${RENDER}"
    --async_update "${ASYNC_UPDATE}"
    --master_use_stag "${MASTER_USE_STAG}"
    --obs_type "${OBS_TYPE}"
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

echo "=== VTRACE ARM A: gamma ${GAMMA}, vtrace_enabled ${VTRACE_ENABLED} ==="
echo "  job        : ${SLURM_JOB_ID}"
echo "  node       : $(hostname)"
echo "  task_dir   : ${FIGHTLADDER_TASK_DIR}"
echo "  c_lr/d_lr  : ${C_LR} / ${D_LR}"
echo "  critic horizon ~$(python -c "print(round(1/(1-${GAMMA}),1))") steps"
# ippo.py itself prints a [config] line reporting the settings it ACTUALLY got,
# including whether the vtrace knobs are ACTIVE or INERT. Check it in the .out
# before trusting that this job ran the arm you intended.
exec "${CMD[@]}"
