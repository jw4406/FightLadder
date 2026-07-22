#!/bin/bash
# Launch the two-timescale LR-ratio sweep (SPAR main-training + dedicated BR).
#
# Fixes c_lr and sweeps d_lr = c_lr*m_d, v_lr = d_lr*m_v (ego < adv < critic).
# Per (m_d, m_v) config it renders a main-training .slurm (repo copied into a
# deterministic per-config tree $WORKDIR/lr_sweep/<tag>/FightLadder) and a
# CPU-only BR-orchestrator .slurm (watchdog runs on a compute node, NOT login),
# and sbatches them. MAIN_TRAINING_DIR=lr_sweep/<tag> keeps training + BR outputs
# (tasks, checkpoints, br_rewards, br_models) together per config.
#
# DRY_RUN=True (default) renders the .slurm files and prints the sbatch commands,
# but submits nothing. Flip to False to go live.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DRY_RUN="True"
PHASE="both"                             # train | br | both
DISCOVER="False"                         # with PHASE=br: BR every existing lr_sweep/<tag> tree (ignores the grid)
CLUSTER="neuronic"                       # neuronic | della
WORKDIR="/scratch/gpfs/FISAC/jw4406/"    # scratch WORKDIR
PLAYER=(Vega)
OPPONENTS=(Sagat Ryu)

# Ratio grid: c_lr fixed; d_lr = c_lr*m_d; v_lr = d_lr*m_v; skip v_lr > MAX_V_LR.
C_LR="1e-5"
D_MULTS=(2 4 8 16)
V_MULTS=(2 4 8)
MAX_V_LR="1e-3"

MAIN_TRAINING_STEPS="150000000"
TRAIN_TIME="096:00:00"
BR_JOB_TIME="096:00:00"                   # orchestrator-job walltime (>= training)

# BR (curve mode): STEP_STRIDE=0 -> BR every checkpoint; PERIODIC_EVAL_FREQ sets
# mid-training snapshot cadence for the exploitability curve.
STEP_STRIDE="0"
PERIODIC_EVAL_FREQ="500000"
BR_TRAINING_STEPS="99999"
EXPLOITER_SAVE_FREQ="1000"
BR_SLURM_TIME="000:12:00"

# GPU packing: 1 = one exploiter per GPU (default, unchanged). Set 2 to co-locate
# 2 exploiters/GPU, each capped at GPU_MEM_FRACTION of the card.
EXPLOITERS_PER_JOB="1"
GPU_MEM_FRACTION="0.45"

CMD=(
    python "${SCRIPT_DIR}/lr_ratio_sweep.py"
    --phase "${PHASE}"
    --c_lr "${C_LR}"
    --d_mults "${D_MULTS[@]}"
    --v_mults "${V_MULTS[@]}"
    --max_v_lr "${MAX_V_LR}"
    --player "${PLAYER[@]}"
    --opponent_list "${OPPONENTS[@]}"
    --main_training_steps "${MAIN_TRAINING_STEPS}"
    --time "${TRAIN_TIME}"
    --workdir "${WORKDIR}"
    --cluster "${CLUSTER}"
    --br_job_time "${BR_JOB_TIME}"
    --step_stride "${STEP_STRIDE}"
    --periodic_eval_freq "${PERIODIC_EVAL_FREQ}"
    --br_training_steps "${BR_TRAINING_STEPS}"
    --exploiter_save_freq "${EXPLOITER_SAVE_FREQ}"
    --br_slurm_time "${BR_SLURM_TIME}"
    --exploiters_per_job "${EXPLOITERS_PER_JOB}"
    --gpu_mem_fraction "${GPU_MEM_FRACTION}"
    --dry_run "${DRY_RUN}"
)
if [ "${DISCOVER}" = "True" ]; then CMD+=(--discover); fi
"${CMD[@]}"
