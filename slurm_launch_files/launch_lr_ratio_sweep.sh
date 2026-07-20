#!/bin/bash
# Launch the two-timescale LR-ratio sweep (SPAR main-training + dedicated BR).
#
# Fixes c_lr and sweeps d_lr = c_lr*m_d, v_lr = d_lr*m_v (ego < adv < critic).
# Renders one main-training .slurm per (m_d, m_v) config from cds_style_template.slurm
# with a deterministic per-config SPAR_TASK_DIR, submits it, and launches a
# dedicated br_slurm_orchestrator.py watchdog pointed at that config's task dirs.
#
# DRY_RUN=True (default) writes the rendered .slurm files and prints the exact
# sbatch / orchestrator commands, but submits nothing. Flip to False to go live.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DRY_RUN="True"
CLUSTER="neuronic"                       # neuronic | della
WORKDIR="/scratch/gpfs/FISAC/jw4406/"    # scratch WORKDIR substituted into the template
PLAYER=(Vega)
OPPONENTS=(Sagat Ryu)

# Ratio grid: c_lr fixed; d_lr = c_lr*m_d; v_lr = d_lr*m_v; skip v_lr > MAX_V_LR.
C_LR="1e-5"
D_MULTS=(2 4 8 16)
V_MULTS=(2 4 8)
MAX_V_LR="1e-3"

MAIN_TRAINING_STEPS="150000000"
TRAIN_TIME="096:00:00"

# BR (curve mode): STEP_STRIDE=0 -> BR every checkpoint; PERIODIC_EVAL_FREQ sets
# mid-training snapshot cadence for the exploitability curve.
STEP_STRIDE="0"
PERIODIC_EVAL_FREQ="500000"
BR_TRAINING_STEPS="99999"
EXPLOITER_SAVE_FREQ="1000"
BR_SLURM_TIME="000:12:00"

CMD=(
    python "${SCRIPT_DIR}/lr_ratio_sweep.py"
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
    --step_stride "${STEP_STRIDE}"
    --periodic_eval_freq "${PERIODIC_EVAL_FREQ}"
    --br_training_steps "${BR_TRAINING_STEPS}"
    --exploiter_save_freq "${EXPLOITER_SAVE_FREQ}"
    --br_slurm_time "${BR_SLURM_TIME}"
    --dry_run "${DRY_RUN}"
)
"${CMD[@]}"
