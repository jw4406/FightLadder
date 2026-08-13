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
PHASE="both"                             # train | br | both  (WHICH JOBS to submit)
# TRANCHE is a SEPARATE axis from PHASE -- it selects what the training job DOES:
#   ""    render exactly as before this flag existed (no minimax knobs touched)
#   p0    joint-action head trains and feeds NOTHING (kappa 0, bitwise inert)
#   p1    head FEEDS the GAE bootstrap (kappa 1, gae_lambda 0, factored head)
# Launch p0 and p1 as two tranches; the tag gains a _p0/_p1 suffix so their
# trees cannot collide. A p1 tranche is ONLY interpretable against a p0 tranche
# at the same gae_lambda, because lambda 0 changes bias/variance on its own.
TRANCHE=""
# Observation. RAM_MASK is REPO-RELATIVE (it is prefixed with the per-config
# repo copy on the compute node) and MUST be committed -- a masked checkpoint is
# unevaluable without the exact mask it trained with.
OBS_TYPE=""
RAM_MASK=""
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
# V-trace worker sequence length T (spar arch only). Empty => keep template
# default (64). Lower it (e.g. 16 or 32) for cheaper V-trace forwards => more
# critic updates/sec, to help the critic track a fast adversary.
VTRACE_SEQ_LEN=""
# Blend the multi-head adversary trunk update (spar arch only). Empty => keep
# template default ("False" = sequential). Set "True" for one mean-over-heads
# trunk step per batch (removes the later-matchup ordering bias).
BLEND_ADVERSARY_HEADS=""
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
# Cross-checkpoint packing: co-locate exploiters from DIFFERENT checkpoints on one
# GPU (needed to fill the card when a config has only 1 replicate = 2 specs/ckpt).
PACK_ACROSS_CHECKPOINTS="False"
PACK_FLUSH_TIMEOUT="300"
# Absolute cpu multiplier for PACKED BR jobs (cpu only; mem still scales by the
# co-located exploiter count). 1 = template base cpus. Set independently of
# EXPLOITERS_PER_JOB, e.g. RESOURCE_SCALE="6".
RESOURCE_SCALE="1"

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
    --pack_across_checkpoints "${PACK_ACROSS_CHECKPOINTS}"
    --pack_flush_timeout "${PACK_FLUSH_TIMEOUT}"
    --resource_scale "${RESOURCE_SCALE}"
    --dry_run "${DRY_RUN}"
)
if [ -n "${TRANCHE}" ]; then CMD+=(--tranche "${TRANCHE}"); fi
if [ -n "${OBS_TYPE}" ]; then CMD+=(--obs_type "${OBS_TYPE}"); fi
if [ -n "${RAM_MASK}" ]; then CMD+=(--ram_mask "${RAM_MASK}"); fi
if [ "${DISCOVER}" = "True" ]; then CMD+=(--discover); fi
if [ -n "${VTRACE_SEQ_LEN}" ]; then CMD+=(--vtrace_seq_len "${VTRACE_SEQ_LEN}"); fi
if [ -n "${BLEND_ADVERSARY_HEADS}" ]; then CMD+=(--blend_adversary_heads "${BLEND_ADVERSARY_HEADS}"); fi
"${CMD[@]}"
