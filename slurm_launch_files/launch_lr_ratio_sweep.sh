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
# V-trace worker sequence length T (spar arch only). Set explicitly (64
# reproduces the prior template default). Lower it (e.g. 16 or 32) for cheaper
# V-trace forwards => more critic updates/sec, to help the critic track a fast
# adversary. (Empty would omit the flag and fall back to ippo.py's default.)
VTRACE_SEQ_LEN="64"
# Blend the multi-head adversary trunk update (spar arch only). "False" = the
# prior sequential update; "True" = one mean-over-heads trunk step per batch
# (removes the later-matchup ordering bias). (Empty would fall back to default.)
BLEND_ADVERSARY_HEADS="False"
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

# ---------------------------------------------------------------------------
# FULL PASS-THROUGH. Every SLURM directive and body knob of BOTH rendered
# templates is set HERE, so a rendered job relies on NOTHING from
# cds_style_template.slurm / br_orchestrator_job_template.slurm defaults. Values
# below equal each template's current default, so flipping to live changes
# nothing until you edit one. (Derived paths -- SAVE_DIR, IPPO_PATH, LOGS_DIR,
# REPO_DIR, TASK_BASE -- stay computed inside the templates from WORKDIR/JOBID.)

# -- SLURM header: training job --
TRAIN_NODES="1"
TRAIN_NTASKS="1"
TRAIN_CPUS_PER_TASK="72"
TRAIN_MEM_PER_CPU="2G"
TRAIN_GRES="gpu:1"
# -- SLURM header: BR-orchestrator job (CPU-only, no --gres) --
ORCH_NODES="1"
ORCH_NTASKS="1"
ORCH_CPUS_PER_TASK="2"
ORCH_MEM_PER_CPU="4G"
# -- mail (both jobs; the template's two mail-type lines collapse to one) --
MAIL_TYPE="BEGIN,END"
MAIL_USER="jw4406@princeton.edu"

# -- training template body knobs --
NUM_WORKERS="1"
RUN_LIVE="True"
NUM_ENV_TO_LOAD="1"
ENV_BATCH_SIZE="24"
EGO_VALUE_HEAD_LR=""                      # empty => omitted (spar); set for ippo/2timescale
NUM_PERTURBS="10"
USE_MIRROR="False"
EGO_SIDE="left"
USE_LR_ANNEALING="False"
LR_ANNEAL_COEFF=".995"
CHECKPOINT_INTERVAL="50000"
TRAINING_BATCH_SIZE="512"
NUM_ENV_STEPS="1024"
# (VTRACE_SEQ_LEN and BLEND_ADVERSARY_HEADS are set near the top with the other
# spar knobs; they flow through the existing conditional appends below.)
# Optional ippo.py knobs: empty => flag omitted => ippo.py's own default applies.
VTRACE_ENABLED=""                         # empty => True
GAMMA=""                                  # empty => spar 0.99
VTRACE_C_BAR=""                           # empty => 1.0
VTRACE_RHO_BAR=""                         # empty => 5.0
POPART=""                                 # empty => False
POPART_BETA=""                            # empty => 3e-4
# Reward scale / zero-sum (ac=1, unscaled).
REWARD_SCALE="1.0"
AGGRESIVE_COEFF="1.0"
VALUE_CLIP_SEPARATE="True"
EGO_STYLE="learning"
ADV_STYLE="learning"
ENVS_PER_MATCHUP="2"
SIDE="both"
RENDER="False"
MODEL_FILE=""
USE_WANDB="False"
TRANSFORM_ACTION="True"
ASYNC_UPDATE="False"
MODEL_ARCH_TYPE="spar"

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
    # -- full pass-through: SLURM headers (both jobs) + mail --
    --train_nodes "${TRAIN_NODES}"
    --train_ntasks "${TRAIN_NTASKS}"
    --train_cpus_per_task "${TRAIN_CPUS_PER_TASK}"
    --train_mem_per_cpu "${TRAIN_MEM_PER_CPU}"
    --train_gres "${TRAIN_GRES}"
    --orch_nodes "${ORCH_NODES}"
    --orch_ntasks "${ORCH_NTASKS}"
    --orch_cpus_per_task "${ORCH_CPUS_PER_TASK}"
    --orch_mem_per_cpu "${ORCH_MEM_PER_CPU}"
    --mail_type "${MAIL_TYPE}"
    --mail_user "${MAIL_USER}"
    # -- full pass-through: training body knobs --
    --num_workers "${NUM_WORKERS}"
    --run_live "${RUN_LIVE}"
    --num_env_to_load "${NUM_ENV_TO_LOAD}"
    --env_batch_size "${ENV_BATCH_SIZE}"
    --ego_value_head_lr "${EGO_VALUE_HEAD_LR}"
    --num_perturbs "${NUM_PERTURBS}"
    --use_mirror "${USE_MIRROR}"
    --ego_side "${EGO_SIDE}"
    --use_lr_annealing "${USE_LR_ANNEALING}"
    --lr_anneal_coeff "${LR_ANNEAL_COEFF}"
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"
    --training_batch_size "${TRAINING_BATCH_SIZE}"
    --num_env_steps "${NUM_ENV_STEPS}"
    --vtrace_enabled "${VTRACE_ENABLED}"
    --gamma "${GAMMA}"
    --vtrace_c_bar "${VTRACE_C_BAR}"
    --vtrace_rho_bar "${VTRACE_RHO_BAR}"
    --popart "${POPART}"
    --popart_beta "${POPART_BETA}"
    --reward_scale "${REWARD_SCALE}"
    --aggresive_coeff "${AGGRESIVE_COEFF}"
    --value_clip_separate "${VALUE_CLIP_SEPARATE}"
    --ego_style "${EGO_STYLE}"
    --adv_style "${ADV_STYLE}"
    --envs_per_matchup "${ENVS_PER_MATCHUP}"
    --side "${SIDE}"
    --render "${RENDER}"
    --model_file "${MODEL_FILE}"
    --use_wandb "${USE_WANDB}"
    --transform_action "${TRANSFORM_ACTION}"
    --async_update "${ASYNC_UPDATE}"
    --model_arch_type "${MODEL_ARCH_TYPE}"
    --dry_run "${DRY_RUN}"
)
if [ -n "${TRANCHE}" ]; then CMD+=(--tranche "${TRANCHE}"); fi
if [ -n "${OBS_TYPE}" ]; then CMD+=(--obs_type "${OBS_TYPE}"); fi
if [ -n "${RAM_MASK}" ]; then CMD+=(--ram_mask "${RAM_MASK}"); fi
if [ "${DISCOVER}" = "True" ]; then CMD+=(--discover); fi
if [ -n "${VTRACE_SEQ_LEN}" ]; then CMD+=(--vtrace_seq_len "${VTRACE_SEQ_LEN}"); fi
if [ -n "${BLEND_ADVERSARY_HEADS}" ]; then CMD+=(--blend_adversary_heads "${BLEND_ADVERSARY_HEADS}"); fi
"${CMD[@]}"
