#!/bin/bash
# FRESH ippo ent05 training: Vega vs {Guile, Blanka} POOL, 160M steps, joint DT,
# multiplicative LR decay (gamma=0.9998). Uniform LRs 3e-5 (ippo uses c_lr for all
# three optimizers; the driver auto-sets ego_value_head_lr=C_LR for non-spar). From
# SCRATCH. Matched to run_vega_ent05_spar_160M.sh for a spar-vs-ippo comparison.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh

export MODEL_ARCH=ippo
export PLAYER=Vega OPPONENTS="Guile Blanka"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export C_LR=3e-5 D_LR=3e-5 V_LR=3e-5 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=True LR_ANNEAL_COEFF=0.9998
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export SEED=0 CHECKPOINT_INTERVAL=80000 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=VegaGuileBlanka_dtj_ent05_160M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
