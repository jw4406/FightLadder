#!/bin/bash
# ARM 3 of 3: ippo, all-constant LRs (all 3e-5, no annealing), charge_preserving_skip=True.
# MODEL_ARCH=ippo auto-sets ego_value_head_lr=c_lr (3e-5).
# Vega vs Guile+Blanka, joint decision timing, ent 0.05, 160M, ck1M-style.
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
export MODEL_ARCH=ippo
export PLAYER=Vega OPPONENTS="Guile Blanka"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export ENVS_PER_MATCHUP=12
export C_LR=3e-5 D_LR=3e-5 V_LR=3e-5 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=False
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export CHARGE_PRESERVING_SKIP=True
export SEED=0 CHECKPOINT_INTERVAL=41667 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=VegaGuileBlanka_dtj_ent05_constant_cps_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
