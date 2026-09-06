#!/bin/bash
# ARM 1 of 3: spar-all-constant (no LR annealing), charge_preserving_skip=True.
# Vega vs Guile+Blanka, joint decision timing, ent 0.05, 160M, ck1M-style.
# Companion arms: run_vega_ent05_spar_egodecayv2_160M.sh, run_vega_ent05_ippo_constant_160M.sh
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
export MINIMAX_Q=False
export PLAYER=Vega OPPONENTS="Guile Blanka"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export ENVS_PER_MATCHUP=12
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=False
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export CHARGE_PRESERVING_SKIP=True
export SEED=0 CHECKPOINT_INTERVAL=41667 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=VegaGuileBlanka_dtj_ent05_constant_cps_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
