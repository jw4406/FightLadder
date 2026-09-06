#!/bin/bash
# DIAGNOSTIC: Balrog MIRROR (Balrog vs Balrog) — a forced-footsie, no-projectile matchup
# to test whether richer play emerges faster than Vega (whose slide is a cheese attractor).
# Matches the ent05 v2 (egodecay) config so the only change is the CHARACTER.
# 6 envs x 1 matchup = 6 total envs (light on CPU); CHECKPOINT_INTERVAL=166667 keeps the
# ~1M-cumulative ckpt cadence (166667 x 6 ~= 1,000,002).
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
export MINIMAX_Q=False
export PLAYER=Balrog OPPONENTS="Balrog"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export ENVS_PER_MATCHUP=6
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=True LR_ANNEAL_COEFF=0.9998
export EGO_LR_ANNEAL_COEFF=0.9996
export ENT_COEF=0.05 DSTB_ENT_COEF=0.05
export CHARGE_PRESERVING_SKIP=True
export SEED=0 CHECKPOINT_INTERVAL=166667 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=BalrogMirror_dtj_ent05_egodecay9996_cps_e6_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
