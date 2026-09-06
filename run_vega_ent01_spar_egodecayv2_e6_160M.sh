#!/bin/bash
# DIAGNOSTIC 4th run: spar v2 (egodecay 0.9998/0.9996) with LOWER entropy (0.01 vs 0.05)
# to test whether reduced exploration cuts the jumpiness seen in the ent05 runs.
# ENVS_PER_MATCHUP=6 (half) to fit CPU alongside the 3 running ent05 trainings without
# oversubscribing 32 cores. CHECKPOINT_INTERVAL=83334 keeps the ~1M-cumulative ckpt
# cadence (6 envs x 2 matchups = 12 total envs; 83334 x 12 ~= 1,000,008).
set -u
PZ=/home/jw4406/codebase/FightLadder/run_minimax_phase0.sh
export MINIMAX_Q=False
export PLAYER=Vega OPPONENTS="Guile Blanka"
export OBS_TYPE=image REWARD_SCALE=1.0
export DECISION_TIMING=joint DWELL_FRAMES=4 ACTIONABLE_STATUSES=512,514,520 MAX_SKIP_FRAMES=90 NUM_STEP_FRAMES=8
export ENVS_PER_MATCHUP=6
export C_LR=3e-5 D_LR=1e-4 V_LR=4e-4 GAMMA=0.94 GAE_LAMBDA=0.95
export USE_LR_ANNEALING=True LR_ANNEAL_COEFF=0.9998
export EGO_LR_ANNEAL_COEFF=0.9996
export ENT_COEF=0.01 DSTB_ENT_COEF=0.01
export CHARGE_PRESERVING_SKIP=True
export SEED=0 CHECKPOINT_INTERVAL=83334 TOTAL_TIMESTEPS=160000000
export REINIT_EGO=False
export RUN_SUFFIX=VegaGuileBlanka_dtj_ent01_egodecay9996_cps_e6_160M_ck1M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$PZ" vtoff
